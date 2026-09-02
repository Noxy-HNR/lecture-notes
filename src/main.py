"""
Lecture Notes App
==================
Figures out what class you're in based on your schedule, records + transcribes
the lecture locally (Whisper), and turns it into clean notes appended to that
class's ongoing notes file (Claude Code CLI / API when available, local
fallback otherwise). Optionally detects different speakers (Q&A) via
pyannote.audio diarization if it's installed and configured.

Audio capture runs on a dedicated background thread (see capture.py), decoupled from
transcription/saving - no matter how long a save/diarization/API call takes, capture
itself never stalls, so the OS audio buffer never gets a chance to silently overflow
and drop audio. This replaced an earlier single-threaded design after a real, confirmed
bug: a slow step blocking the main loop caused audio to be silently lost mid-lecture.

Usage:
    python src/main.py                 # auto-detect class from schedule, prompt for audio source
    python src/main.py --class "BIOL 1440"   # override class selection
    python src/main.py --source mic    # skip the audio-source prompt (mic | system)
    python src/main.py --chunk 15      # seconds per transcription chunk (default 15, tuned for
                                        # accuracy - more context per chunk. Lower it for more
                                        # frequent live output at a slight accuracy cost)
    python src/main.py --list          # list all classes in the schedule and exit

Stop recording any time with Ctrl+C. Notes are saved (with a safety autosave
every few minutes) even if you stop abruptly.
"""
import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import schedule as sched
import audio
import capture as capture_module
import transcribe
import notes
import docx_export
import diarize
import vocab as vocab_module
import soundfile as sf
import console_colors as c
import keep_awake

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_DIR.mkdir(exist_ok=True)

AUTOSAVE_EVERY_SECONDS = 5 * 60  # flush partial notes periodically, not just at the end


def choose_class(args):
    if args.klass:
        cls = sched.get_class_by_code(args.klass)
        if not cls:
            print(f"No class found matching '{args.klass}'. Known classes:")
            for code, title in sched.list_all_classes():
                print(f"  {code} - {title}")
            sys.exit(1)
        return {"code": cls["code"], "title": cls["title"], "status": "manual"}

    detected = sched.find_current_class()
    if detected:
        status_msg = {
            "in_session": "currently in session",
            "starting_soon": "starting soon",
            "just_ended": "just ended",
        }[detected["status"]]
        print(c.success(f"Detected class from your schedule: {detected['code']} - {detected['title']} "
                         f"({detected['location']}, {status_msg})"))
        return detected

    print(c.warning("No class found in your schedule for right now."))
    print("Known classes:")
    codes = sched.list_all_classes()
    for i, (code, title) in enumerate(codes, 1):
        print(f"  {i}. {code} - {title}")
    choice = input("Pick a number to transcribe anyway, or press Enter to quit: ").strip()
    if not choice:
        sys.exit(0)
    try:
        code, title = codes[int(choice) - 1]
        return {"code": code, "title": title, "status": "manual"}
    except (ValueError, IndexError):
        print(c.error("Invalid choice."))
        sys.exit(1)


def choose_source(args):
    if args.source in ("mic", "system"):
        return args.source
    print("Audio source:")
    print("  1. Microphone (in-person lecture)")
    print("  2. System audio / loopback (online lecture, e.g. Zoom)")
    choice = input("Choose [1]: ").strip() or "1"
    return "mic" if choice == "1" else "system"


def choose_formatting_mode(args):
    if args.formatting in notes.FORMATTING_MODES:
        return args.formatting
    print("Note formatting:")
    print("  1. Auto - Claude CLI/API when available, falls back to local GPU model then heuristic (default)")
    print("  2. Local only - local GPU model + heuristic only, no network calls at all (CLI/API skipped)")
    print("  3. Heuristic only - no LLM anywhere, fastest and fully deterministic")
    choice = input("Choose [1]: ").strip() or "1"
    return {"1": "auto", "2": "local", "3": "heuristic"}.get(choice, "auto")


class RollingTranscriber:
    """Wraps chunked transcription with two accuracy improvements over transcribing
    each chunk in total isolation:
      - a small audio overlap between chunks, so a word split across a chunk
        boundary doesn't get clipped/mis-heard (segments fully inside the overlap
        window are dropped as already-emitted, since they were transcribed last call)
      - a rolling text prompt (recent transcript + class vocab) fed back in as
        Whisper's initial_prompt, giving it continuity across chunks instead of
        transcribing each one cold
    """

    OVERLAP_SECONDS = 1.5
    CONTEXT_CHARS = 200

    def __init__(self, vocab_prompt: str | None):
        self.vocab_prompt = vocab_prompt
        self.overlap_audio = np.zeros(0, dtype=np.float32)
        self.recent_text = ""
        self.consecutive_silent_chunks = 0

    def process(self, new_chunk: np.ndarray) -> list[dict]:
        """Returns segments (start/end/text) with timestamps relative to `new_chunk`,
        ready to pass to Session.write_chunk alongside it."""
        overlap_duration = len(self.overlap_audio) / audio.SAMPLE_RATE
        combined = np.concatenate([self.overlap_audio, new_chunk])

        if audio.is_silent(combined):
            # Skip Whisper entirely on (near-)silence - otherwise it tends to hallucinate
            # repeated punctuation/filler ("...", "you") rather than emitting nothing,
            # which is what happens if the mic gets muted/disconnected or the "system
            # audio" source goes quiet while the app keeps running unattended.
            self.consecutive_silent_chunks += 1
            overlap_frames = int(self.OVERLAP_SECONDS * audio.SAMPLE_RATE)
            self.overlap_audio = new_chunk[-overlap_frames:].copy() if len(new_chunk) > overlap_frames else new_chunk.copy()
            return []
        self.consecutive_silent_chunks = 0

        prompt = " ".join(p for p in (self.vocab_prompt, self.recent_text) if p) or None
        segments = transcribe.transcribe_chunk_segments(combined, initial_prompt=prompt)

        kept = []
        for seg in segments:
            if seg["end"] <= overlap_duration:
                continue  # entirely inside the overlap window - already emitted last call
            kept.append({
                "start": max(0.0, seg["start"] - overlap_duration),
                "end": seg["end"] - overlap_duration,
                "text": seg["text"],
            })

        if kept:
            new_text = " ".join(s["text"] for s in kept)
            self.recent_text = (self.recent_text + " " + new_text)[-self.CONTEXT_CHARS:]

        overlap_frames = int(self.OVERLAP_SECONDS * audio.SAMPLE_RATE)
        self.overlap_audio = new_chunk[-overlap_frames:].copy() if len(new_chunk) > overlap_frames else new_chunk.copy()

        return kept


class Session:
    """Tracks the running WAV recording + transcript segments for one lecture. Autosaves
    just flush plain transcript text (no diarization - see pop_pending_text for why);
    only the final save at the end of the session hands off audio/text to diarization."""

    def __init__(self, wav_path):
        self.wav_path = wav_path
        self._sf = sf.SoundFile(str(wav_path), mode="w", samplerate=audio.SAMPLE_RATE,
                                 channels=1, subtype="FLOAT")
        self.total_frames = 0
        self.last_save_frame = 0
        self.last_save_elapsed = 0.0
        self.pending_segments = []  # [{"start","end","text"}] absolute session-elapsed seconds
        self.session_start = time.time()

    def elapsed(self):
        return time.time() - self.session_start

    def write_chunk(self, chunk_audio, whisper_segments):
        chunk_offset = self.total_frames / audio.SAMPLE_RATE
        self._sf.write(chunk_audio)
        self.total_frames += len(chunk_audio)
        for seg in whisper_segments:
            self.pending_segments.append({
                "start": chunk_offset + seg["start"],
                "end": chunk_offset + seg["end"],
                "text": seg["text"],
            })

    def pop_pending_text(self, run_diarization: bool = False) -> str:
        """Returns speaker-labeled (if diarization succeeds and run_diarization=True) or
        plain transcript text for everything recorded since the last save, and resets
        the save window.

        run_diarization defaults to False deliberately: the whole app is single-threaded,
        so running diarization here blocks audio capture for however long it takes - on
        a real multi-minute autosave window that was found to be minutes long, silently
        losing whatever was actually said during the gap. Diarization is not real-time
        (pyannote needs a complete clip to compute speaker segments, it can't label
        speakers incrementally), so there was never a live-transcription benefit being
        traded away here - only autosaves being pointlessly slow and lossy. Callers
        should only pass True for the final save at the end of a session, after
        recording has already stopped."""
        if not self.pending_segments:
            return ""

        window_start_sec = self.last_save_elapsed
        window_start_frame = self.last_save_frame
        window_end_frame = self.total_frames

        text = " ".join(seg["text"] for seg in self.pending_segments)

        if run_diarization and diarize.available() and window_end_frame > window_start_frame:
            self._sf.flush()
            clip_path = self.wav_path.with_name(self.wav_path.stem + "_clip.wav")
            try:
                with sf.SoundFile(str(self.wav_path), mode="r") as full:
                    full.seek(window_start_frame)
                    clip_data = full.read(window_end_frame - window_start_frame, dtype="float32")
                sf.write(str(clip_path), clip_data, audio.SAMPLE_RATE, subtype="FLOAT")

                diar_segments = diarize.diarize(clip_path)
                if diar_segments:
                    relative_segments = [
                        {"start": s["start"] - window_start_sec, "end": s["end"] - window_start_sec,
                         "text": s["text"]}
                        for s in self.pending_segments
                    ]
                    labeled = diarize.assign_speakers(relative_segments, diar_segments)
                    labeled_text = diarize.to_labeled_transcript(labeled)
                    if labeled_text.strip():
                        text = labeled_text
            except Exception:
                pass  # diarization is best-effort; fall back to the plain joined text
            finally:
                clip_path.unlink(missing_ok=True)

        self.last_save_frame = self.total_frames
        self.last_save_elapsed = self.elapsed()
        self.pending_segments = []
        return text

    def close(self):
        self._sf.close()


MIN_FREE_DISK_GB = 2.0


def run_preflight_checks(source: str) -> bool:
    """Quick sanity checks before committing to a recording session, so a broken mic
    or a dead GPU shows up now - not silently mid-lecture, which is how most of this
    app's real bugs actually surfaced. Warnings don't block starting; only a genuinely
    unusable audio device or a Whisper model that fails to load does (returns False)."""
    print(c.heading("Running preflight checks..."))
    ok = True

    try:
        test_recorder = audio.get_recorder(source)
        with test_recorder:
            test_clip = audio.record_chunk(test_recorder, 0.5)
        peak = float(np.max(np.abs(test_clip))) if test_clip.size else 0.0
        source_label = "microphone" if source == "mic" else "system audio"
        if audio.is_silent(test_clip):
            print(c.warning(f"  Audio device opened, but picked up only silence (peak {peak:.4f}) - "
                             f"check your {source_label} is unmuted/active before you start talking."))
        else:
            print(c.success(f"  Audio device OK ({source_label}, peak level {peak:.3f})"))
    except Exception as e:
        print(c.error(f"  Audio device FAILED to open: {e}"))
        ok = False

    try:
        usage = shutil.disk_usage(STATE_DIR)
        free_gb = usage.free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            print(c.warning(f"  Low disk space: {free_gb:.1f}GB free - audio/notes backups need room."))
        else:
            print(c.success(f"  Disk space OK ({free_gb:.1f}GB free)"))
    except Exception:
        pass  # non-critical, skip silently if the check itself fails

    try:
        transcribe.get_model()
        print(c.success(f"  Whisper model OK ({transcribe.device_info()})"))
    except Exception as e:
        print(c.error(f"  Whisper model FAILED to load: {e}"))
        ok = False

    if diarize.available():
        print(c.info("  Speaker diarization: enabled (HUGGINGFACE_TOKEN set)"))
    else:
        print(c.dim("  Speaker diarization: off (set HUGGINGFACE_TOKEN to enable Q&A speaker labeling)"))

    print()
    return ok


def _resume_backup_paths(given: Path) -> tuple[Path, Path, str, str]:
    """Given either a *_raw.txt or *.wav backup path from state/, returns
    (raw_log_path, wav_path, class_code, session_date) - the sibling backup file (which
    may or may not exist) and the class/date parsed from the shared filename stem
    ("{CODE}_{YYYYMMDD_HHMMSS}")."""
    name = given.name
    if name.endswith("_raw.txt"):
        base = name[: -len("_raw.txt")]
    elif name.endswith(".wav"):
        base = name[: -len(".wav")]
    else:
        raise ValueError(f"Not a recognized backup filename: {name}")

    m = re.match(r"^(.+)_(\d{8}_\d{6})$", base)
    if not m:
        raise ValueError(f"Couldn't parse class code/timestamp from filename: {name}")
    code_slug, ts = m.groups()
    class_code = code_slug.replace("_", " ")
    session_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
    session_date = f"{session_dt.strftime('%A, %B')} {session_dt.day}, {session_dt.strftime('%Y')}"

    directory = given.parent
    return directory / f"{base}_raw.txt", directory / f"{base}.wav", class_code, session_date


def _find_date_section_boundary(content: str, session_date: str) -> int:
    """Returns the byte offset where `session_date`'s "## <date>" section(s) first
    appear in `content`. Used so a resume's condense pass only touches today's content
    (any pre-crash autosaves plus the newly recovered tail) and leaves every other
    lecture date in the file untouched. Returns len(content) if that date doesn't
    appear yet (nothing to protect - the recovered section will be the first)."""
    for m in re.finditer(r"^## (.+)$", content, re.MULTILINE):
        if m.group(1).strip() == session_date:
            return m.start()
    return len(content)


def list_session_backups():
    backups = sorted(STATE_DIR.glob("*_raw.txt"))
    if not backups:
        print("No session backups found in state/.")
        return
    print("Session backups available to resume:")
    for raw_path in backups:
        wav_path = raw_path.parent / raw_path.name.replace("_raw.txt", ".wav")
        audio_note = f" (+ audio: {wav_path.name})" if wav_path.exists() else " (no audio backup)"
        print(f"  {raw_path.name}{audio_note}")


def run_resume(target: str, args):
    """Recovers a crashed/interrupted session from its state/ backups: re-transcribes
    the audio backup if present (enabling diarization, since resuming only happens
    after recording has already stopped), or falls back to the raw transcript log text
    if the audio backup is missing, then formats and saves it exactly like a normal
    final save - including condensing against any content already saved for that same
    date (e.g. from autosaves that succeeded before the crash)."""
    given = Path(target)
    if not given.exists():
        print(c.error(f"Backup not found: {given}"))
        sys.exit(1)

    try:
        raw_log_path, wav_path, class_code, session_date = _resume_backup_paths(given)
    except ValueError as e:
        print(c.error(str(e)))
        sys.exit(1)

    cls_info = sched.get_class_by_code(class_code)
    class_title = cls_info["title"] if cls_info else class_code

    formatting_mode = choose_formatting_mode(args)
    print(c.heading(f"\nRecovering session for {class_code} - {class_title} ({session_date})\n"))

    transcript_text = None
    if wav_path.exists():
        print(c.info(f"Found audio backup ({wav_path.name}) - re-transcribing so diarization "
                      f"can run on it (safe now: recording has already stopped)..."))
        transcribe.get_model()
        initial_prompt = vocab_module.initial_prompt_for_class(class_code)
        transcriber = RollingTranscriber(initial_prompt)

        with sf.SoundFile(str(wav_path), mode="r") as f:
            data = f.read(len(f), dtype="float32")

        chunk_frames = max(1, int(args.chunk * audio.SAMPLE_RATE))
        all_segments = []
        offset_seconds = 0.0
        for pos in range(0, len(data), chunk_frames):
            piece = data[pos: pos + chunk_frames]
            for seg in transcriber.process(piece):
                all_segments.append({
                    "start": offset_seconds + seg["start"],
                    "end": offset_seconds + seg["end"],
                    "text": seg["text"],
                })
            offset_seconds += len(piece) / audio.SAMPLE_RATE

        transcript_text = " ".join(s["text"] for s in all_segments)

        if diarize.available() and all_segments:
            print(c.info("Running speaker diarization on the recovered audio..."))
            diar_segments = diarize.diarize(wav_path)
            if diar_segments:
                labeled = diarize.assign_speakers(all_segments, diar_segments)
                labeled_text = diarize.to_labeled_transcript(labeled)
                if labeled_text.strip():
                    transcript_text = labeled_text
    elif raw_log_path.exists():
        print(c.warning(f"No audio backup found - recovering from the raw transcript log only "
                         f"({raw_log_path.name}); no diarization possible without audio."))
        parts = []
        for line in raw_log_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s?(.*)$", line)
            parts.append(m.group(1) if m else line)
        transcript_text = " ".join(p for p in parts if p.strip())
    else:
        print(c.error(f"Neither {wav_path.name} nor {raw_log_path.name} exist - nothing to recover."))
        sys.exit(1)

    if not transcript_text or not transcript_text.strip():
        print(c.warning("Nothing was transcribed from the backup - nothing to save."))
        return

    existing_notes_path = notes.notes_path(class_code)
    existing_content = existing_notes_path.read_text(encoding="utf-8") if existing_notes_path.exists() else ""
    boundary = _find_date_section_boundary(existing_content, session_date)

    path, method = notes.format_and_save(class_code, class_title, transcript_text,
                                          session_date, mode=formatting_mode)
    print(c.success(f"Recovered transcript formatted and saved (method: {method})."))

    if formatting_mode == "auto" and notes.cli_available():
        print(c.info("Condensing recovered notes against any earlier autosaves from the same date..."))
        ok, reason = notes.condense_session(class_code, class_title, boundary, mode=formatting_mode)
        if ok:
            print(c.success("Recovered notes condensed and cleaned up."))
        else:
            print(c.warning(f"Skipped condense: {reason}"))
    elif formatting_mode != "auto":
        print(c.dim(f"Skipped condense: formatting mode is '{formatting_mode}' (no CLI/API calls)."))
    else:
        print(c.dim("Skipped condense: Claude CLI not available."))

    print(c.info(f"Notes (markdown): {notes.notes_path(class_code)}"))
    print(c.info(f"Notes (Word): {docx_export.docx_path(class_code)}"))


def run():
    parser = argparse.ArgumentParser(description="Auto-transcribe your current lecture into notes.")
    parser.add_argument("--class", dest="klass", help="Override class code, e.g. 'BIOL 1440'")
    parser.add_argument("--source", choices=["mic", "system"], help="Audio source, skips the prompt")
    parser.add_argument("--chunk", type=float, default=15.0, help="Seconds per transcription chunk")
    parser.add_argument("--formatting", choices=notes.FORMATTING_MODES,
                         help="Note formatting mode, skips the prompt (auto/local/heuristic)")
    parser.add_argument("--list", action="store_true", help="List classes from schedule.json and exit")
    parser.add_argument("--list-sessions", action="store_true",
                         help="List session backups in state/ available to --resume")
    parser.add_argument("--resume", metavar="PATH",
                         help="Recover a crashed/interrupted session from its state/ backup "
                              "(.wav or _raw.txt path) instead of recording")
    args = parser.parse_args()

    if args.list:
        for code, title in sched.list_all_classes():
            print(f"{code} - {title}")
        return

    if args.list_sessions:
        list_session_backups()
        return

    if args.resume:
        run_resume(args.resume, args)
        return

    cls = choose_class(args)
    source = choose_source(args)
    formatting_mode = choose_formatting_mode(args)
    initial_prompt = vocab_module.initial_prompt_for_class(cls["code"])

    print()
    if not run_preflight_checks(source):
        print(c.error("Preflight checks failed - fix the issue above before recording "
                       "(this prevents starting a session that would silently fail)."))
        sys.exit(1)

    mode_descriptions = {
        "auto": "auto (Claude CLI/API -> local GPU model -> heuristic)",
        "local": "local only (GPU model -> heuristic, no network calls)",
        "heuristic": "heuristic only (no LLM anywhere)",
    }
    print(c.info(f"Note formatting: {mode_descriptions[formatting_mode]}"))

    print(c.heading(f"\nRecording from {'microphone' if source == 'mic' else 'system audio'} for "
                     f"{cls['code']} - {cls['title']}. Press Ctrl+C to stop and save notes.\n"))

    now = datetime.now()
    session_date = f"{now.strftime('%A, %B')} {now.day}, {now.strftime('%Y')}"
    last_autosave = time.time()

    existing_notes_path = notes.notes_path(cls["code"])
    session_start_offset = len(existing_notes_path.read_text(encoding="utf-8")) if existing_notes_path.exists() else 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_path = STATE_DIR / f"{cls['code'].replace(' ', '_')}_{ts}_raw.txt"
    wav_path = STATE_DIR / f"{cls['code'].replace(' ', '_')}_{ts}.wav"
    session = Session(wav_path)
    transcriber = RollingTranscriber(initial_prompt)
    silence_warned = False
    SILENCE_WARNING_SECONDS = 120  # warn once if this much continuous silence is seen

    keep_awake.prevent_sleep()  # a system sleep mid-recording silently kills the whole
                                 # process, not just the display - block that for the
                                 # duration of the session, released in the finally below

    # Audio capture runs on its own thread, continuously draining the OS buffer into a
    # queue - decoupled from transcription/saving below, so no matter how long any
    # processing step takes (a slow API call, diarization, anything), capture itself
    # never stalls and the OS buffer never gets a chance to silently overflow and drop
    # audio. See capture.py for the full story (this was a real, confirmed bug).
    capture = capture_module.CaptureThread(source)
    capture.start()

    try:
        while True:
            chunk_audio = capture.collect_chunk(args.chunk)
            for err in capture.poll_errors():
                print(c.error(f"Audio device error (auto-recovering, capture continues): {err}"))
            if chunk_audio is None:
                continue  # nothing captured yet (e.g. very start) - keep waiting

            whisper_segments = transcriber.process(chunk_audio)
            session.write_chunk(chunk_audio, whisper_segments)

            silent_seconds = transcriber.consecutive_silent_chunks * args.chunk
            if silent_seconds >= SILENCE_WARNING_SECONDS and not silence_warned:
                print(c.warning(f"No audio detected for ~{silent_seconds/60:.1f} min - "
                                 f"check your {'microphone' if source == 'mic' else 'system audio'} "
                                 f"source (muted? disconnected?)."))
                silence_warned = True
            elif transcriber.consecutive_silent_chunks == 0:
                silence_warned = False

            for seg in whisper_segments:
                stamp = time.strftime("%H:%M:%S")
                print(f"{c.timestamp('[' + stamp + ']')} {c.transcript(seg['text'])}")
                with open(raw_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{stamp}] {seg['text']}\n")

            if time.time() - last_autosave > AUTOSAVE_EVERY_SECONDS and session.pending_segments:
                _save(cls, session, session_date, formatting_mode, note="(autosave)")
                last_autosave = time.time()
    except KeyboardInterrupt:
        print(c.autosave("\nStopping - wrapping up your notes now, this can take a moment..."))
    finally:
        shutdown_start = time.time()

        def _step(msg):
            print(c.heading(f"[{time.time() - shutdown_start:5.1f}s] ") + c.info(msg))

        keep_awake.allow_sleep()

        _step("Stopping audio capture...")
        capture.stop()
        print(c.dim("  capture thread stopped."))

        tail_audio = capture.drain_available()
        if tail_audio is not None and len(tail_audio) > 0:
            tail_seconds = len(tail_audio) / audio.SAMPLE_RATE
            _step(f"Transcribing final {tail_seconds:.1f}s of buffered audio...")
            whisper_segments = transcriber.process(tail_audio)
            session.write_chunk(tail_audio, whisper_segments)
            for seg in whisper_segments:
                stamp = time.strftime("%H:%M:%S")
                print(f"{c.timestamp('[' + stamp + ']')} {c.transcript(seg['text'])}")
                with open(raw_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{stamp}] {seg['text']}\n")
            print(c.dim(f"  {len(whisper_segments)} segment(s) transcribed from the tail."))

        if session.pending_segments:
            n_pending = len(session.pending_segments)
            if diarize.available():
                _step(f"Running speaker diarization on the final segment ({n_pending} transcribed "
                      f"chunk(s) pending) - this is the slow part, can take a while on a long segment...")
            else:
                _step(f"Formatting and saving the final segment ({n_pending} transcribed chunk(s))...")
            save_start = time.time()
            _save(cls, session, session_date, formatting_mode, is_final=True)
            print(c.dim(f"  final save completed in {time.time() - save_start:.1f}s."))
        else:
            _step("Nothing pending to save (already flushed by the last autosave).")
        session.close()

        elapsed_min = session.elapsed() / 60
        print(c.success(f"\nRecording done - captured ~{elapsed_min:.1f} minutes of lecture."))

        if formatting_mode != "auto":
            print(c.dim(f"Skipped condense pass: formatting mode is '{formatting_mode}' (no CLI/API calls)."))
        elif notes.cli_available():
            _step("Reviewing this session's notes with Claude CLI (condensing duplicate autosave "
                  "sections, cleaning up formatting)...")
            condense_start = time.time()
            ok, reason = notes.condense_session(cls["code"], cls["title"], session_start_offset, mode=formatting_mode)
            if ok:
                print(c.success(f"  Notes condensed and cleaned up ({time.time() - condense_start:.1f}s)."))
            else:
                print(c.warning(f"  Skipped condense pass: {reason}"))
        else:
            print(c.dim("Skipped condense pass: Claude CLI not available."))

        total_shutdown = time.time() - shutdown_start
        print(c.heading(f"\nAll done ({total_shutdown:.1f}s from Ctrl+C to finish)."))
        print(c.info(f"Notes (markdown): {notes.notes_path(cls['code'])}"))
        print(c.info(f"Notes (Word): {docx_export.docx_path(cls['code'])}"))
        print(c.dim(f"Raw transcript backup: {raw_log_path}"))
        print(c.dim(f"Audio backup: {wav_path}"))


def _save(cls, session: Session, session_date, mode, note="", is_final=False):
    full_text = session.pop_pending_text(run_diarization=is_final)
    path, method = notes.format_and_save(cls["code"], cls["title"], full_text, session_date, mode=mode)
    chose_this_mode = mode != "auto"
    tag, colorize = {
        "cli": (" [formatted with Claude Code CLI]", c.success),
        "api": (" [formatted with Claude API]", c.success),
        "gpu": (" [formatted with local GPU model]" if chose_this_mode
                else " [formatted with local GPU model - Claude CLI/API unavailable]", c.warning),
        "local": (" [heuristic local formatting]" if chose_this_mode
                  else " [heuristic local formatting - Claude CLI/API/GPU model unavailable]", c.warning),
        "none": ("", c.dim),
    }[method]
    note_colored = c.autosave(note) if note else ""
    print(colorize(f"Saved notes to {path}{tag}") + (f" {note_colored}" if note_colored else ""))


if __name__ == "__main__":
    run()

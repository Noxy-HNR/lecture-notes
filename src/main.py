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
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
import gpu_formatter
import vocab as vocab_module
import soundfile as sf
import console_colors as c
import keep_awake
import mic_mute

try:
    import msvcrt  # Windows-only stdlib module for non-blocking console keypress detection
except ImportError:
    msvcrt = None

# Windows consoles default to a legacy codepage (e.g. cp1252), not UTF-8. Notes/flashcard
# text is LLM-generated and routinely contains characters outside that range (en dashes,
# arrows, "≈", degree signs, etc.) - printing one crashes with UnicodeEncodeError. Confirmed
# live: a flashcard explanation containing "≈" killed a quiz mid-session. Reconfigure stdout/
# stderr to UTF-8 unconditionally so any such content just prints correctly.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_DIR.mkdir(exist_ok=True)

AUTOSAVE_EVERY_SECONDS = 5 * 60  # flush partial notes periodically, not just at the end
SAVE_NOW_KEY = b"s"


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


def choose_action(args) -> str:
    """Top-level startup menu, shown whenever no flag already decided what to do (--list,
    --resume, --study-guide, --flashcards, etc. all return before this runs). Lets "generate
    a study guide"/"quiz me" be picked interactively instead of only being reachable via flags."""
    print(c.heading("What would you like to do?"))
    print("  1. Record a lecture (default)")
    print("  2. Generate a study guide from a class's notes so far")
    print("  3. Quiz yourself with flashcards")
    choice = input("Choose [1]: ").strip() or "1"
    return {"2": "study_guide", "3": "flashcards"}.get(choice, "record")


def choose_notes_class(args) -> str:
    if args.klass:
        return args.klass
    codes = sched.list_all_classes()
    print("Which class?")
    for i, (code, title) in enumerate(codes, 1):
        print(f"  {i}. {code} - {title}")
    choice = input("Choose a number: ").strip()
    try:
        code, _ = codes[int(choice) - 1]
        return code
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


class SaveWorker:
    """Runs the slow part of a save (notes.format_and_save, which can block for however
    long a Claude CLI/API call takes - up to CLI_TIMEOUT_SECONDS = 180s) on a background
    thread, so an autosave doesn't stall the main loop from pulling new audio off the
    capture queue and printing/transcribing it. Confirmed live: the main loop visibly
    stopped producing new transcript lines for the whole duration of a slow autosave -
    audio itself was never lost (capture.py's own thread keeps buffering regardless of
    what the main loop is doing), but live output and the save itself both stalled.

    Jobs run strictly one at a time, in submission order (a plain queue + a single
    worker thread, not a thread pool) - two saves running concurrently could interleave
    writes to the same notes file and corrupt it. wait_idle() blocks until every
    submitted job has finished; call it before anything that reads the notes file
    (the final Ctrl+C save, full-session diarization's truncation, the condense pass)
    so none of them can run against a file a still-in-flight autosave hasn't finished
    writing to yet."""

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="save-worker")
        self._thread.start()

    def _run(self):
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception as e:
                print(c.error(f"Background save failed: {e}"))
            finally:
                self._queue.task_done()

    def submit(self, job):
        self._queue.put(job)

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def wait_idle(self):
        self._queue.join()


class TranscriptionWorker:
    """Runs RollingTranscriber.process() (the blocking Whisper inference call) on a
    background thread, so the main loop is never stuck inside it - it can immediately go
    back to the interruptible capture.collect_chunk() wait, instead of only checking for
    a Ctrl+C/save-now keypress once the current chunk's transcription happens to finish.
    Confirmed live: that could add several real seconds of extra delay before a
    keypress registered, on top of the resolved-elsewhere autosave stall.

    Chunks are processed strictly one at a time, in submission order (single worker
    thread + ordered queue, not a thread pool) - RollingTranscriber keeps rolling state
    (overlap audio, recent text, silence counter) across calls that only makes sense if
    calls happen in the same order the audio was captured; two chunks processed out of
    order, or concurrently, would corrupt that state.

    `on_result(chunk_audio, whisper_segments)` runs on the worker thread for each chunk,
    in order - the caller is expected to do its printing/session-writing/silence-check
    there. Since only this one thread ever calls it, none of that needs its own lock."""

    def __init__(self, transcriber: "RollingTranscriber", on_result):
        self._transcriber = transcriber
        self._on_result = on_result
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="transcription-worker")
        self._thread.start()

    def _run(self):
        while True:
            chunk_audio = self._queue.get()
            try:
                whisper_segments = self._transcriber.process(chunk_audio)
                self._on_result(chunk_audio, whisper_segments)
            except Exception as e:
                print(c.error(f"Transcription failed for a chunk (continuing): {e}"))
            finally:
                self._queue.task_done()

    def submit(self, chunk_audio):
        self._queue.put(chunk_audio)

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def wait_idle(self):
        self._queue.join()


class SaveNowListener:
    """Background thread that watches for a keypress (Windows console only, via
    msvcrt - no Enter needed, a raw keystroke is enough) to request an immediate
    manual save. Deliberately does nothing but set an event: the actual save always
    happens on the main loop, same as autosave, so there's no risk of the listener
    thread and the main loop both touching Session/notes state at the same time.
    A no-op (start() does nothing) if msvcrt isn't available (non-Windows)."""

    def __init__(self, key: bytes = SAVE_NOW_KEY):
        self.key = key.lower()
        self.requested = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="save-now-listener", daemon=True)

    def start(self):
        if msvcrt is None:
            return
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch.lower() == self.key:
                    self.requested.set()
            time.sleep(0.1)


class Session:
    """Tracks the running WAV recording + transcript segments for one lecture. Autosaves
    just flush plain transcript text (no diarization - see pop_pending_text for why);
    the final save at session end can instead re-diarize the WHOLE session (see
    run()'s shutdown sequence) using all_segments, which - unlike pending_segments -
    is never cleared, so it always has the complete session's transcript available.

    write_chunk() and pop_pending_text() can now run on different threads at the same
    time (chunk transcription moved to a background TranscriptionWorker, and an autosave
    can fire from the main loop while a later chunk is still being written) - _lock
    protects every read/mutation of the shared frame counters and segment lists so the
    two never corrupt each other's view of the session."""

    def __init__(self, wav_path):
        self.wav_path = wav_path
        self._sf = sf.SoundFile(str(wav_path), mode="w", samplerate=audio.SAMPLE_RATE,
                                 channels=1, subtype="FLOAT")
        self._lock = threading.Lock()
        self.total_frames = 0
        self.last_save_frame = 0
        self.last_save_elapsed = 0.0
        self.pending_segments = []  # [{"start","end","text"}] since the last save - cleared on save
        self.all_segments = []      # every segment for the whole session - never cleared
        self.session_start = time.time()

    def elapsed(self):
        return time.time() - self.session_start

    def flush(self):
        self._sf.flush()

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self.pending_segments)

    def write_chunk(self, chunk_audio, whisper_segments):
        with self._lock:
            chunk_offset = self.total_frames / audio.SAMPLE_RATE
            self._sf.write(chunk_audio)
            self.total_frames += len(chunk_audio)
            for seg in whisper_segments:
                entry = {
                    "start": chunk_offset + seg["start"],
                    "end": chunk_offset + seg["end"],
                    "text": seg["text"],
                }
                self.pending_segments.append(entry)
                self.all_segments.append(entry)

    def pop_pending_text(self, run_diarization: bool = False) -> str:
        """Returns speaker-labeled (if diarization succeeds and run_diarization=True) or
        plain transcript text for everything recorded since the last save, and resets
        the save window.

        Only the quick part (snapshotting pending segments, resetting the window) holds
        _lock - diarization (when run_diarization=True) does slow file I/O and GPU
        inference against a local snapshot instead, so it never blocks a concurrent
        write_chunk() call from a still-running TranscriptionWorker. In practice that
        never actually happens today: run_diarization=True is only ever passed for the
        final save, by which point the caller has already drained the transcription
        worker - but the snapshot-then-release pattern keeps this correct regardless of
        when it's called, not just under today's call sites.

        run_diarization defaults to False deliberately: diarization needs a complete
        clip to compute speaker segments (it can't label speakers incrementally), so
        there's no live-transcription benefit to running it on every autosave - only
        cost. Callers should only pass True for the final save at the end of a session,
        after recording has already stopped."""
        with self._lock:
            if not self.pending_segments:
                return ""
            window_start_sec = self.last_save_elapsed
            window_start_frame = self.last_save_frame
            window_end_frame = self.total_frames
            pending_snapshot = self.pending_segments
            text = " ".join(seg["text"] for seg in pending_snapshot)
            self.last_save_frame = self.total_frames
            self.last_save_elapsed = self.elapsed()
            self.pending_segments = []

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
                        for s in pending_snapshot
                    ]
                    labeled = diarize.assign_speakers(relative_segments, diar_segments)
                    labeled_text = diarize.to_labeled_transcript(labeled)
                    if labeled_text.strip():
                        text = labeled_text
            except Exception:
                pass  # diarization is best-effort; fall back to the plain joined text
            finally:
                clip_path.unlink(missing_ok=True)

        return text

    def close(self):
        self._sf.close()


MIN_FREE_DISK_GB = 2.0
PREFLIGHT_AUDIO_SAMPLE_SECONDS = 3.0  # long enough that a natural pause between words
                                       # doesn't get mistaken for a dead/muted mic


def _check_audio_device(source: str) -> tuple[bool, str]:
    """Returns (blocks_startup, message). Runs concurrently with the other preflight
    checks - safe because it only touches the audio subsystem (WASAPI), no shared
    state with the GPU/Whisper or filesystem checks."""
    try:
        test_recorder = audio.get_recorder(source)
        with test_recorder:
            test_clip = audio.record_chunk(test_recorder, PREFLIGHT_AUDIO_SAMPLE_SECONDS)
        peak = float(np.max(np.abs(test_clip))) if test_clip.size else 0.0
        source_label = "microphone" if source == "mic" else "system audio"
        # Check sub-windows individually rather than the RMS of the whole sample - a
        # single silent instant during a longer sample would otherwise dilute a real,
        # brief utterance below the threshold even though speech was genuinely present.
        # (A too-short single sample was a real false-positive bug: on 0.5s, roughly a
        # coin flip whether it landed on a natural pause between words, even during
        # active conversation - see git history if this needs adjusting again.)
        window_frames = int(0.5 * audio.SAMPLE_RATE)
        any_sound = any(
            not audio.is_silent(test_clip[i:i + window_frames])
            for i in range(0, len(test_clip), window_frames)
        )
        if not any_sound:
            return False, c.warning(f"  Audio device opened, but picked up only silence over "
                                     f"{PREFLIGHT_AUDIO_SAMPLE_SECONDS:.0f}s (peak {peak:.4f}) - "
                                     f"check your {source_label} is unmuted/active before you start talking.")
        return False, c.success(f"  Audio device OK ({source_label}, peak level {peak:.3f})")
    except Exception as e:
        return True, c.error(f"  Audio device FAILED to open: {e}")


def _check_disk_space() -> str | None:
    try:
        usage = shutil.disk_usage(STATE_DIR)
        free_gb = usage.free / 1e9
        if free_gb < MIN_FREE_DISK_GB:
            return c.warning(f"  Low disk space: {free_gb:.1f}GB free - audio/notes backups need room.")
        return c.success(f"  Disk space OK ({free_gb:.1f}GB free)")
    except Exception:
        return None  # non-critical, skip silently if the check itself fails


def _check_whisper_model() -> tuple[bool, str]:
    try:
        transcribe.get_model()
        return False, c.success(f"  Whisper model OK ({transcribe.device_info()})")
    except Exception as e:
        return True, c.error(f"  Whisper model FAILED to load: {e}")


class PreflightRunner:
    """Same sanity checks as before (audio device, disk space, Whisper model load, and -
    "local" mode only - the local GPU formatter's warm-up), but started as early as
    possible instead of all at once right before recording. The interactive prompts
    between "record a lecture" and actually starting (which class, which audio source,
    which formatting mode) involve real wall-clock time spent waiting on the user to
    type an answer - dead time that was previously wasted, since none of these checks
    used to start until every prompt had already been answered.

    __init__ kicks off the two checks that don't depend on any user choice (disk space,
    and Whisper model load - the slowest of the four, especially cold) immediately.
    start_audio_check()/start_gpu_warmup() kick off the other two as soon as their own
    answer (source / formatting_mode) is known, rather than waiting for every remaining
    prompt too. finish() is the only thing that prints anything - it starts any check
    that didn't get an early opportunity (so this class is still correct and safe to use
    even if a caller skips the early start_*() calls entirely), then waits for and prints
    every result together in the same fixed order as before, so the visible checks-then-
    result output is unchanged; only the wall-clock time to get there shrinks."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._disk_future = self._executor.submit(_check_disk_space)
        self._whisper_future = self._executor.submit(_check_whisper_model)
        self._audio_future = None
        self._gpu_future = None

    def start_audio_check(self, source: str):
        if self._audio_future is not None:
            return
        if source == "mic":
            # Synchronous, and must happen before the mic gets opened by the audio-device
            # check below - if muted at the OS level, this clears it first so that check
            # actually picks up real audio instead of reporting silence. Only covers
            # OS-level mute (see mic_mute.py docstring for what it can't see).
            if mic_mute.check_and_unmute() == "unmuted":
                print(c.warning("  Microphone was muted at the system level - unmuted it automatically."))
        self._audio_future = self._executor.submit(_check_audio_device, source)

    def start_gpu_warmup(self, formatting_mode: str):
        if self._gpu_future is not None:
            return
        # Only in "local" mode is this tier guaranteed to be needed - in "auto" mode
        # it's a fallback that's usually never touched (Claude CLI/API handle
        # everything), so warming it up there would just burn VRAM/startup time for
        # nothing in the common case. See gpu_formatter.warm_up() for the full reasoning.
        if formatting_mode == "local" and gpu_formatter.available():
            self._gpu_future = self._executor.submit(gpu_formatter.warm_up)

    def finish(self, source: str, formatting_mode: str) -> bool:
        self.start_audio_check(source)
        self.start_gpu_warmup(formatting_mode)

        print(c.heading("Running preflight checks..."))
        ok = True

        blocks, msg = self._audio_future.result()
        print(msg)
        ok = ok and not blocks

        disk_msg = self._disk_future.result()
        if disk_msg is not None:
            print(disk_msg)

        blocks, msg = self._whisper_future.result()
        print(msg)
        ok = ok and not blocks

        if self._gpu_future is not None:
            print(c.success("  Local GPU formatter warmed up and ready.")
                  if self._gpu_future.result() else
                  c.warning("  Local GPU formatter failed to warm up - will retry lazily when actually needed."))

        if diarize.available():
            print(c.info("  Speaker diarization: enabled (HUGGINGFACE_TOKEN set)"))
        else:
            print(c.dim("  Speaker diarization: off (set HUGGINGFACE_TOKEN to enable Q&A speaker labeling)"))

        print()
        self._executor.shutdown(wait=False)
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


def _relative_time(dt: datetime) -> str:
    seconds = (datetime.now() - dt).total_seconds()
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hr ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def list_session_backups():
    backups = sorted(STATE_DIR.glob("*_raw.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        print("No session backups found in state/.")
        return
    print(c.heading("Session backups available to resume (most recent first):\n"))
    for raw_path in backups:
        wav_path = raw_path.parent / raw_path.name.replace("_raw.txt", ".wav")
        try:
            _, _, class_code, _ = _resume_backup_paths(raw_path)
        except ValueError:
            class_code = raw_path.stem

        m = re.search(r"(\d{8}_\d{6})", raw_path.name)
        when = _relative_time(datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")) if m else "?"

        if wav_path.exists():
            size_mb = wav_path.stat().st_size / 1e6
            duration_min = wav_path.stat().st_size / 4 / audio.SAMPLE_RATE / 60  # mono float32
            audio_note = c.success(f"~{duration_min:.0f} min audio, {size_mb:.0f}MB")
        else:
            audio_note = c.warning("no audio backup (raw transcript only)")

        print(f"  {class_code:<16} {when:<12} {audio_note}")
        print(c.dim(f"    --resume \"{wav_path if wav_path.exists() else raw_path}\""))


def prune_backups(older_than_days: int, confirm: bool):
    """Deletes state/ backups (.wav and _raw.txt) older than `older_than_days`. Dry-run
    by default - only actually deletes when `confirm` is True, since these are the only
    recovery path (--resume) for a crashed session, so deleting them isn't reversible."""
    cutoff = time.time() - older_than_days * 86400
    candidates = sorted(
        p for pattern in ("*.wav", "*_raw.txt") for p in STATE_DIR.glob(pattern)
        if p.stat().st_mtime < cutoff
    )
    if not candidates:
        print(c.info(f"No backups older than {older_than_days} day(s) found - nothing to prune."))
        return

    total_bytes = sum(p.stat().st_size for p in candidates)
    verb = "Deleting" if confirm else "Would delete"
    print(c.heading(f"{verb} {len(candidates)} backup file(s) older than {older_than_days} "
                     f"day(s), {total_bytes / 1e9:.2f}GB total:\n"))
    for p in candidates:
        print(f"  {p.name} ({p.stat().st_size / 1e6:.1f}MB)")

    if not confirm:
        print(c.warning(f"\nDry run only - nothing deleted. Re-run with --prune-backups "
                         f"--confirm to actually delete these {len(candidates)} file(s)."))
        return

    deleted = 0
    for p in candidates:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(c.error(f"  Failed to delete {p.name}: {e}"))
    print(c.success(f"\nDeleted {deleted}/{len(candidates)} file(s), freed {total_bytes / 1e9:.2f}GB."))


def _pause_before_exit():
    """Keeps the console window open until a keypress before the process exits. Real
    problem when the app is launched from a shortcut/double-click rather than an
    already-open terminal: the window vanishes the instant the process exits, so a
    result printed right before returning (a quiz score, a study guide path) is never
    actually seen - confirmed live, this is why a flashcard score seemed to "not show"."""
    input("\nPress Enter to exit...")


def _resolve_class_for_llm_feature(code: str, feature: str) -> dict:
    """Shared setup for --study-guide/--flashcards: resolve the class code and confirm
    Claude CLI/API is available (both features need real cross-lecture synthesis - see
    notes.generate_study_guide/generate_flashcards for why the local model isn't used)."""
    cls = sched.get_class_by_code(code)
    if cls is None:
        print(c.error(f"No class with code '{code}' found in schedule.json."))
        print(c.info("Known classes:"))
        for known_code, title in sched.list_all_classes():
            print(f"  {known_code} - {title}")
        _pause_before_exit()
        sys.exit(1)

    if not notes.cli_available() and not os.environ.get("ANTHROPIC_API_KEY"):
        print(c.error(f"{feature} needs the Claude CLI (logged in) or "
                       "ANTHROPIC_API_KEY - neither is available right now."))
        _pause_before_exit()
        sys.exit(1)
    return cls


def run_study_guide(code: str):
    """Generates a consolidated, cross-lecture study guide for one class from all of its
    notes so far. CLI/API only - see notes.generate_study_guide for why the local model
    isn't used here."""
    cls = _resolve_class_for_llm_feature(code, "Study guide generation")

    print(c.heading(f"Building study guide for {cls['code']} - {cls['title']} "
                     f"from all notes so far..."))
    start = time.time()
    ok, reason, path = notes.generate_study_guide(cls["code"], cls["title"])
    if not ok:
        print(c.error(f"Could not generate study guide: {reason}"))
        _pause_before_exit()
        sys.exit(1)

    print(c.success(f"Study guide saved to {path} ({time.time() - start:.1f}s)."))
    print(c.info(f"Study guide (Word): {docx_export.study_guide_docx_path(cls['code'])}"))
    _pause_before_exit()


def run_flashcards(code: str):
    """Generates a multiple-choice quiz from a class's notes so far, then runs it
    interactively in the terminal: one question at a time, immediate right/wrong feedback
    with a short explanation, and a final score. CLI/API only - see
    notes.generate_flashcards for why the local model isn't used here.

    Also exports the same question set as an Anki-importable deck (recall-style, not
    multiple-choice - see notes.export_flashcards_anki for why) - the terminal quiz is a
    one-off test of where you stand right now, the Anki export is for actual ongoing
    spaced-repetition review afterward, which Anki's scheduler already does well."""
    cls = _resolve_class_for_llm_feature(code, "Flashcard generation")

    print(c.heading(f"Building a {notes.FLASHCARD_COUNT}-question quiz for {cls['code']} - "
                     f"{cls['title']} from all notes so far..."))
    start = time.time()
    ok, reason, questions = notes.generate_flashcards(cls["code"], cls["title"])
    if not ok:
        print(c.error(f"Could not generate flashcards: {reason}"))
        _pause_before_exit()
        sys.exit(1)
    print(c.success(f"Quiz ready ({len(questions)} questions, {time.time() - start:.1f}s).\n"))

    anki_path = notes.export_flashcards_anki(cls["code"], cls["title"], questions)
    print(c.info(f"Anki-importable deck saved to {anki_path} (Anki: File > Import).\n"))

    random.shuffle(questions)
    score = 0
    answered = 0
    letters = ["A", "B", "C", "D"]

    for i, q in enumerate(questions, 1):
        # Shuffle each question's option order independently, rather than trusting the LLM
        # to vary the correct answer's position on its own (models tend toward a favorite
        # slot, e.g. always "B") - keeps the quiz from becoming guessable by pattern.
        order = list(range(4))
        random.shuffle(order)
        shuffled_options = [q["options"][j] for j in order]
        correct_letter = letters[order.index(q["correct_index"])]

        print(c.heading(f"Q{i}/{len(questions)}: ") + q["question"])
        for letter, option in zip(letters, shuffled_options):
            print(f"  {letter}. {option}")

        answer = input("Your answer (A/B/C/D, or 'q' to quit): ").strip().upper()
        if answer == "Q":
            break
        while answer not in letters:
            answer = input("Please enter A, B, C, D, or 'q' to quit: ").strip().upper()
            if answer == "Q":
                break
        if answer == "Q":
            break

        answered += 1
        if answer == correct_letter:
            score += 1
            print(c.success(f"Correct! {q['explanation']}\n"))
        else:
            print(c.error(f"Incorrect - the answer was {correct_letter}. {q['explanation']}\n"))

    if answered == 0:
        print(c.dim("No questions answered - quiz ended."))
        _pause_before_exit()
        return

    pct = 100 * score / answered
    print(c.heading("=" * 40))
    print(c.heading(f"Quiz complete! Score: {score}/{answered} ({pct:.0f}%)"))
    print(c.heading("=" * 40))
    _pause_before_exit()


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


def try_full_session_diarization(cls, session: Session, session_date, session_start_offset, formatting_mode) -> bool:
    """At session end, if diarization is available, re-diarizes the WHOLE session's
    audio (not just whatever's pending since the last autosave) using session.all_segments
    - which, unlike pending_segments, holds the complete session transcript - so Q&A
    sections come out across the entire lecture instead of just the last few minutes.
    Replaces this session's whole contribution to the notes file with one clean,
    fully speaker-labeled section (discarding the plain-text autosave sections that
    already exist for it - there's nothing to condense-merge anymore since this writes
    the complete, correct version in one shot).

    Returns True if this path was used (caller should skip the normal tail-save +
    condense flow entirely); False if diarization isn't available/failed, in which
    case the caller should fall back to the standard flow."""
    if not diarize.available() or not session.all_segments:
        return False

    print(c.info("Running speaker diarization on the FULL session audio (not just the "
                  "final segment) so Q&A sections cover the whole lecture - this can "
                  "take a while on a long recording..."))
    diar_start = time.time()
    session.flush()
    diar_segments = diarize.diarize(session.wav_path)
    if not diar_segments:
        print(c.warning("  Full-session diarization returned nothing - falling back to a normal save."))
        return False

    labeled = diarize.assign_speakers(session.all_segments, diar_segments)
    labeled_text = diarize.to_labeled_transcript(labeled)
    if not labeled_text.strip():
        return False
    print(c.dim(f"  Diarization complete ({time.time() - diar_start:.1f}s)."))

    notes_path = notes.notes_path(cls["code"])
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    # Discard everything already written for TODAY's date - not just session_start_offset
    # (this session's own start). If the app was run more than once today for this class
    # (recording paused and resumed in a fresh process), session_start_offset from a later
    # run wouldn't know about an earlier run's "## <date>" section for the same day,
    # leaving two separate headers side by side with nothing to merge them (this path
    # skips the condense pass entirely, so that duplication would never get cleaned up).
    # Finding the date heading fresh sweeps in everything for today regardless of how
    # many separate runs wrote it, and safely falls back to session_start_offset's
    # equivalent (append after everything) if today's date hasn't appeared yet.
    truncate_at = _find_date_section_boundary(existing, session_date)
    notes_path.write_text(existing[:truncate_at], encoding="utf-8")

    save_start = time.time()
    path, method = notes.format_and_save(cls["code"], cls["title"], labeled_text, session_date,
                                          mode=formatting_mode, combine_proofread=True)
    print(c.success(f"  Full session formatted and saved as one clean section "
                     f"(method: {method}, {time.time() - save_start:.1f}s)."))
    return True


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
    parser.add_argument("--prune-backups", action="store_true",
                         help="Delete old state/ backups (dry-run unless --confirm is also given)")
    parser.add_argument("--older-than", type=int, default=30, metavar="DAYS",
                         help="Age threshold in days for --prune-backups (default 30)")
    parser.add_argument("--confirm", action="store_true",
                         help="Actually perform the deletion for --prune-backups (otherwise dry-run only)")
    parser.add_argument("--study-guide", metavar="CODE",
                         help="Generate a consolidated, topic-organized study guide from all of a "
                              "class's notes so far (e.g. 'PSYC 1300') and exit. Requires Claude "
                              "CLI or API - needs real synthesis across lectures, the local model "
                              "isn't reliable enough for this.")
    parser.add_argument("--flashcards", metavar="CODE",
                         help="Quiz yourself with an interactive multiple-choice flashcard session "
                              "generated from all of a class's notes so far, and exit. Same CLI/API "
                              "requirement as --study-guide.")
    args = parser.parse_args()

    if args.list:
        for code, title in sched.list_all_classes():
            print(f"{code} - {title}")
        return

    if args.list_sessions:
        list_session_backups()
        return

    if args.prune_backups:
        prune_backups(args.older_than, args.confirm)
        return

    if args.resume:
        run_resume(args.resume, args)
        return

    if args.study_guide:
        run_study_guide(args.study_guide)
        return

    if args.flashcards:
        run_flashcards(args.flashcards)
        return

    action = choose_action(args)
    if action == "study_guide":
        run_study_guide(choose_notes_class(args))
        return
    if action == "flashcards":
        run_flashcards(choose_notes_class(args))
        return

    # Starts the checks that don't depend on any answer below (disk space, Whisper model
    # load - the slowest of the four) right now, so they run during the wall-clock time
    # spent waiting on the user to answer the prompts below instead of only starting
    # once every prompt is already answered.
    preflight = PreflightRunner()

    cls = choose_class(args)
    source = choose_source(args)
    preflight.start_audio_check(source)  # don't wait for the formatting-mode prompt too
    formatting_mode = choose_formatting_mode(args)
    preflight.start_gpu_warmup(formatting_mode)
    initial_prompt = vocab_module.initial_prompt_for_class(cls["code"])

    print()
    if not preflight.finish(source, formatting_mode):
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
                     f"{cls['code']} - {cls['title']}."))
    save_now_hint = (f" Press '{SAVE_NOW_KEY.decode()}' anytime to save immediately "
                      f"(also resets the autosave timer)." if msvcrt is not None else "")
    print(c.heading(f"Press Ctrl+C to stop and save notes.{save_now_hint}\n"))

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

    save_listener = SaveNowListener()
    save_listener.start()
    save_worker = SaveWorker()

    def _handle_transcription_result(chunk_audio, whisper_segments):
        nonlocal silence_warned
        session.write_chunk(chunk_audio, whisper_segments)
        for seg in whisper_segments:
            stamp = time.strftime("%H:%M:%S")
            print(f"{c.timestamp('[' + stamp + ']')} {c.transcript(seg['text'])}")
            with open(raw_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {seg['text']}\n")

        silent_seconds = transcriber.consecutive_silent_chunks * args.chunk
        if silent_seconds >= SILENCE_WARNING_SECONDS and not silence_warned:
            print(c.warning(f"No audio detected for ~{silent_seconds/60:.1f} min - "
                             f"check your {'microphone' if source == 'mic' else 'system audio'} "
                             f"source (muted? disconnected?)."))
            silence_warned = True
        elif transcriber.consecutive_silent_chunks == 0:
            silence_warned = False

    transcription_worker = TranscriptionWorker(transcriber, _handle_transcription_result)

    try:
        while True:
            chunk_audio = capture.collect_chunk(args.chunk, interrupt_event=save_listener.requested)
            for err in capture.poll_errors():
                print(c.error(f"Audio device error (auto-recovering, capture continues): {err}"))

            if chunk_audio is not None:
                transcription_worker.submit(chunk_audio)

            if save_listener.requested.is_set():
                save_listener.requested.clear()
                if session.has_pending():
                    print(c.autosave(f"\n'{SAVE_NOW_KEY.decode()}' pressed - saving now..."))
                    _save(cls, session, session_date, formatting_mode, note="(manual save)",
                          save_worker=save_worker)
                else:
                    print(c.dim(f"\n'{SAVE_NOW_KEY.decode()}' pressed, but nothing new to save yet."))
                last_autosave = time.time()  # reset either way - that's what was asked for
                continue

            if chunk_audio is None:
                continue  # nothing captured yet (e.g. very start) - keep waiting

            if time.time() - last_autosave > AUTOSAVE_EVERY_SECONDS and session.has_pending():
                # Printed the instant this kicks off, not just when it finishes - the
                # save itself runs on save_worker's background thread now, so without
                # this it could look like nothing was happening for however long that
                # takes, then a "Saved..." line would just appear out of nowhere.
                print(c.autosave("\n(autosaving in background...)"))
                _save(cls, session, session_date, formatting_mode, note="(autosave)",
                      save_worker=save_worker)
                last_autosave = time.time()
    except KeyboardInterrupt:
        print(c.autosave("\nStopping - wrapping up your notes now, this can take a moment..."))
    finally:
        shutdown_start = time.time()

        def _step(msg):
            print(c.heading(f"[{time.time() - shutdown_start:5.1f}s] ") + c.info(msg))

        keep_awake.allow_sleep()
        save_listener.stop()

        _step("Stopping audio capture...")
        capture.stop()
        print(c.dim("  capture thread stopped."))

        # Drain any chunks still queued for background transcription before touching
        # `transcriber` directly below (tail-audio processing) or reading
        # session.all_segments further down - RollingTranscriber's rolling state
        # (overlap audio, recent text) only makes sense processed in capture order, and
        # a still-in-flight chunk's segments wouldn't be in the session yet otherwise.
        if transcription_worker.has_pending():
            _step("Waiting for background transcription to catch up...")
        transcription_worker.wait_idle()

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

        if save_worker.has_pending():
            _step("Waiting for a still-in-flight background autosave to finish...")
        save_worker.wait_idle()  # nothing below this point may run against the notes file
                                  # until every prior background autosave has actually written it

        used_full_diarization = False
        if diarize.available() and session.all_segments:
            used_full_diarization = try_full_session_diarization(
                cls, session, session_date, session_start_offset, formatting_mode)

        if not used_full_diarization:
            if session.pending_segments:
                n_pending = len(session.pending_segments)
                _step(f"Formatting and saving the final segment ({n_pending} transcribed chunk(s))...")
                save_start = time.time()
                _save(cls, session, session_date, formatting_mode, is_final=False)
                print(c.dim(f"  final save completed in {time.time() - save_start:.1f}s."))
            else:
                _step("Nothing pending to save (already flushed by the last autosave).")
        session.close()

        elapsed_min = session.elapsed() / 60
        print(c.success(f"\nRecording done - captured ~{elapsed_min:.1f} minutes of lecture."))

        if used_full_diarization:
            print(c.dim("Skipped condense pass: full-session diarization already wrote one clean "
                         "section covering the whole lecture - nothing left to merge."))
        elif formatting_mode != "auto":
            print(c.dim(f"Skipped condense pass: formatting mode is '{formatting_mode}' (no CLI/API calls)."))
        elif notes.cli_available():
            _step("Reviewing this session's notes with Claude CLI (condensing duplicate autosave "
                  "sections, cleaning up formatting)...")
            condense_start = time.time()
            # Use the boundary of TODAY's date heading, not just this session's own
            # start - if the app was run more than once today for this class (e.g.
            # recording paused and resumed in a fresh process), an earlier run's
            # session_start_offset wouldn't know about a still-earlier run's section
            # for the same date, leaving duplicate "## <date>" headers that never get
            # merged. Finding the date heading fresh at condense time sweeps in
            # everything for today regardless of how many separate runs wrote it.
            current_notes_content = notes.notes_path(cls["code"]).read_text(encoding="utf-8")
            condense_boundary = _find_date_section_boundary(current_notes_content, session_date)
            ok, reason = notes.condense_session(cls["code"], cls["title"], condense_boundary, mode=formatting_mode)
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


def _save(cls, session: Session, session_date, mode, note="", is_final=False,
          save_worker: "SaveWorker | None" = None):
    """pop_pending_text() always runs synchronously on the calling thread right now, so
    it correctly snapshots whatever's pending at this instant - the actual slow work
    (format_and_save's CLI/API/file-write call) is handed to `save_worker` when given,
    so the main loop can immediately go back to transcribing instead of blocking on it.
    Pass save_worker=None (the final Ctrl+C save does this) to run it synchronously
    instead - at that point there's no more live output to keep flowing anyway, and the
    caller needs it to have actually finished before moving on to condensing."""
    full_text = session.pop_pending_text(run_diarization=is_final)

    def do_save():
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

    if save_worker is not None:
        save_worker.submit(do_save)
    else:
        do_save()


if __name__ == "__main__":
    run()

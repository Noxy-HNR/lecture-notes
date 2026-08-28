"""
Lecture Notes App
==================
Figures out what class you're in based on your schedule, records + transcribes
the lecture locally (Whisper), and turns it into clean notes appended to that
class's ongoing notes file (Claude Code CLI / API when available, local
fallback otherwise). Optionally detects different speakers (Q&A) via
pyannote.audio diarization if it's installed and configured.

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
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import schedule as sched
import audio
import transcribe
import notes
import docx_export
import diarize
import vocab as vocab_module
import soundfile as sf

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
        print(f"Detected class from your schedule: {detected['code']} - {detected['title']} "
              f"({detected['location']}, {status_msg})")
        return detected

    print("No class found in your schedule for right now.")
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
        print("Invalid choice.")
        sys.exit(1)


def choose_source(args):
    if args.source in ("mic", "system"):
        return args.source
    print("Audio source:")
    print("  1. Microphone (in-person lecture)")
    print("  2. System audio / loopback (online lecture, e.g. Zoom)")
    choice = input("Choose [1]: ").strip() or "1"
    return "mic" if choice == "1" else "system"


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

    def process(self, new_chunk: np.ndarray) -> list[dict]:
        """Returns segments (start/end/text) with timestamps relative to `new_chunk`,
        ready to pass to Session.write_chunk alongside it."""
        overlap_duration = len(self.overlap_audio) / audio.SAMPLE_RATE
        combined = np.concatenate([self.overlap_audio, new_chunk])

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
    """Tracks the running WAV recording + transcript segments for one lecture, so
    saves (autosave or final) can hand off just the audio/text since the last save
    to diarization, keeping speaker labels in sync with what's actually being saved."""

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

    def pop_pending_text(self) -> str:
        """Returns speaker-labeled (if diarization succeeds) or plain transcript text
        for everything recorded since the last save, and resets the save window."""
        if not self.pending_segments:
            return ""

        window_start_sec = self.last_save_elapsed
        window_start_frame = self.last_save_frame
        window_end_frame = self.total_frames

        text = " ".join(seg["text"] for seg in self.pending_segments)

        if diarize.available() and window_end_frame > window_start_frame:
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


def run():
    parser = argparse.ArgumentParser(description="Auto-transcribe your current lecture into notes.")
    parser.add_argument("--class", dest="klass", help="Override class code, e.g. 'BIOL 1440'")
    parser.add_argument("--source", choices=["mic", "system"], help="Audio source, skips the prompt")
    parser.add_argument("--chunk", type=float, default=15.0, help="Seconds per transcription chunk")
    parser.add_argument("--list", action="store_true", help="List classes from schedule.json and exit")
    args = parser.parse_args()

    if args.list:
        for code, title in sched.list_all_classes():
            print(f"{code} - {title}")
        return

    cls = choose_class(args)
    source = choose_source(args)
    initial_prompt = vocab_module.initial_prompt_for_class(cls["code"])

    print(f"\nLoading local Whisper model (first run downloads it, may take a minute)...")
    transcribe.get_model()  # warm up / trigger download before recording starts
    print(f"Whisper running on: {transcribe.device_info()}")

    if diarize.available():
        print("Speaker diarization: enabled (HUGGINGFACE_TOKEN set)")
    else:
        print("Speaker diarization: off (set HUGGINGFACE_TOKEN to enable Q&A speaker labeling)")

    print(f"Recording from {'microphone' if source == 'mic' else 'system audio'} for "
          f"{cls['code']} - {cls['title']}. Press Ctrl+C to stop and save notes.\n")

    recorder = audio.get_recorder(source)
    now = datetime.now()
    session_date = f"{now.strftime('%A, %B')} {now.day}, {now.strftime('%Y')}"
    last_autosave = time.time()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_path = STATE_DIR / f"{cls['code'].replace(' ', '_')}_{ts}_raw.txt"
    wav_path = STATE_DIR / f"{cls['code'].replace(' ', '_')}_{ts}.wav"
    session = Session(wav_path)
    transcriber = RollingTranscriber(initial_prompt)

    try:
        with recorder:
            while True:
                chunk_audio = audio.record_chunk(recorder, args.chunk)
                whisper_segments = transcriber.process(chunk_audio)
                session.write_chunk(chunk_audio, whisper_segments)

                for seg in whisper_segments:
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] {seg['text']}")
                    with open(raw_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{stamp}] {seg['text']}\n")

                if time.time() - last_autosave > AUTOSAVE_EVERY_SECONDS and session.pending_segments:
                    _save(cls, session, session_date, note="(autosave)")
                    last_autosave = time.time()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if session.pending_segments:
            _save(cls, session, session_date)
        session.close()
        elapsed_min = session.elapsed() / 60
        print(f"\nDone. Recorded ~{elapsed_min:.1f} minutes.")
        print(f"Notes (markdown): {notes.notes_path(cls['code'])}")
        print(f"Notes (Word): {docx_export.docx_path(cls['code'])}")
        print(f"Raw transcript backup: {raw_log_path}")
        print(f"Audio backup: {wav_path}")


def _save(cls, session: Session, session_date, note=""):
    full_text = session.pop_pending_text()
    path, method = notes.format_and_save(cls["code"], cls["title"], full_text, session_date)
    tag = {
        "cli": " [formatted with Claude Code CLI]",
        "api": " [formatted with Claude API]",
        "local": " [local formatting - Claude CLI/API unavailable]",
        "none": "",
    }[method]
    print(f"Saved notes to {path}{tag} {note}")


if __name__ == "__main__":
    run()

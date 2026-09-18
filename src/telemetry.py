"""Live session telemetry, written to state/status.json for the diagnostics dashboard.

The recorder and the dashboard are separate processes (deliberately - the dashboard must
never be able to slow down or crash a recording in progress), so they talk through two
small files instead of a socket:

  state/status.json   recorder -> dashboard   what's happening right now
  state/command.json  dashboard -> recorder   "save now" / "stop", mirroring the
                                              terminal's 's' key and Ctrl+C

Writes are atomic (temp file + os.replace) because the dashboard polls this file on its
own schedule and would otherwise occasionally read a half-written JSON document. Every
write is also best-effort: telemetry must never be able to take down a live recording,
so all failures here are swallowed rather than raised.
"""
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATUS_PATH = STATE_DIR / "status.json"
COMMAND_PATH = STATE_DIR / "command.json"

MAX_TRANSCRIPT_LINES = 300   # what the dashboard mirrors from the terminal's live view
MAX_EVENT_LINES = 120        # saves, warnings, device errors


class Telemetry:
    """Accumulates session state and flushes it to status.json. Thread-safe: the
    recorder touches this from the main loop, the transcription worker, and the save
    worker, all concurrently."""

    def __init__(self):
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        self._transcript = deque(maxlen=MAX_TRANSCRIPT_LINES)
        self._events = deque(maxlen=MAX_EVENT_LINES)
        self._state = {
            "active": False,
            "class_code": "",
            "class_title": "",
            "source": "",
            "formatting_mode": "",
            "started_at": 0.0,
            "chunks_transcribed": 0,
            "segments_transcribed": 0,
            "audio_seconds_transcribed": 0.0,
            "transcribe_seconds_spent": 0.0,
            "realtime_factor": 0.0,
            "last_chunk_seconds": 0.0,
            "last_chunk_duration": 0.0,
            "queue_capture_blocks": 0,
            "queue_transcription": 0,
            "queue_saves": 0,
            "last_save_at": 0.0,
            "last_save_method": "",
            "last_save_seconds": 0.0,
            "saves_completed": 0,
            "consecutive_silent_chunks": 0,
            "claude_cli": "unknown",
            "claude_api": "unknown",
            "gpu_model": "unknown",
            "diarization": "unknown",
            "model_device": "",
            "notes_path": "",
            "wav_path": "",
        }

    # -- writes from the recorder ------------------------------------------------

    def start_session(self, **fields):
        with self._lock:
            self._state.update(fields)
            self._state["active"] = True
            self._state["started_at"] = time.time()
        self.flush()

        self._heartbeat_stop.clear()
        def heartbeat():
            while not self._heartbeat_stop.wait(5):
                self.flush()
        self._heartbeat_thread = threading.Thread(target=heartbeat, daemon=True, name="notes-heartbeat")
        self._heartbeat_thread.start()

    def end_session(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        with self._lock:
            self._state["active"] = False
        self.flush()

    def update(self, **fields):
        with self._lock:
            self._state.update(fields)

    def record_chunk(self, audio_seconds: float, transcribe_seconds: float, segments: int):
        """Feeds the realtime-factor metric: how many seconds of GPU time each second of
        audio costs. Below 1.0 means transcription outruns the lecture (healthy);
        sustained above 1.0 means it's falling behind and the live view will lag."""
        with self._lock:
            s = self._state
            s["chunks_transcribed"] += 1
            s["segments_transcribed"] += segments
            s["audio_seconds_transcribed"] += audio_seconds
            s["transcribe_seconds_spent"] += transcribe_seconds
            s["last_chunk_seconds"] = round(audio_seconds, 2)
            s["last_chunk_duration"] = round(transcribe_seconds, 2)
            if s["audio_seconds_transcribed"] > 0:
                s["realtime_factor"] = round(
                    s["transcribe_seconds_spent"] / s["audio_seconds_transcribed"], 3)

    def record_save(self, method: str, seconds: float):
        with self._lock:
            self._state["saves_completed"] += 1
            self._state["last_save_at"] = time.time()
            self._state["last_save_method"] = method
            self._state["last_save_seconds"] = round(seconds, 2)

    def add_transcript(self, stamp: str, text: str):
        with self._lock:
            self._transcript.append({"time": stamp, "text": text})

    def add_event(self, level: str, message: str):
        """level: info | success | warning | error - drives colour in the dashboard,
        same information the terminal conveys with its colour scheme."""
        with self._lock:
            self._events.append({"time": time.strftime("%H:%M:%S"),
                                  "level": level, "message": message})

    def flush(self):
        # Serialize snapshot and replacement: heartbeat and result callbacks share a file.
        with self._write_lock:
            self._flush_locked()

    def _flush_locked(self):
        with self._lock:
            payload = dict(self._state)
            payload["updated_at"] = time.time()
            payload["elapsed"] = (time.time() - payload["started_at"]) if payload["active"] else 0.0
            payload["transcript"] = list(self._transcript)
            payload["events"] = list(self._events)
        try:
            STATE_DIR.mkdir(exist_ok=True)
            tmp = STATUS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, STATUS_PATH)  # atomic - readers never see a partial file
        except Exception:
            pass  # telemetry must never break a recording


# -- command channel (dashboard -> recorder) -------------------------------------

def send_command(action: str) -> bool:
    """Called by the dashboard. Returns whether it was written."""
    try:
        STATE_DIR.mkdir(exist_ok=True)
        tmp = COMMAND_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"action": action, "at": time.time()}), encoding="utf-8")
        os.replace(tmp, COMMAND_PATH)
        return True
    except Exception:
        return False


def take_command() -> str | None:
    """Called by the recorder each loop. Consumes and returns any pending action, so a
    single dashboard click can't be executed twice."""
    try:
        if not COMMAND_PATH.exists():
            return None
        action = json.loads(COMMAND_PATH.read_text(encoding="utf-8")).get("action")
        COMMAND_PATH.unlink(missing_ok=True)
        return action
    except Exception:
        return None


def read_status() -> dict:
    """Called by the dashboard. Returns {} when no session has ever run."""
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

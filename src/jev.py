"""TypeSafe's Jev: quick, typed judgments over lecture text.

Used for three things, all optional and all run from the dashboard process, never the recorder:
  - highlights.py   exam hints, "not on the exam" remarks and announcements in each transcript
                    segment, plus segments that probably contain a mis-heard word
  - search.py       re-ranking meaning-search results by whether they answer the question

Jev answers questions about a small JSON "state" with probabilities rather than generated
text, so the code here decides what to do with them. This module is the only place that talks
to TypeSafe. What it sends is exactly the state its callers build (a transcript segment with a
little surrounding text, a notes line, a search query) after `redact()`; nothing else from the
machine goes with it, and neither requests nor responses are logged or stored as text.

The API key is the one saved in the C:\\AI dashboard's Settings tab (Switchboard's .env), so
there is one place to paste it; TYPESAFE_API_KEY in the environment takes precedence. Jev can
be switched off from the Diagnostics page (state/jev_settings.json).
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"
SETTINGS_PATH = STATE_DIR / "jev_settings.json"
SWITCHBOARD_ENV = PROJECT_ROOT.parents[1] / "Switchboard" / ".env"

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
# Only input tokens are billed. Same rate Switchboard uses (config/switchboard.toml).
USD_PER_INPUT_MTOK = 0.042
TIMEOUT_SECONDS = 20
RETRIES = 2
WORKERS = 8


class JevError(RuntimeError):
    """A failed call. `kind` is one of: off, unconfigured, auth, rate, overloaded, invalid,
    network, server, bad_response. Messages carry status codes and exception names only -
    never a response body, which can echo the request back."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Key and settings
# ---------------------------------------------------------------------------

_KEY_LINE = re.compile(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*?)\s*$")


def api_key() -> tuple[str | None, str]:
    """(key, where it came from). The key itself is never logged or returned by any API."""
    value = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if value:
        return value, "environment"
    try:
        text = SWITCHBOARD_ENV.read_text(encoding="utf-8-sig")
    except OSError:
        return None, "missing"
    for line in text.splitlines():
        match = _KEY_LINE.match(line)
        if match:
            value = match.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if value:
                return value, "Switchboard settings"
    return None, "missing"


def settings() -> dict:
    try:
        saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {"enabled": saved.get("enabled", True) is not False}


def save_settings(enabled: bool) -> dict:
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"enabled": enabled}) + "\n", encoding="utf-8")
    os.replace(tmp, SETTINGS_PATH)
    return status()


def status() -> dict:
    key, source = api_key()
    enabled = settings()["enabled"]
    return {"enabled": enabled, "key": source, "ready": enabled and key is not None}


def label() -> str:
    """One line for the Diagnostics service chips."""
    s = status()
    if not s["enabled"]:
        return "turned off"
    if s["key"] == "missing":
        return "no key (add it in the C:\\AI dashboard's Settings tab)"
    return f"enabled, key from {s['key']}"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Lectures are mostly subject matter, but instructors read out email addresses and phone
# numbers, and a student ID can end up in a transcript. The lookbehinds anchor each pattern
# at the start of its run, so matching stays linear in the text length.
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\d(])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone number]", text)
    return _LONG_NUMBER.sub("[number]", text)


def clip(text: str, limit: int, keep: str = "start") -> str:
    """Cuts at a word boundary. keep="end" keeps the last `limit` characters instead."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    if keep == "end":
        cut = text[-limit:]
        return cut.split(" ", 1)[-1] if " " in cut else cut
    cut = text[:limit]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

_HTTP_KINDS = {401: "auth", 403: "auth", 422: "invalid", 429: "rate", 529: "overloaded"}


def ask(state, questions: dict, key: str | None = None, timeout: float = TIMEOUT_SECONDS) -> dict:
    """One request: every question is answered about the same state, in parallel on
    TypeSafe's side. Returns {"answers", "model", "input_tokens"}. Retries rate limits,
    overload and server errors with backoff; raises JevError otherwise."""
    if key is None:
        if not settings()["enabled"]:
            raise JevError("off", "Jev is turned off on the Diagnostics page")
        key = api_key()[0]
    if not key:
        raise JevError("unconfigured", "No TypeSafe key: add it in the C:\\AI dashboard's Settings tab")
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode("utf-8")
    for attempt in range(RETRIES + 1):
        request = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": "lecture-notes"})
        wait = 0.6 * 2 ** attempt
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            kind = _HTTP_KINDS.get(error.code, "server" if error.code >= 500 else "invalid")
            retry_after = error.headers.get("Retry-After") if error.headers else None
            error.close()
            if kind not in ("rate", "overloaded", "server") or attempt == RETRIES:
                raise JevError(kind, f"TypeSafe answered HTTP {error.code}") from None
            if retry_after and retry_after.strip().isdigit():
                wait = min(10.0, float(retry_after))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            if attempt == RETRIES:
                kind = "bad_response" if isinstance(error, ValueError) else "network"
                raise JevError(kind, f"TypeSafe request failed: {type(error).__name__}") from None
        time.sleep(wait)
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict) or any(not isinstance(answers.get(q), dict) for q in questions):
        raise JevError("bad_response", "TypeSafe's answer is missing a question")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tokens = usage.get("input_tokens")
    return {"answers": answers, "model": str(data.get("model") or MODEL),
            "input_tokens": tokens if isinstance(tokens, int) and tokens >= 0 else 0}


def noul(answers: dict, name: str) -> float:
    value = (answers.get(name) or {}).get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise JevError("bad_response", f"TypeSafe returned no probability for {name}")
    return float(value)


class Usage:
    """Calls and billed tokens for one job, for the dashboard's cost line."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self._lock = threading.Lock()

    def add(self, input_tokens: int):
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "est_cost_usd": round(self.input_tokens * USD_PER_INPUT_MTOK / 1_000_000, 6)}


# Errors that will fail every remaining call the same way: stop instead of repeating them.
FATAL_KINDS = ("off", "unconfigured", "auth", "invalid")


def ask_many(jobs: list, questions: dict, usage: Usage | None = None,
             workers: int = WORKERS, should_stop=None) -> list:
    """Asks the same questions about many states at once. Returns one entry per state: the
    answers dict, or the JevError it failed with. A fatal error (bad key, Jev switched off)
    makes the rest fail fast with the same error instead of each trying on its own."""
    if not settings()["enabled"]:
        return [JevError("off", "Jev is turned off on the Diagnostics page")] * len(jobs)
    key = api_key()[0]
    fatal: list[JevError] = []

    def one(state):
        if fatal:
            return fatal[0]
        if should_stop and should_stop():
            return JevError("off", "Stopped")
        try:
            result = ask(state, questions, key=key or "")
        except JevError as error:
            if error.kind in FATAL_KINDS:
                fatal.append(error)
            return error
        if usage is not None:
            usage.add(result["input_tokens"])
        return result["answers"]

    if not key:
        return [JevError("unconfigured", "No TypeSafe key: add it in the C:\\AI dashboard's Settings tab")] * len(jobs)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1)),
                            thread_name_prefix="jev") as pool:
        return list(pool.map(one, jobs))

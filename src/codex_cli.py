"""One prompt through the Codex CLI, on the user's ChatGPT/Codex subscription. Used by the lesson
builder when a GPT model is chosen on the Lessons page (never for note saves).

Each call is `codex exec` in a fresh, empty temporary folder with a read-only sandbox, no saved
session (--ephemeral) and without the user's config.toml (--ignore-user-config: their MCP
servers, plugins and defaults don't apply; sign-in still comes from the normal Codex login).
--strict-config makes Codex refuse a setting it doesn't recognise instead of ignoring it.

Settings, as checked against codex-cli 0.167.0's model catalog (`codex debug models`):
  model_reasoning_effort  low / medium / high (all four models also offer xhigh; not offered here)
  service_tier            "fast" = the catalog's Fast tier ("priority": 2x speed and more usage on
                          Sol and Astra, 1.5x on Terra and Luna). Omitted otherwise.
  web_search              "live" for the video-clip search, "disabled" for everything else
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MODELS = {
    "gpt-6.2-sol": "GPT-6.2 Sol",
    "gpt-6.2-astra": "GPT-6.2 Astra",
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-6-luna": "GPT-6 Luna",
}
EFFORTS = ("low", "medium", "high")

# Codex is an agent; here it is only a writer. The lesson prompts already say exactly what to reply.
_PREAMBLE = ("You are being used as a plain text generator inside another program. Do not run commands, "
             "read or write files, or use any tool{search}. Reply with only the content the request asks for.\n\n")

last_failure = None   # why the last run() returned None, in plain words


def find_cli() -> list[str] | None:
    """The command that starts Codex. npm installs `codex.cmd`, a batch file, which can't be given
    quoted arguments safely, so the JavaScript entry point it wraps is run with node directly."""
    candidates = []
    npm_dir = Path(os.environ.get("APPDATA", "")) / "npm"
    candidates.append(npm_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    found = shutil.which("codex")
    if found:
        candidates.append(Path(found).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    node = shutil.which("node")
    for script in candidates:
        if node and script.is_file():
            return [node, str(script)]
    exe = shutil.which("codex.exe")
    return [exe] if exe else None


def available() -> tuple[bool, str]:
    if find_cli() is None:
        return False, "the Codex CLI isn't installed (npm install -g @openai/codex)"
    return True, ""


def command(work_dir: str, model: str, effort: str, fast: bool, web_search: bool, reply_path: str) -> list[str]:
    cli = find_cli() or ["codex"]
    return [*cli, "exec", "--strict-config", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "-s", "read-only", "-C", work_dir, "-m", model,
            "-c", f'model_reasoning_effort="{effort}"',
            "-c", f'web_search="{"live" if web_search else "disabled"}"',
            *(["-c", 'service_tier="fast"'] if fast else []),
            "-o", reply_path, "-"]


def run(prompt: str, model: str, effort: str = "high", fast: bool = False, timeout: int = 900,
        web_search: bool = False) -> str | None:
    """The model's reply, or None with the reason in `last_failure`."""
    global last_failure
    last_failure = None
    if model not in MODELS:
        last_failure = f"{model!r} isn't one of the Codex models this app offers"
        return None
    if find_cli() is None:
        last_failure = available()[1]
        return None
    effort = effort if effort in EFFORTS else "high"
    label = MODELS[model]
    with tempfile.TemporaryDirectory(prefix="lesson_codex_") as work:
        reply_path = str(Path(work) / "reply.txt")
        text = _PREAMBLE.format(search=" except web search" if web_search else "") + prompt
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command(work, model, effort, fast, web_search, reply_path), cwd=work,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", creationflags=flags)
        try:
            _, err = process.communicate(text, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            last_failure = f"{label} (Codex) did not finish within {timeout // 60} minutes"
            return None
        except Exception as error:
            _kill_tree(process)
            last_failure = f"Codex could not be run ({type(error).__name__}: {error})"
            return None
        reply = Path(reply_path).read_text(encoding="utf-8", errors="replace").strip() \
            if Path(reply_path).is_file() else ""
    if process.returncode != 0:
        detail = " ".join((err or "").strip().splitlines()[-3:])[-300:]
        last_failure = f"{label} (Codex) exited with code {process.returncode}" + (f": {detail}" if detail else "")
        return None
    if not reply:
        last_failure = f"{label} (Codex) returned an empty reply"
        return None
    return reply


def _kill_tree(process):
    """codex.js starts the native codex.exe; stopping only node would leave that running."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=10)
    except Exception:
        pass

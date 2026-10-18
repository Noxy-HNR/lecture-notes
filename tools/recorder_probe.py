"""Shared bounded child runner for offline recorder experiments (Windows).

Children must print PROBE_RESULT followed by JSON. No live recorder files are
written by this runner. The full report goes to a new output path.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from recorder_startup_benchmark import available_ram_gib

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "PROBE_RESULT "


def emit(result):
    print(PREFIX + json.dumps(result), flush=True)


def run_probe(script, arguments, output, *, timeout=180, minimum_ram=3):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Probe timeout must be finite and positive")
    import psutil
    sys.path.insert(0, str(ROOT / "src"))
    import recorder_launcher
    if recorder_launcher.recorder_state()["running"]:
        raise RuntimeError("Close the recorder normally before running an offline probe")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    progress = output.with_suffix(".progress.jsonl")
    if progress.exists():
        raise FileExistsError(f"Progress report already exists: {progress}")
    initial = available_ram_gib()
    if initial < minimum_ram:
        raise RuntimeError(f"Need {minimum_ram:g} GiB available RAM; found {initial:.2f}")
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_stream = progress.open("x", encoding="utf-8")
    try:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(script), *arguments, "--child"],
                                 cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    except BaseException:
        progress_stream.close()
        raise
    stdout, stderr = [], []
    def drain(pipe, lines, progress_file=None):
        if progress_file is None:
            for line in pipe:
                lines.append(line)
            return
        with progress_file as stream:
            for line in pipe:
                lines.append(line)
                if line.startswith(PREFIX):
                    stream.write(line[len(PREFIX):])
                    stream.flush()
    readers = [threading.Thread(target=drain, args=(child.stdout, stdout, progress_stream)),
               threading.Thread(target=drain, args=(child.stderr, stderr))]
    for reader in readers:
        reader.start()
    root = psutil.Process(child.pid)
    start, lowest, peak, stopped = time.monotonic(), initial, 0, None
    next_recorder_check = start + 1
    descendants = []
    try:
        while child.poll() is None:
            try:
                descendants = root.children(recursive=True)
                peak = max(peak, sum(p.memory_info().rss for p in [root, *descendants]))
            except psutil.Error:
                pass
            lowest = min(lowest, available_ram_gib())
            if time.monotonic() >= next_recorder_check:
                if recorder_launcher.recorder_state()["running"]:
                    stopped = "Recorder opened during the probe"
                next_recorder_check = time.monotonic() + 1
            if lowest < 1.2:
                stopped = "Available RAM fell below 1.2 GiB"
            elif time.monotonic() - start > timeout:
                stopped = f"Hard timeout ({timeout:g}s)"
            if stopped:
                break
            time.sleep(.1)
    finally:
        if child.poll() is None:
            try:
                descendants = root.children(recursive=True)
            except psutil.Error:
                pass
            for process in reversed([root, *descendants]):
                try:
                    process.kill()
                except psutil.Error:
                    pass
        child.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)
    measurements = [json.loads(line[len(PREFIX):]) for line in stdout if line.startswith(PREFIX)]
    report = {"returncode": child.returncode, "stopped": stopped,
              "initial_available_ram_gib": initial, "minimum_available_ram_gib": lowest,
              "peak_process_tree_rss_gib": peak / 2**30, "measurements": measurements,
              "stderr_tail": "".join(stderr)[-3500:]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("measurements", "stderr_tail")}))
    print(f"Report: {output.resolve()}")
    if child.returncode or stopped:
        print(report["stderr_tail"])
        raise RuntimeError(stopped or f"Probe exited {child.returncode}")
    return report


def isolate_performance():
    sys.path.insert(0, str(ROOT / "src"))
    import performance
    from contextlib import nullcontext
    performance.stage = lambda *a, **kw: nullcontext()


def prepare_cuda(torch, required_free_gib=6):
    """Reserve headroom before weights load; avoid spilling into host RAM."""
    if not torch.cuda.is_available():
        raise RuntimeError("This decoder probe requires a CUDA GPU")
    torch.cuda.set_per_process_memory_fraction(.8)
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < required_free_gib:
        raise RuntimeError(f"Need {required_free_gib:g} GiB free VRAM; found {free / 2**30:.2f}")

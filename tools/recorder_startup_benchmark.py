"""Bounded startup/inference probes. Never opens a microphone or writes recorder state.

Run with the recorder stopped. Model loading runs in a separate process monitored for
available system RAM and a hard timeout. sherpa packages/models belong under the ignored
state/startup_experiment directory, keeping the recorder's environment unchanged.

    python tools/recorder_startup_benchmark.py --backend hf --menu-seconds 3
    python tools/recorder_startup_benchmark.py --backend sherpa-cpu --manifest clips.json

--baseline-dir can point to a frozen copy of main.py/transcribe.py/docx_export.py.
Results distinguish time to the menu from the wait after choosing Record. Filesystem
caches are not flushed, so repeated probes measure fresh processes with warm file caches.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "state" / "startup_experiment"
MODEL_NAME = "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
RESULT_PREFIX = "STARTUP_RESULT "


def available_ram_gib():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "available_phys", "total_page", "available_page",
             "total_virtual", "available_virtual", "available_extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.available_phys / 2**30


def worker(args):
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    if args.baseline_dir:
        sys.path.insert(0, str(Path(args.baseline_dir).resolve()))
    dll_handles = []
    if args.backend == "hf":
        import main
        import transcribe
        transcribe.COHERE_PATH = ROOT / "state" / "cohere_transcribe"
        import builtins

        def answer(prompt):
            result["menu_shown_seconds"] = time.perf_counter() - started
            time.sleep(args.menu_seconds)
            return "1"

        result = {"backend": "hf", "menu_seconds": args.menu_seconds,
                  "menu_preparation": not args.no_menu_preparation and not args.baseline_dir}
        if hasattr(transcribe, "_import_dependencies"):
            import_dependencies = transcribe._import_dependencies

            def timed_imports():
                beginning = time.perf_counter()
                dependencies = import_dependencies()
                result["dependency_seconds"] = time.perf_counter() - beginning
                return dependencies

            transcribe._import_dependencies = timed_imports
        builtins.input = answer
        prepare = getattr(transcribe, "start_dependency_preparation", None)
        if args.no_menu_preparation and prepare is not None:
            transcribe.start_dependency_preparation = lambda: None
        try:
            main.choose_action(None)
        finally:
            if prepare is not None:
                transcribe.start_dependency_preparation = prepare
        selected = time.perf_counter()
        backend = transcribe.get_model()
        backend._torch.cuda.synchronize()
        result["after_choice_seconds"] = time.perf_counter() - selected
        result["ready_seconds"] = time.perf_counter() - started
        decode = lambda samples, sr: transcribe.transcribe_chunk(samples, sr)
    else:
        provider = "cuda" if args.backend == "sherpa-cuda" else "cpu"
        sys.path.insert(0, str(EXPERIMENT / (provider + "_packages")))
        if provider == "cuda":
            # Resolve CUDA/cuDNN DLLs without importing Torch into the native process.
            for directory in (ROOT / "venv" / "Lib" / "site-packages" / "torch" / "lib",
                              EXPERIMENT / "cuda_packages" / "sherpa_onnx" / "lib"):
                if directory.exists():
                    dll_handles.append(os.add_dll_directory(str(directory)))
                    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        import sherpa_onnx
        imported = time.perf_counter()
        model = EXPERIMENT / MODEL_NAME
        configured_provider = (provider + ":" + str(Path(args.provider_config).resolve())
                               if args.provider_config else provider)
        recognizer = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=str(model / "encoder.int8.onnx"),
            decoder=str(model / "decoder.int8.onnx"), tokens=str(model / "tokens.txt"),
            language="en", use_punct=True, use_itn=False, num_threads=4, provider=configured_provider)
        result = {"backend": args.backend, "version": sherpa_onnx.__version__,
                  "import_seconds": imported - started,
                  "model_seconds": time.perf_counter() - imported,
                  "ready_seconds": time.perf_counter() - started}

        def decode(samples, sr):
            if len(samples) / sr > 35:
                raise ValueError("native probe clips must be <=35s; no alternate long-audio splitter")
            stream = recognizer.create_stream()
            stream.accept_waveform(sr, samples)
            recognizer.decode_stream(stream)
            return stream.result.text.strip()

    # Emit startup before decoding so a decoder failure still preserves this measurement.
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    if args.manifest:
        import soundfile as sf
        records = []
        for clip in json.loads(Path(args.manifest).read_text(encoding="utf-8")):
            samples, sr = sf.read(clip["path"], dtype="float32")
            if samples.ndim != 1 or sr != 16000:
                raise ValueError("probe clips must be mono 16kHz")
            beginning = time.perf_counter()
            text = decode(samples, sr)
            records.append({**clip, "text": text, "audio_seconds": len(samples) / sr,
                            "decode_seconds": time.perf_counter() - beginning})
        result["clips"] = records
        print(RESULT_PREFIX + json.dumps(result), flush=True)


def monitored_probe(args):
    import psutil
    status = ROOT / "state" / "status.json"
    if status.exists():
        try:
            live = json.loads(status.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            live = {}
        if live.get("active") and time.time() - live.get("updated_at", 0) < 90:
            raise RuntimeError("stop the live recorder before benchmarking its model")
    initial_ram = available_ram_gib()
    minimum_start = 2 if args.backend == "hf" else 4
    if initial_ram < minimum_start:
        raise RuntimeError(f"need {minimum_start} GiB available RAM; found {initial_ram:.2f}")
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                              *sys.argv[1:], "--child"], cwd=ROOT, env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace",
                             creationflags=subprocess.CREATE_NO_WINDOW)
    output, errors = [], []

    def drain(pipe, lines):
        for line in pipe:
            lines.append(line)

    readers = [threading.Thread(target=drain, args=(child.stdout, output)),
               threading.Thread(target=drain, args=(child.stderr, errors))]
    for reader in readers:
        reader.start()
    root = psutil.Process(child.pid)
    started = time.monotonic()
    minimum_ram, peak_rss = initial_ram, 0
    stopped = None
    descendants = []
    try:
        while child.poll() is None:
            try:
                descendants = root.children(recursive=True)
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in descendants + [root]
                                            if p.is_running()))
            except psutil.Error:
                pass
            minimum_ram = min(minimum_ram, available_ram_gib())
            if minimum_ram < 1.2:
                stopped = "available RAM fell below 1.2 GiB"
            elif time.monotonic() - started > args.timeout:
                stopped = f"probe exceeded {args.timeout:g}s"
            if stopped:
                break
            time.sleep(0.1)
    finally:
        if child.poll() is None:
            for process in reversed([root] + descendants):
                try:
                    process.kill()
                except psutil.Error:
                    pass
        child.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)
    measurements = [json.loads(line[len(RESULT_PREFIX):]) for line in output
                    if line.startswith(RESULT_PREFIX)]
    result = {"backend": args.backend, "baseline": bool(args.baseline_dir),
              "returncode": child.returncode, "stopped": stopped,
              "initial_available_ram_gib": initial_ram, "min_available_ram_gib": minimum_ram,
              "peak_process_tree_rss_gib": peak_rss / 2**30,
              "measurements": measurements, "stderr_tail": "".join(errors).replace("\x00", "")[-3000:]}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Do not print lecture transcripts into the console when an output artifact was saved.
    summary = {**result, "measurements": [{k: v for k, v in measurement.items() if k != "clips"}
                                         for measurement in measurements]}
    print(json.dumps(summary, indent=2))
    return 1 if stopped or child.returncode else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "sherpa-cpu", "sherpa-cuda"], default="hf")
    parser.add_argument("--menu-seconds", type=float, default=3)
    parser.add_argument("--no-menu-preparation", action="store_true",
                        help="Control run: import transcription dependencies only after choosing Record")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--provider-config", help="Optional sherpa-onnx provider/session config file")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.child:
        worker(options)
    else:
        sys.exit(monitored_probe(options))

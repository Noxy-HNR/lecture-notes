"""Offline contract checks: never touch an installed manager, model, or user settings."""
import ast
import hashlib
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recorder_gpu_isolated", ROOT / "src/gpu_formatter.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
ORIGINAL_REQUEST = gpu._request_json


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    model = tmp_path / gpu.MODEL_FILENAME
    model.write_bytes(b"fake, not a GGUF")
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"activeModel": "another-model.gguf", "models": {
        gpu.MODEL_FILENAME: {"extraArgs": "--load-mode none --cache-ram 0",
                             "minimumAvailableRamGiB": 3.3}}}))
    monkeypatch.setattr(gpu, "MODEL_PATH", model)
    monkeypatch.setattr(gpu, "MODELS_CONFIG_PATH", config)
    monkeypatch.setattr(gpu, "_available_physical_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(gpu, "_port_open", lambda: False)
    monkeypatch.setattr(gpu, "_manager_port_open", lambda: True)
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [])
    monkeypatch.setattr(gpu, "_process_identity", lambda pid: (pid, 789))
    monkeypatch.setattr(gpu, "_process_details", lambda pid: (
        "C:/AI/Tools/llama-native/bin/llama-server.exe",
        ["-m", str(gpu.MODEL_PATH), "--load-mode", "none", "--cache-ram", "0"]))
    monkeypatch.setattr(gpu, "_request_json", lambda *a, **kw: pytest.fail("Unexpected HTTP"))
    yield config


def props():
    return {"model_path": str(gpu.MODEL_PATH), "default_generation_settings": {"n_ctx": 32768}}


def test_available_does_not_need_old_model_or_private_executable():
    assert gpu.available()
    assert gpu.PORT == 10000


@pytest.mark.parametrize("extra", ["", "--load-mode mmap --cache-ram 0", "--load-mode none --cache-ram 8",
                                  "--load-mode none --cache-ram 0 --cache-ram 8"])
def test_missing_safety_flags_refused(offline, extra):
    raw = json.loads(offline.read_text())
    raw["models"][gpu.MODEL_FILENAME]["extraArgs"] = extra
    offline.write_text(json.dumps(raw))
    assert not gpu.available()
    assert not gpu.warm_up()


@pytest.mark.parametrize("minimum", [True, "3.3", -1, 0, float("inf")])
def test_invalid_saved_ram_floor_refused(offline, minimum):
    raw = json.loads(offline.read_text())
    raw["models"][gpu.MODEL_FILENAME]["minimumAvailableRamGiB"] = minimum
    offline.write_text(json.dumps(raw))
    assert not gpu.available()


def test_reuse_exact_model_without_manager_load(monkeypatch):
    monkeypatch.setattr(gpu, "_port_open", lambda: True)
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
    calls = []
    def request(base, path, *a, **kw):
        calls.append((base, path))
        return {"/props": props(), "/health": {"status": "ok"},
                "/slots": [{"is_processing": False}]}[path]
    monkeypatch.setattr(gpu, "_request_json", request)
    assert gpu.warm_up()
    assert len(calls) == 3 and all(base == gpu.BASE_URL for base, _ in calls)


@pytest.mark.parametrize("change", ["wrong_model", "small_context", "busy", "empty_slots", "unknown_slot"])
def test_busy_wrong_or_unverifiable_server_never_evicted(monkeypatch, change):
    monkeypatch.setattr(gpu, "_port_open", lambda: True)
    p = props()
    if change == "wrong_model": p["model_path"] = "C:/AI/models/another-model.gguf"
    if change == "small_context": p["default_generation_settings"]["n_ctx"] = 4096
    slot = [{"is_processing": change == "busy"}]
    if change == "empty_slots": slot = []
    if change == "unknown_slot": slot = [{}]
    def request(base, path, *a, **kw):
        assert base == gpu.BASE_URL
        return {"/props": p, "/health": {"status": "ok"}, "/slots": slot}[path]
    monkeypatch.setattr(gpu, "_request_json", request)
    assert not gpu.warm_up()


@pytest.mark.parametrize("pids", [[123], [123, 456], None])
def test_stuck_or_unknown_process_inventory_refuses_load(monkeypatch, pids):
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: pids)
    assert not gpu.warm_up()


def test_insufficient_ram_prevents_manager_call(monkeypatch):
    monkeypatch.setattr(gpu, "_available_physical_bytes", lambda: 3 * 2**30)
    assert not gpu.warm_up()


def test_manager_load_only_when_idle_no_persistence(monkeypatch, offline):
    original = offline.read_bytes()
    calls = []
    def request(base, path, payload=None, **kw):
        calls.append((base, path, payload, kw))
        if path.endswith("preflight"):
            return {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": 3.3}}
        assert path.endswith("load")
        monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
        return {"success": True, "pid": 123}
    monkeypatch.setattr(gpu, "_request_json", request)
    monkeypatch.setattr(gpu, "_server_healthy", lambda: True)
    assert gpu.warm_up()
    assert calls[1][2] == {"filename": gpu.MODEL_FILENAME, "persist": False,
                           "onlyIfIdle": True, "settings": {"contextSize": 32768}}
    assert calls[1][3]["timeout"] == gpu.LOAD_TIMEOUT_SECONDS
    assert offline.read_bytes() == original


@pytest.mark.parametrize("reply", [
    {"success": False}, {"success": True},
    {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": False, "requiredGiB": 3}},
    {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": 20}},
    {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": float("nan")}},
])
def test_shared_manager_preflight_refusals(monkeypatch, reply):
    def request(base, path, *a, **kw):
        assert path.endswith("preflight")
        return reply
    monkeypatch.setattr(gpu, "_request_json", request)
    assert not gpu.warm_up()


def test_race_after_preflight_never_posts_load(monkeypatch):
    inv = iter([[], [123]])
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: next(inv))
    def request(base, path, *a, **kw):
        assert path.endswith("preflight")
        return {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": 3.3}}
    monkeypatch.setattr(gpu, "_request_json", request)
    assert not gpu.warm_up()


@pytest.mark.parametrize("reply", [{"success": False}, [], None])
def test_failed_or_malformed_load_response_refuses(monkeypatch, reply):
    def request(base, path, *a, **kw):
        return ({"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": 3.3}}
                if path.endswith("preflight") else reply)
    monkeypatch.setattr(gpu, "_request_json", request)
    assert not gpu.warm_up()


def test_malformed_config_falls_back(offline):
    offline.write_text("[]")
    assert not gpu.available()
    assert not gpu.warm_up()


def test_malformed_running_props_falls_back_without_manager(monkeypatch):
    monkeypatch.setattr(gpu, "_port_open", lambda: True)
    monkeypatch.setattr(gpu, "_request_json", lambda *a, **kw: [])
    assert not gpu.warm_up()


def test_timeout_refuses_without_stopping_shared_server(monkeypatch):
    def request(*a, **kw):
        raise TimeoutError("fixture deadline")
    monkeypatch.setattr(gpu, "_request_json", request)
    assert not gpu.warm_up()
    gpu.stop_server()  # No HTTP or process termination on exit/idle.


def test_startup_deadline_does_not_stop_unverifiable_manager_child(monkeypatch):
    def request(base, path, *a, **kw):
        return ({"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed": True, "requiredGiB": 3.3}}
                if path.endswith("preflight") else {"success": True})
    monkeypatch.setattr(gpu, "_request_json", request)
    monkeypatch.setattr(gpu, "_server_healthy", lambda: False)
    monkeypatch.setattr(gpu, "STARTUP_TIMEOUT_SECONDS", .001)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    assert not gpu.warm_up()


@pytest.mark.parametrize("finish,content,expected", [("stop", " Notes ", "Notes"),
    ("length", "Partial", None), ("stop", "", None), ("tool_calls", "", None)])
def test_chat_payload_and_truncation(monkeypatch, finish, content, expected):
    monkeypatch.setattr(gpu, "_ensure_server_running", lambda: True)
    monkeypatch.setattr(gpu, "_selected_server_idle", lambda: True)
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
    def request(base, path, payload=None, **kw):
        assert path == "/v1/chat/completions"
        assert payload["messages"] == [{"role": "system", "content": "SYSTEM"},
                                       {"role": "user", "content": "SOURCE"}]
        assert payload["max_tokens"] == 777
        assert payload["cache_prompt"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert kw["process_identity"] == (123, 789) and kw["watch_ram"]
        return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    monkeypatch.setattr(gpu, "_request_json", request)
    assert gpu._chat("SYSTEM", "SOURCE", 777) == expected


def test_stop_is_bookkeeping_only(monkeypatch):
    monkeypatch.setattr(gpu, "_last_used", 123.0)
    gpu.stop_server()
    assert gpu._last_used == 0


def test_shutdown_bookkeeping_never_waits_for_inference_lock():
    acquired = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    def hold_inference():
        with gpu._server_lock:
            acquired.set()
            release.wait(1)
    holder = threading.Thread(target=hold_inference)
    holder.start()
    assert acquired.wait(1)
    caller = threading.Thread(target=lambda: (gpu.stop_server(),finished.set()))
    caller.start()
    try:
        assert finished.wait(.2), "Shutdown must not wait for a shared inference request"
    finally:
        release.set()
        holder.join(1)
        caller.join(1)


@pytest.mark.parametrize("mismatch", ["pid", "creation", "model", "missing"])
def test_emergency_stop_refuses_replacement_or_unknown_process(monkeypatch, mismatch):
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [999] if mismatch == "pid" else [123])
    monkeypatch.setattr(gpu, "_process_identity", lambda pid: (pid, 999) if mismatch == "creation" else (pid, 789))
    if mismatch == "model":
        monkeypatch.setattr(gpu, "_process_details", lambda pid: (
            "C:/AI/Tools/llama-native/bin/llama-server.exe", ["-m", "C:/AI/models/another.gguf"]))
    assert not gpu._emergency_stop(None if mismatch == "missing" else (123, 789))


def test_emergency_stop_uses_manager_expected_pid(monkeypatch):
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
    calls = []
    def request(base, path, payload=None, **kw):
        calls.append((base, path, payload))
        assert path != "/props", "Emergency cleanup must not depend on inference readiness"
        return {"success": True}
    monkeypatch.setattr(gpu, "_request_json", request)
    assert gpu._emergency_stop((123, 789))
    assert calls[-1] == (gpu.MANAGER_URL, "/api/stop", {"service": "llama-server", "expectedPid": 123})


def test_source_fidelity_prompt_and_public_formatting(monkeypatch):
    after = (ROOT / "src/gpu_formatter.py").read_text(encoding="utf-8")
    # Root strengthened this prompt after live semantic review found added facts.
    assert 'Preserve every stated fact, negation' in gpu._FORMAT_SYSTEM_PROMPT
    assert 'Do not add definitions, classifications' in gpu._FORMAT_SYSTEM_PROMPT
    assert '**Q (Student):**' in gpu._FORMAT_SYSTEM_PROMPT
    monkeypatch.setattr(gpu, "_chat", lambda system, user, max_tokens: "- **Source** stays.")
    assert gpu.format_transcript("Biology", "2026-10-16", "Source") == "## 2026-10-16\n\n- **Source** stays."
    assert gpu.condense_transcript("Source") == "- **Source** stays."


@pytest.mark.parametrize('output,expected', [
    ('### Q&A\n**Q (Student):** Why?\n**A (Instructor):** Unknown.', '## 2026-10-16\n\n### Q&A\n**Q (Student):** Why?\n**A (Instructor):** Unknown.'),
    ('## 2026-10-16\n\n- Fact.', '## 2026-10-16\n\n- Fact.'),
    ('## Invented date\n- Fact.', None),
    ('# Unrequested title\n- Fact.', None),
    ('- Fact.\n## Second boundary\n- Other.', None),
    ('- Fact.\n # Second boundary\n- Other.', None),
    ('- Fact.\n  ## Second boundary\n- Other.', None),
    ('- Fact.\n   ## Second boundary\n- Other.', None),
    ('- Fact.\n##', None),
])
def test_exact_session_heading_and_no_fabricated_boundaries(monkeypatch, output, expected):
    monkeypatch.setattr(gpu,'_chat',lambda *a,**kw:output)
    assert gpu.format_transcript('Biology','2026-10-16','Source')==expected


def test_listener_table_avoids_windows_delayed_connection_refusal(monkeypatch):
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil,'net_connections',lambda **kw:[])
    monkeypatch.setattr(socket,'create_connection',lambda *a,**kw:pytest.fail('No TCP timeout probe'))
    assert gpu._listening_port(10000) is False
    listener=SimpleNamespace(status=psutil.CONN_LISTEN,laddr=SimpleNamespace(ip='0.0.0.0',port=9090))
    monkeypatch.setattr(psutil,'net_connections',lambda **kw:[listener])
    assert gpu._listening_port(9090) is True
    assert gpu._listening_port(10000) is False


def test_listener_inventory_failure_is_not_permission_to_launch(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil,'net_connections',lambda **kw:(_ for _ in ()).throw(OSError('inventory failed')))
    assert gpu._listening_port(9090) is True


def test_no_independent_model_process_launch_or_termination():
    tree = ast.parse((ROOT / "src/gpu_formatter.py").read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("atexit" in ast.unparse(n) for n in imports)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(name.endswith((".terminate", ".kill")) for name in calls)
    launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".Popen")]
    assert len(launches) == 1
    command = ast.unparse(launches[0].args[0])
    assert "MANAGER_SHELL" in command and "MANAGER_SCRIPT" in command
    assert "LLAMA" not in command and "MODEL_PATH" not in command
    assert ast.unparse(next(k.value for k in launches[0].keywords if k.arg == "creationflags")) == "subprocess.CREATE_NO_WINDOW"


def test_caption_job_helpers_preserved_byte_for_byte():
    after = (ROOT / "src/gpu_formatter.py").read_text(encoding="utf-8")
    helpers = after[after.index("def _attach_windows_kill_job"):after.index("_FORMAT_SYSTEM_PROMPT =")]
    assert hashlib.sha256(helpers.encode()).hexdigest() == "f8a76df0cb6441de8364a836b6d63ee9bd22fd62565251f57b464f9de5c1c11c"


@pytest.mark.parametrize("reason", ["deadline", "low_ram"])
def test_http_guard_interrupts_socket_without_live_network(monkeypatch, reason):
    # Call the saved implementation while all process/network primitives are fake.
    original = ORIGINAL_REQUEST
    done = threading.Event()
    emergency = []
    class Sock:
        def shutdown(self, how): done.set()
    class Connection:
        sock = Sock()
        def __init__(self, *a, **kw): pass
        def request(self, *a, **kw): pass
        def getresponse(self):
            done.wait(1)
            raise OSError("interrupted fixture socket")
        def close(self): pass
    monkeypatch.setattr(gpu.http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(gpu, "_WATCHDOG_POLL_SECONDS", .002)
    monkeypatch.setattr(gpu, "_RAM_CRITICAL_SECONDS", .005)
    monkeypatch.setattr(gpu, "_available_physical_bytes", lambda: 0)
    monkeypatch.setattr(gpu, "_emergency_stop", emergency.append)
    with pytest.raises(OSError):
        original(gpu.BASE_URL, "/v1/chat/completions", {}, timeout=.03 if reason == "deadline" else .5,
                 watch_ram=reason == "low_ram", process_identity=(123, 789))
    assert done.is_set()
    assert emergency == ([(123, 789)] if reason == "low_ram" else [])


def test_emergency_stop_survives_hung_inference_props(monkeypatch):
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
    calls = []
    def request(base, path, payload=None, **kw):
        if base == gpu.BASE_URL:
            raise TimeoutError("Inference is deliberately hung")
        calls.append((path, payload))
        return {"success": True}
    monkeypatch.setattr(gpu, "_request_json", request)
    assert gpu._emergency_stop((123,789))
    assert calls == [("/api/stop", {"service":"llama-server", "expectedPid":123})]


@pytest.mark.parametrize("case", ["wrong_binary", "unsafe_cache", "duplicate_model", "missing_model"])
def test_os_identity_proof_refuses_unapproved_process(monkeypatch, case):
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123])
    args = ["-m", str(gpu.MODEL_PATH), "--load-mode", "none", "--cache-ram", "0"]
    if case == "unsafe_cache": args[-1] = "8"
    if case == "duplicate_model": args += ["--model", str(gpu.MODEL_PATH)]
    if case == "missing_model": args = args[2:]
    monkeypatch.setattr(gpu, "_process_details", lambda pid: (
        "C:/wrong/llama-server.exe" if case == "wrong_binary" else
        "C:/AI/Tools/llama-native/bin/llama-server.exe", args))
    assert not gpu._verified_process((123,789))
    assert not gpu._emergency_stop((123,789))


@pytest.mark.parametrize("cause", ["critical_ram", "deadline", "pid_reused"])
def test_startup_guard_cleans_exact_requested_child_only(monkeypatch, cause):
    loaded = threading.Event()
    stopped = threading.Event()
    calls = []
    identity_reads = []
    def identity(pid):
        identity_reads.append(pid)
        return (pid, 999 if cause == "pid_reused" and len(identity_reads) > 1 else 789)
    monkeypatch.setattr(gpu, "_process_identity", identity)
    def request(base, path, payload=None, **kw):
        calls.append((base, path, payload))
        if path.endswith("preflight"):
            return {"success": True, "filename": gpu.MODEL_FILENAME, "admission": {"allowed":True,"requiredGiB":3.3}}
        if path.endswith("load"):
            loaded.set()
            return {"success":True,"pid":123}
        assert path == "/api/stop"
        assert payload == {"service":"llama-server","expectedPid":123}
        stopped.set()
        return {"success":True}
    monkeypatch.setattr(gpu, "_request_json", request)
    monkeypatch.setattr(gpu, "_llama_process_ids", lambda: [123] if loaded.is_set() else [])
    monkeypatch.setattr(gpu, "_available_physical_bytes", lambda:
        0 if loaded.is_set() and cause == "critical_ram" else 8*2**30)
    monkeypatch.setattr(gpu, "_server_healthy", lambda: False)
    monkeypatch.setattr(gpu, "STARTUP_TIMEOUT_SECONDS", .03 if cause != "critical_ram" else .5)
    monkeypatch.setattr(gpu, "_WATCHDOG_POLL_SECONDS", .002)
    monkeypatch.setattr(gpu, "_RAM_CRITICAL_SECONDS", .005)
    assert not gpu.warm_up()
    assert loaded.is_set()
    assert stopped.is_set() is (cause != "pid_reused")
    assert "Startup" in gpu._last_error
    assert sum(path == "/api/stop" for _,path,_ in calls) == (0 if cause == "pid_reused" else 1)


def test_startup_timeout_missing_pid_leaves_explicit_diagnostic(monkeypatch):
    def request(base,path,*a,**kw):
        if path.endswith("preflight"):
            return {"success":True,"filename":gpu.MODEL_FILENAME,"admission":{"allowed":True,"requiredGiB":3.3}}
        assert path.endswith("load")
        return {"success":True}
    monkeypatch.setattr(gpu,"_request_json",request)
    assert not gpu.warm_up()
    assert "no exact child PID" in gpu._last_error


def _preflight_reply():
    return {"filename":gpu.MODEL_FILENAME, "success":True,
            "admission":{"allowed":True,"requiredGiB":3.3}}


def _bootstrap_fixture(monkeypatch,tmp_path,*,ready=True,owned=True):
    started = threading.Event()
    launches = []
    monkeypatch.setattr(gpu,"MANAGER_STATE_DIR",tmp_path/"manager-state")
    monkeypatch.setattr(gpu,"_manager_port_open",lambda:False)
    monkeypatch.setattr(gpu,"_management_paths_verified",lambda:True)
    monkeypatch.setattr(gpu,"_acquire_manager_bootstrap",lambda:("fake-mutex" if owned else None,owned))
    monkeypatch.setattr(gpu,"_release_manager_bootstrap",lambda handle:None)
    def request(base,path,payload=None,**kw):
        assert base == gpu.MANAGER_URL and path == "/api/models/preflight"
        assert payload == {"filename":gpu.MODEL_FILENAME}
        if ready and started.is_set(): return _preflight_reply()
        raise ConnectionRefusedError("Fixture manager unavailable")
    monkeypatch.setattr(gpu,"_request_json",request)
    class Manager:
        pid = 555
        def poll(self): return None
    def popen(command,**kw):
        launches.append((command,kw))
        started.set()
        return Manager()
    monkeypatch.setattr(gpu.subprocess,"Popen",popen)
    monkeypatch.setattr(gpu,"MANAGER_STARTUP_TIMEOUT_SECONDS",.04 if not ready else .5)
    return started,launches


def test_manager_bootstrap_uses_only_exact_hidden_powershell_command(monkeypatch,tmp_path):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path)
    assert gpu._ensure_manager(3.3*2**30) == _preflight_reply()
    assert len(launches)==1
    command,options=launches[0]
    assert command == ["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        "-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File","C:\\AI\\dashboard-server.ps1"]
    assert options["creationflags"] == gpu.subprocess.CREATE_NO_WINDOW
    assert options["stdin"] == gpu.subprocess.DEVNULL and options["close_fds"] is True
    assert options["cwd"] == "C:\\AI"
    files=list(gpu.MANAGER_STATE_DIR.glob("*.json"))
    assert len(files)==1
    record=json.loads(files[0].read_text())
    assert record["pid"]==555 and record["creation_time"]==789
    assert Path(record["stdout"]).is_file() and Path(record["stderr"]).is_file()


def test_existing_expected_manager_never_starts_second_controller(monkeypatch):
    monkeypatch.setattr(gpu,"_request_json",lambda *a,**kw:_preflight_reply())
    monkeypatch.setattr(gpu.subprocess,"Popen",lambda *a,**kw:pytest.fail("No second manager"))
    assert gpu._ensure_manager(3.3*2**30) == _preflight_reply()


@pytest.mark.parametrize("cause",["unknown_port","busy","low_ram","unsafe_path"])
def test_manager_bootstrap_refuses_unsafe_start(monkeypatch,tmp_path,cause):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path)
    if cause=="unknown_port":monkeypatch.setattr(gpu,"_manager_port_open",lambda:True)
    if cause=="busy":monkeypatch.setattr(gpu,"_llama_process_ids",lambda:[123])
    if cause=="low_ram":monkeypatch.setattr(gpu,"_available_physical_bytes",lambda:2**30)
    if cause=="unsafe_path":monkeypatch.setattr(gpu,"_management_paths_verified",lambda:False)
    assert gpu._ensure_manager(3.3*2**30) is None
    assert not launches
    assert "Shared manager unavailable" in gpu._last_error


def test_manager_bootstrap_timeout_never_requests_model_load(monkeypatch,tmp_path):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path,ready=False)
    assert not gpu.warm_up()
    assert len(launches)==1
    assert "no model load" in gpu._last_error
    gpu.stop_server()  # Must not terminate or signal even the controller launched here.


def test_manager_bootstrap_mutex_waiter_does_not_launch_controller(monkeypatch,tmp_path):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path,ready=False,owned=False)
    assert gpu._ensure_manager(3.3*2**30) is None
    assert not launches


def test_unrecognized_manager_preflight_refused(monkeypatch):
    monkeypatch.setattr(gpu,"_request_json",lambda *a,**kw:{**_preflight_reply(),"filename":"other.gguf"})
    monkeypatch.setattr(gpu.subprocess,"Popen",lambda *a,**kw:pytest.fail("Unknown manager must be left alone"))
    assert gpu._ensure_manager(3.3*2**30) is None
    assert "unverified" in gpu._last_error


def test_management_file_verification_rejects_symlink(monkeypatch,tmp_path):
    class Unsafe:
        def is_file(self):return True
        def absolute(self):return self
        def resolve(self,strict=False):return self
        def __str__(self):return "C:/AI/dashboard-server.ps1"
        @property
        def parents(self):return ()
        def lstat(self):return type("Stat",(),{"st_file_attributes":0x400})()
        def is_symlink(self):return False
    monkeypatch.setattr(gpu,"MANAGER_SCRIPT",Unsafe())
    assert not gpu._management_paths_verified()


def test_persisted_own_manager_record_prevents_duplicate_retry(monkeypatch,tmp_path):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path,ready=False)
    gpu.MANAGER_STATE_DIR.mkdir()
    (gpu.MANAGER_STATE_DIR/"manager-123.json").write_text(json.dumps({"pid":555,"creation_time":789}))
    arguments=[str(gpu.MANAGER_SHELL),"-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass",
               "-File",str(gpu.MANAGER_SCRIPT)]
    monkeypatch.setattr(gpu,"_process_details",lambda pid:(str(gpu.MANAGER_SHELL),arguments))
    assert gpu._recorded_manager_running()
    assert gpu._ensure_manager(3.3*2**30) is None
    assert not launches


def test_pid_reused_manager_record_does_not_claim_ownership(monkeypatch,tmp_path):
    _,launches=_bootstrap_fixture(monkeypatch,tmp_path)
    gpu.MANAGER_STATE_DIR.mkdir()
    (gpu.MANAGER_STATE_DIR/"manager-123.json").write_text(json.dumps({"pid":555,"creation_time":888}))
    assert not gpu._recorded_manager_running()
    assert gpu._ensure_manager(3.3*2**30) == _preflight_reply()
    assert len(launches)==1


def test_output_cap_fits_notes_that_keep_the_source_wording():
    # Measured on real lecture chunks: faithful notes run 0.9-1.0x the estimated input tokens.
    words = lambda n: " ".join(["word"] * n)
    for n in (1100, 2600):
        estimated = n * 1.3
        assert gpu._scaled_max_tokens(words(n)) >= estimated * 1.0
    assert gpu._scaled_max_tokens("a few words") == 400
    # Never more than the ceiling, and never more than fits beside the prompt in the context window.
    assert gpu._scaled_max_tokens(words(20000)) <= 8000
    huge = gpu._scaled_max_tokens(words(24000))
    assert 256 <= huge <= gpu.CONTEXT_SIZE - 24000 * 1.3

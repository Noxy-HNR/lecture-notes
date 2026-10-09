"""The dashboard's feature endpoints: access rules and how errors map to HTTP statuses.
The feature modules themselves (quiz, health, lesson_cleanup, lessons.cancel_job_request)
have their own tests; here they are replaced by fakes."""
import http.client
import json
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dashboard


class Conflict(Exception):
    http_status = 409


@pytest.fixture
def server(monkeypatch):
    calls = []

    def fake(name, **functions):
        module = types.ModuleType(name)
        for attr, fn in functions.items():
            setattr(module, attr, fn)
        monkeypatch.setitem(sys.modules, name, module)

    def boom(payload):
        raise RuntimeError("secret internal detail")

    fake("quiz", get_state=lambda lesson: calls.append(("state", lesson)) or {"lesson": lesson},
         review=lambda: {"missed": []},
         record_answer=lambda payload: calls.append(("answer", payload)) or {"ok": True},
         reset=lambda payload: boom(payload))
    fake("health", snapshot=lambda: {"ok": True}, test_claude=lambda payload: {"reachable": True})
    fake("lesson_cleanup", scan=lambda: {"items": []},
         delete=lambda payload: (_ for _ in ()).throw(Conflict("a build is running")))
    monkeypatch.setattr(dashboard.Handler, "log_message", lambda *a, **k: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_port, calls
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request(method, path, body, headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, (json.loads(data) if data.startswith(b"{") or data.startswith(b"[") else data)


OWN_PAGE = {"Content-Type": "application/json", "X-Notes-Dashboard": "1"}


def test_get_routes_call_their_module(server):
    port, calls = server
    assert request(port, "GET", "/api/quiz?lesson=abc") == (200, {"lesson": "abc"})
    assert request(port, "GET", "/api/health") == (200, {"ok": True})
    assert request(port, "GET", "/api/cleanup") == (200, {"items": []})
    assert request(port, "GET", "/api/quiz/review") == (200, {"missed": []})
    assert calls == [("state", "abc")]


def test_posts_need_this_dashboards_own_page(server):
    port, calls = server
    body = json.dumps({"lesson_id": "x"})
    assert request(port, "POST", "/api/quiz/answer", body, {"Content-Type": "application/json"})[0] == 403
    assert request(port, "POST", "/api/quiz/answer", body,
                   {"Content-Type": "application/json", "X-Notes-Dashboard": "1",
                    "Origin": "http://attacker.example"})[0] == 403
    assert request(port, "POST", "/api/quiz/answer", body, {"X-Notes-Dashboard": "1",
                                                          "Content-Type": "text/plain"})[0] == 415
    assert calls == []
    assert request(port, "POST", "/api/quiz/answer", body, OWN_PAGE) == (200, {"ok": True})
    assert calls == [("answer", {"lesson_id": "x"})]


def test_errors_map_to_statuses_without_leaking_internals(server):
    port, _ = server
    status, data = request(port, "POST", "/api/cleanup/delete", "{}", OWN_PAGE)
    assert status == 409 and data == {"error": "a build is running"}
    status, data = request(port, "POST", "/api/quiz/reset", "{}", OWN_PAGE)
    assert status == 500 and "secret" not in json.dumps(data)
    assert request(port, "POST", "/api/quiz/answer", "[1]", OWN_PAGE)[0] == 400   # not a JSON object
    assert request(port, "POST", "/api/quiz/answer", "{bad json", OWN_PAGE)[0] == 400


def test_a_missing_feature_module_is_a_404_not_a_crash(monkeypatch, server):
    port, _ = server
    monkeypatch.setitem(sys.modules, "health", None)  # importing it now raises ImportError
    assert request(port, "GET", "/api/health")[0] == 404

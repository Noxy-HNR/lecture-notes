"""A refused POST must reach the client as its status code, not as a reset connection.

http.client sends a request's headers and its body in two separate send() calls. If the
dashboard answered and closed as soon as it had read the headers, the body could arrive at a
closed socket; Windows then resets the connection and throws away the unread reply, so the
client saw WinError 10053/10054 instead of the 403. That made test_corrections'
test_http_write_origin_and_validation fail about one run in three.

These tests force that timing with a raw socket: send the headers, pause so the server has read
them, then send the body, and check the reply still arrives.
"""
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

import dashboard

FOREIGN = {"Origin": "https://elsewhere.example"}
OURS = {"X-Notes-Dashboard": "1"}


@pytest.fixture
def port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()


def post_with_late_body(port, path, headers, body, content_type="application/json",
                        length=None, pause=0.3, close_after=True):
    """Returns the status code the client read, or the name of the error it got instead."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        lines = [f"POST {path} HTTP/1.1", f"Host: 127.0.0.1:{port}",
                 f"Content-Type: {content_type}", f"Content-Length: {len(body) if length is None else length}"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        time.sleep(pause)  # the server has the headers; the body is still "in flight"
        try:
            sock.sendall(body)
        except OSError:
            pass  # the server already closed - the read below says what the client sees
        reply = b""
        try:
            while chunk := sock.recv(65536):
                reply += chunk
        except OSError as error:
            return type(error).__name__
        return int(reply.split(b" ", 2)[1]) if reply else None
    finally:
        sock.close()


REFUSALS = [
    ("corrections: other site", "/api/corrections/save", FOREIGN, b"{}", "application/json", 403),
    ("corrections: not JSON", "/api/corrections/save", {}, b"a=b", "application/x-www-form-urlencoded", 415),
    ("lessons: other site", "/api/lessons/create", FOREIGN, b"{}", "application/json", 403),
    ("lessons: too large", "/api/lessons/create", {}, b" " * 20_000, "application/json", 400),
    ("jev: not from the dashboard", "/api/jev/settings", {}, b'{"enabled": false}', "application/json", 403),
    ("jev: too large", "/api/jev/settings", OURS, b" " * 5_000, "application/json", 400),
    ("command: not from the dashboard", "/api/command", {}, b'{"action": "stop"}', "application/json", 403),
    ("command: too large", "/api/command", OURS, b" " * 5_000, "application/json", 400),
    ("dashboard stop: not from the dashboard", "/api/dashboard/stop", {}, b"{}", "application/json", 403),
    ("unknown route", "/api/nothing-here", {}, b"{}", "application/json", 404),
]


@pytest.mark.parametrize("name,path,headers,body,content_type,status", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusal_arrives_when_the_body_is_late(port, name, path, headers, body, content_type, status):
    assert post_with_late_body(port, path, headers, body, content_type) == status


def test_a_body_that_never_arrives_does_not_hold_the_reply_forever(port, monkeypatch):
    monkeypatch.setattr(dashboard, "DISCARD_BODY_TIMEOUT_SECONDS", 0.3)
    started = time.monotonic()
    # Claims 100 bytes, sends 10, and keeps the connection open.
    status = post_with_late_body(port, "/api/corrections/save", FOREIGN, b"x" * 10, length=100, pause=0)
    assert status == 403 and time.monotonic() - started < 5


def test_oversized_bodies_are_not_read(port, monkeypatch):
    """Discarding is bounded: a body over the limit is left alone rather than read to the end."""
    reads = []
    real = dashboard.Handler._discard_body

    def spy(self):
        reads.append(self.headers.get("Content-Length"))
        return real(self)

    monkeypatch.setattr(dashboard.Handler, "_discard_body", spy)
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall((f"POST /api/corrections/save HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                      f"Origin: https://elsewhere.example\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {dashboard.DISCARD_BODY_LIMIT + 1}\r\n\r\n").encode())
        started = time.monotonic()
        reply = sock.recv(65536)
    finally:
        sock.close()
    assert reply.startswith(b"HTTP/1.0 403") and time.monotonic() - started < 1
    assert reads == [str(dashboard.DISCARD_BODY_LIMIT + 1)]

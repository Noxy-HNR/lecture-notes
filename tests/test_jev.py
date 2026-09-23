"""Jev client: key lookup, redaction, retries, and what an error may carry. No network: every
call goes through a fake urlopen."""
import io
import json
import urllib.error

import pytest

import jev


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "SWITCHBOARD_ENV", tmp_path / "switchboard.env")
    monkeypatch.setattr(jev, "SETTINGS_PATH", tmp_path / "jev_settings.json")
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    return tmp_path


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def answer(**nouls):
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 120, "output_tokens": 4},
            "answers": {name: {"type": "noul", "noul": p} for name, p in nouls.items()}}


def http_error(code, body=b'{"detail": "echo of the request: secret passage"}', headers=None):
    return urllib.error.HTTPError(jev.API_URL, code, "error", headers or {}, io.BytesIO(body))


QUESTIONS = {"q": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "y", "false": "n"}}}


def test_key_comes_from_switchboard_settings_file(isolated):
    jev.SWITCHBOARD_ENV.write_text("﻿# comment\nOTHER=1\nTYPESAFE_API_KEY=ts_abc123\n", encoding="utf-8")
    assert jev.api_key() == ("ts_abc123", "Switchboard settings")


def test_quoted_key_and_environment_precedence(isolated, monkeypatch):
    jev.SWITCHBOARD_ENV.write_text('TYPESAFE_API_KEY="ts_quoted"\n', encoding="utf-8")
    assert jev.api_key()[0] == "ts_quoted"
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_env")
    assert jev.api_key() == ("ts_env", "environment")


def test_missing_key_is_reported_not_raised(isolated):
    assert jev.api_key() == (None, "missing")
    assert jev.status() == {"enabled": True, "key": "missing", "ready": False}
    assert "no key" in jev.label()


def test_switching_off_persists_and_stops_calls(isolated, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_env")
    jev.save_settings(False)
    assert jev.status()["ready"] is False and jev.label() == "turned off"
    calls = []
    monkeypatch.setattr(jev.urllib.request, "urlopen", lambda *a, **k: calls.append(1))
    with pytest.raises(jev.JevError) as error:
        jev.ask({"passage": "x"}, QUESTIONS)
    assert error.value.kind == "off" and not calls
    assert all(isinstance(r, jev.JevError) for r in jev.ask_many([{}, {}], QUESTIONS)) and not calls
    with pytest.raises(ValueError):
        jev.save_settings("yes")


def test_redaction_removes_contact_details_but_keeps_science():
    text = ("Email me at dr.steele@univ.edu or call 940-555-0123, student ID 123456789. "
            "Avogadro's number is 6.022 times ten to the 23rd, and pH 7.4 at 37 degrees.")
    out = jev.redact(text)
    assert "steele" not in out and "555" not in out and "123456789" not in out
    assert "[email]" in out and "[phone number]" in out and "[number]" in out
    assert "6.022" in out and "7.4" in out and "37 degrees" in out


def test_redaction_is_linear_on_hostile_input():
    import time
    start = time.perf_counter()
    jev.redact("a" * 200_000 + "." * 200_000 + "1" * 200_000)
    assert time.perf_counter() - start < 2


def test_clip_cuts_at_words_from_either_end():
    assert jev.clip("one two three four", 9) == "one two"
    assert jev.clip("one two three four", 10, keep="end") == "four"
    assert jev.clip("  spaced\n  out ", 50) == "spaced out"


def test_ask_sends_state_and_questions_and_reads_usage(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["auth"] = request.headers["Authorization"]
        return FakeResponse(json.dumps(answer(q=0.8)).encode())

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    result = jev.ask({"passage": "hello"}, QUESTIONS, key="ts_k")
    assert seen["body"] == {"model": "jev-latest", "state": {"passage": "hello"}, "questions": QUESTIONS}
    assert seen["auth"] == "Bearer ts_k"
    assert result["input_tokens"] == 120 and jev.noul(result["answers"], "q") == 0.8


def test_rate_limit_is_retried(monkeypatch):
    replies = [http_error(429, headers={"Retry-After": "1"}), FakeResponse(json.dumps(answer(q=0.1)).encode())]

    def urlopen(*a, **k):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    assert jev.noul(jev.ask({}, QUESTIONS, key="k")["answers"], "q") == 0.1
    assert not replies


def test_bad_key_is_not_retried_and_error_never_carries_the_body(monkeypatch):
    calls = []

    def urlopen(*a, **k):
        calls.append(1)
        raise http_error(401)

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    with pytest.raises(jev.JevError) as error:
        jev.ask({"passage": "secret passage"}, QUESTIONS, key="k")
    assert error.value.kind == "auth" and len(calls) == 1
    assert "secret" not in str(error.value) and "401" in str(error.value)


def test_network_failure_gives_up_after_retries(monkeypatch):
    calls = []

    def urlopen(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    with pytest.raises(jev.JevError) as error:
        jev.ask({}, QUESTIONS, key="k")
    assert error.value.kind == "network" and len(calls) == jev.RETRIES + 1


def test_missing_or_out_of_range_answers_are_rejected(monkeypatch):
    monkeypatch.setattr(jev.urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse(json.dumps(answer(other=0.5)).encode()))
    with pytest.raises(jev.JevError) as error:
        jev.ask({}, QUESTIONS, key="k")
    assert error.value.kind == "bad_response"
    for bad in (1.5, -0.1, True, None, "0.5"):
        with pytest.raises(jev.JevError):
            jev.noul({"q": {"noul": bad}}, "q")


def test_ask_many_stops_after_a_rejected_key(isolated, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "bad")
    calls = []

    def urlopen(*a, **k):
        calls.append(1)
        raise http_error(401)

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    results = jev.ask_many([{"i": i} for i in range(30)], QUESTIONS, workers=1)
    assert all(isinstance(r, jev.JevError) and r.kind == "auth" for r in results)
    assert len(calls) == 1


def test_ask_many_keeps_order_and_counts_usage(isolated, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")

    def urlopen(request, timeout):
        i = json.loads(request.data)["state"]["i"]
        return FakeResponse(json.dumps(answer(q=i / 10)).encode())

    monkeypatch.setattr(jev.urllib.request, "urlopen", urlopen)
    usage = jev.Usage()
    results = jev.ask_many([{"i": i} for i in range(10)], QUESTIONS, usage=usage)
    assert [jev.noul(r, "q") for r in results] == [i / 10 for i in range(10)]
    assert usage.as_dict() == {"calls": 10, "input_tokens": 1200,
                               "est_cost_usd": round(1200 * jev.USD_PER_INPUT_MTOK / 1e6, 6)}

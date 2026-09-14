"""Failure-oriented tests, with temporary recordings and no devices or LLM calls."""
import http.client
import json
import sys
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main
import capture
import notes
import search
import dashboard
import storage


def test_audio_survives_failed_transcription_and_offsets_do_not_shift(tmp_path):
    session = main.Session(tmp_path / "TEST_20260911_120000.wav")
    first, second = np.ones(1600, np.float32)*.1, np.ones(1600,np.float32)*.2
    session.persist_audio(first)
    session.persist_audio(second)
    class Transcriber:
        calls = 0
        def process(self, chunk):
            self.calls += 1
            if self.calls <= 2: raise RuntimeError("GPU failure")
            return [{"start":0,"end":.1,"text":"second chunk"}]
    worker = main.TranscriptionWorker(Transcriber(),
        lambda chunk, segments, duration, offset: session.record_segments(segments, offset),
        session.read_audio)
    worker.submit(first)
    worker.submit(second)
    worker.wait_idle()
    session.close()
    recording, rate = sf.read(session.wav_path, dtype="float32")
    np.testing.assert_array_equal(recording, np.concatenate([first,second]))
    assert session.all_segments[0]["start"] == .1
    assert json.loads(session.segments_path.read_text())["start"] == .1


def test_capture_persists_before_queueing_and_stops_on_disk_error(monkeypatch):
    class Recorder:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(capture.audio, "get_recorder", lambda source: Recorder())
    monkeypatch.setattr(capture.audio, "record_chunk", lambda *args: np.zeros(10))
    seen = []
    def persist(block):
        assert recorder.audio_queue.empty()
        seen.append(block)
        raise OSError("disk full")
    recorder = capture.CaptureThread("mic", persist)
    recorder._run()
    assert recorder.fatal_error is not None
    assert recorder.audio_queue.empty()
    assert len(seen) == 1


def test_repeated_recovery_replaces_only_its_session(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(notes, "_local_format", lambda *a: "## Today\n\n" + a[2])
    monkeypatch.setattr(notes.docx_export, "rebuild", lambda *a: None)
    notes.format_and_save("TEST", "Test", "Earlier lecture", "Today", mode="heuristic", session_id="first")
    for _ in range(2):
        notes.format_and_save("TEST", "Test", "Recovered lecture", "Today", mode="heuristic",
                              session_id="second", replace_session=True)
    text = notes.notes_path("TEST").read_text()
    assert text.count("Recovered lecture") == 1
    assert text.count("Earlier lecture") == 1
    assert list((tmp_path / ".revisions" / "TEST.md").glob("*.txt"))


def test_atomic_replace_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "notes.md"
    path.write_text("original")
    def fail(*args): raise OSError("file locked")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError): storage.atomic_write(path, "replacement", backup=False)
    assert path.read_text() == "original"
    assert not list(tmp_path.glob("*.tmp"))


def test_condense_keeps_other_session_and_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(notes,"NOTES_DIR",tmp_path)
    monkeypatch.setattr(notes.docx_export,"rebuild",lambda *a:None)
    text = storage.update_session("# Test\n", "first", "Keep me")
    text = storage.update_session(text, "second", "Condense me")
    path = notes.notes_path("TEST")
    path.write_text(text)
    monkeypatch.setattr(notes,"_try_claude_cli_format",lambda prompt:"Condensed")
    ok, _ = notes.condense_session("TEST","Test",0,session_id="second")
    assert ok
    assert "Keep me" in path.read_text()
    assert "Condensed" in path.read_text()
    assert "Condense me" in next((tmp_path/".revisions"/"TEST.md").glob("*.txt")).read_text()


def test_search_uses_audio_offsets_and_tolerates_partial_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(search,"STATE_DIR",tmp_path)
    stem = "TEST_20260911_120000"
    (tmp_path/(stem+"_raw.txt")).write_text("[12:01:00] delayed words")
    (tmp_path/(stem+".wav")).write_bytes(b"audio")
    (tmp_path/(stem+"_segments.jsonl")).write_text(json.dumps({"text":"delayed words","start":3.5,"end":4})+'\n{"partial"')
    hit, = search.search_transcripts("delayed")
    assert hit.audio_seconds == 3.5
    assert hit.audio_file == stem+".wav"


def test_audio_range_playback_and_path_confinement(tmp_path,monkeypatch):
    monkeypatch.setattr(dashboard,"STATE_DIR",tmp_path)
    filename = "TEST_20260911_120000.wav"
    (tmp_path/filename).write_bytes(bytes(range(100)))
    server = ThreadingHTTPServer(("127.0.0.1",0),dashboard.Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=5)
        conn.request("GET","/api/audio?file="+filename,headers={"Range":"bytes=10-19"})
        response = conn.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 10-19/100"
        assert response.read() == bytes(range(10,20))
        conn.request("GET","/api/audio?file="+filename,headers={"Range":"bytes=100-"})
        response = conn.getresponse()
        assert response.status == 416
        response.read()
        conn.request("GET","/api/audio?file=../private.wav")
        response = conn.getresponse()
        assert response.status == 400
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_class_paths_cannot_escape_library(tmp_path,monkeypatch):
    monkeypatch.setattr(dashboard,"NOTES_DIR",tmp_path)
    assert dashboard._class_files("../private") == {}


def test_recording_does_not_overwrite_existing_audio(tmp_path):
    target = tmp_path / "existing.wav"
    target.write_bytes(b"keep this")
    with pytest.raises(Exception): main.Session(target)
    assert target.read_bytes() == b"keep this"


def test_same_date_navigation_has_unique_anchors():
    html = dashboard.markdown_to_html("## Today\nfirst\n## Today\nsecond\n")
    assert 'id="today"' in html
    assert 'id="today-2"' in html
    assert dashboard.lecture_anchors(["Today","Today"]) == ["today","today-2"]


def test_backup_listing_includes_audio_without_transcript(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main,"STATE_DIR",tmp_path)
    target = tmp_path / "TEST_20260911_120000.wav"
    target.write_bytes(bytes(64))
    main.list_session_backups()
    output = capsys.readouterr().out
    assert target.name in output
    assert "--resume" in output

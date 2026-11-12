"""Animations (downloaded from Wikimedia Commons) and video clips (verified YouTube embeds) in lessons."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import lessons


def test_animation_and_video_visuals_are_validated():
    assert lessons._clean_visual({"type": "animation", "queries": ["mitosis animation", ""], "caption": "c"}) == \
        {"type": "animation", "queries": ["mitosis animation"], "caption": "c"}
    assert lessons._clean_visual({"type": "animation", "queries": []}) == {"type": "none"}
    assert lessons._clean_visual({"type": "video", "find": "a cell dividing", "caption": "c"}) == \
        {"type": "video", "find": "a cell dividing", "search": "a cell dividing", "caption": "c"}
    assert lessons._clean_visual({"type": "video"}) == {"type": "none"}


@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=f-ldPgEfAHI", "f-ldPgEfAHI"),
    ("https://youtu.be/f-ldPgEfAHI?t=30", "f-ldPgEfAHI"),
    ("https://m.youtube.com/watch?v=f-ldPgEfAHI&list=x", "f-ldPgEfAHI"),
    ("https://www.youtube.com/shorts/f-ldPgEfAHI", "f-ldPgEfAHI"),
    ("https://www.youtube.com/embed/f-ldPgEfAHI", "f-ldPgEfAHI"),
    ("https://evil.example/watch?v=f-ldPgEfAHI", None),
    ("https://www.youtube.com/watch?v=short", None),
    ("not a url", None), (None, None),
])
def test_youtube_links_are_parsed_strictly(url, expected):
    assert lessons.youtube_id(url) == expected


def test_clip_ranges_are_bounded():
    assert lessons._clip_range(0, None) == (0, lessons.MAX_VIDEO_CLIP_SECONDS)
    assert lessons._clip_range(30, 150) == (30, 150)
    assert lessons._clip_range(100, 50) == (100, 100 + lessons.MAX_VIDEO_CLIP_SECONDS)
    assert lessons._clip_range("x", 9999) == (0, lessons.MAX_VIDEO_CLIP_SECONDS)


def _gif(frames: int) -> bytes:
    from PIL import Image
    images = [Image.new("RGB", (40, 40), (i * 40 % 255, 0, 0)) for i in range(frames)]
    out = io.BytesIO()
    images[0].save(out, "GIF", save_all=frames > 1, append_images=images[1:])
    return out.getvalue()


def test_only_real_animations_are_stored(tmp_path):
    assert lessons._store_animation(_gif(3), "gif", tmp_path / "slide-01").name == "slide-01.gif"
    with pytest.raises(ValueError):
        lessons._store_animation(_gif(1), "gif", tmp_path / "slide-02")        # a still image
    assert lessons._store_animation(b"\x1a\x45\xdf\xa3rest", "webm", tmp_path / "slide-03").name == "slide-03.webm"
    with pytest.raises(ValueError):
        lessons._store_animation(b"<html>", "webm", tmp_path / "slide-04")


def test_commons_animations_keep_short_loops_and_small_video_copies(monkeypatch):
    def fake(url):
        if "filemime" in url:
            return {"query": {"pages": {
                "1": {"index": 1, "title": "File:Mitosis process.gif", "imageinfo": [{
                    "url": "https://upload/x.gif", "mime": "image/gif", "size": 120000, "width": 600,
                    "metadata": [{"name": "frameCount", "value": 40}, {"name": "duration", "value": 8}]}]},
                "2": {"index": 2, "title": "File:Still.gif", "imageinfo": [{
                    "url": "https://upload/s.gif", "mime": "image/gif", "size": 1000, "width": 600,
                    "metadata": [{"name": "frameCount", "value": 1}]}]}}}}
        return {"query": {"pages": {
            "3": {"index": 1, "title": "File:Clip.ogv", "imageinfo": [{
                "url": "https://upload/c.ogv", "mime": "application/ogg", "size": 9e6, "width": 960,
                "metadata": [{"name": "length", "value": 10.4}]}],
                  "videoinfo": [{"derivatives": [
                      {"src": "https://upload/c.240.webm", "type": "video/webm", "height": 240},
                      {"src": "https://upload/c.480.webm", "type": "video/webm", "height": 480},
                      {"src": "https://upload/c.1080.webm", "type": "video/webm", "height": 1080}]}]},
            "4": {"index": 2, "title": "File:Lecture.webm", "imageinfo": [{
                "url": "https://upload/l.webm", "mime": "video/webm", "size": 5e7, "width": 1280,
                "metadata": [{"name": "length", "value": 3825}]}]}}}}
    monkeypatch.setattr(lessons, "_get_json", fake)
    found = lessons._commons_animations("mitosis animation")
    assert [(c["title"], c["format"], c["url"]) for c in found] == [
        ("Mitosis process.gif", "gif", "https://upload/x.gif"),
        ("Clip.ogv", "webm", "https://upload/c.480.webm")]


def test_videos_are_kept_only_when_youtube_confirms_them(monkeypatch):
    replies = []
    reply = json.dumps({"videos": {"2": {"url": "https://youtu.be/f-ldPgEfAHI", "start": 20, "end": 140},
                                   "4": {"url": "https://www.youtube.com/watch?v=zzzzzzzzzzz"},
                                   "5": {"url": "https://example.com/video"}}})
    monkeypatch.setattr(lessons.notes, "_try_claude_cli_format",
                        lambda prompt, timeout=0, web_search=False: (replies.append(web_search), reply)[1])

    def oembed(video_id):
        if video_id != "f-ldPgEfAHI":
            raise ValueError("no such video")
        return {"title": "Mitosis: The Amazing Cell Process", "channel": "Amoeba Sisters"}
    monkeypatch.setattr(lessons, "youtube_details", oembed)
    requests = [{"slide": n, "title": "t", "find": "f", "search": "s", "caption": f"cap {n}"} for n in (2, 4, 5)]
    found = lessons.find_videos(requests)
    assert replies == [True]                         # the search call, with web search allowed
    assert found == {2: {"type": "video", "youtube_id": "f-ldPgEfAHI", "start": 20, "end": 140,
                         "video_title": "Mitosis: The Amazing Cell Process", "channel": "Amoeba Sisters",
                         "caption": "cap 2"}}


def test_a_failed_video_search_finds_nothing(monkeypatch):
    monkeypatch.setattr(lessons.notes, "_try_claude_cli_format", lambda *a, **k: None)
    assert lessons.find_videos([{"slide": 1, "title": "t", "find": "f", "search": "s", "caption": ""}]) == {}


def test_animation_assets_are_served(tmp_path, monkeypatch):
    monkeypatch.setattr(lessons, "LESSONS_DIR", tmp_path)
    (tmp_path / "L1").mkdir()
    (tmp_path / "L1" / "lesson.json").write_text("{}", encoding="utf-8")
    for name in ("slide-03.gif", "slide-04.webm"):
        (tmp_path / "L1" / name).write_bytes(b"x")
        assert lessons.lesson_asset("L1", name).name == name
    with pytest.raises(FileNotFoundError):
        lessons.lesson_asset("L1", "slide-03.exe")


def test_web_search_only_unlocks_web_search(monkeypatch):
    import notes, types
    calls = []
    monkeypatch.setattr(notes, "_find_claude_cli", lambda: "claude")
    monkeypatch.setattr(notes.subprocess, "run",
                        lambda args, **kw: (calls.append(args), types.SimpleNamespace(returncode=0, stdout="{}", stderr=""))[1])
    notes._try_claude_cli_format("p", web_search=True)
    notes._try_claude_cli_format("p")
    searching, plain = calls
    assert searching[searching.index("--tools") + 1] == "WebSearch"
    assert "WebSearch" not in searching[searching.index("--disallowedTools") + 1].split()
    assert "Bash" in searching[searching.index("--disallowedTools") + 1].split()
    assert plain[plain.index("--tools") + 1] == ""
    assert "WebSearch" in plain[plain.index("--disallowedTools") + 1].split()


def test_a_throttled_download_waits_and_retries(monkeypatch):
    import urllib.error
    calls = []

    class Response:
        headers = type("H", (), {"get_content_type": staticmethod(lambda: "image/gif")})()
        def read(self, n): return b"GIF89a..."
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "1"}, None)
        return Response()
    monkeypatch.setattr(lessons.urllib.request, "urlopen", urlopen)
    slept = []
    monkeypatch.setattr(lessons.time, "sleep", slept.append)
    assert lessons._http_get("https://upload.wikimedia.org/x.gif") == (b"GIF89a...", "image/gif")
    assert len(calls) == 2 and slept == [1.0]


def test_a_clip_is_downloaded_as_just_the_chosen_part(tmp_path, monkeypatch):
    import types
    seen = {}

    class FakeDownloader:
        def __init__(self, options): seen.update(options)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def download(self, urls):
            seen["urls"] = urls
            (tmp_path / "slide-02.mp4").write_bytes(b"\x00" * 20_000)

    fake = types.SimpleNamespace(YoutubeDL=FakeDownloader, utils=types.SimpleNamespace(
        download_range_func=lambda chapters, ranges: ("ranges", ranges)))
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    out = lessons.download_clip("f-ldPgEfAHI", 20, 140, tmp_path / "slide-02")
    assert out.name == "slide-02.mp4"
    assert seen["urls"] == ["https://www.youtube.com/watch?v=f-ldPgEfAHI"]
    assert seen["download_ranges"] == ("ranges", [(20, 140)])
    assert "height<=720" in seen["format"] and seen["max_filesize"] == lessons.MAX_VIDEO_BYTES


def test_a_failed_clip_download_raises(tmp_path, monkeypatch):
    import types

    class NothingDownloaded:
        def __init__(self, options): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def download(self, urls): pass

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=NothingDownloaded, utils=types.SimpleNamespace(download_range_func=lambda *a: None)))
    with pytest.raises(ValueError):
        lessons.download_clip("f-ldPgEfAHI", 0, 60, tmp_path / "slide-02")


def test_clip_files_are_served(tmp_path, monkeypatch):
    monkeypatch.setattr(lessons, "LESSONS_DIR", tmp_path)
    (tmp_path / "L1").mkdir()
    (tmp_path / "L1" / "lesson.json").write_text("{}", encoding="utf-8")
    (tmp_path / "L1" / "slide-02.mp4").write_bytes(b"x")
    assert lessons.lesson_asset("L1", "slide-02.mp4").name == "slide-02.mp4"

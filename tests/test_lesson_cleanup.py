"""Lesson cleanup: what scan() offers and what delete() will and won't touch.

Everything lives under tmp_path; the builders are faked (a set of "alive" job ids), nothing real
is read or removed. Windows junctions are made with _winapi, which needs no administrator rights.
"""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import lesson_cleanup
import lessons

DAY = 86400
LESSON_ID = "CHEM_1450_20260909-20260909_20260909100000"
PLAN_NAME = "CHEM_1450_20260909-20260909_0123456789ab.json"
REAL_UNLINK = os.unlink


@pytest.fixture
def root(tmp_path, monkeypatch):
    folder = tmp_path / "lessons"
    for sub in (".jobs", ".plans", ".quiz"):
        (folder / sub).mkdir(parents=True)
    monkeypatch.setattr(lessons, "LESSONS_DIR", folder)
    monkeypatch.setattr(lessons, "_processes", {})
    return folder


@pytest.fixture
def alive(monkeypatch):
    """Job ids whose builder counts as running."""
    ids = set()
    monkeypatch.setattr(lessons, "_builder_alive", lambda job: job.get("id") in ids)
    return ids


def age(path: Path, days: float):
    stamp = time.time() - days * DAY
    os.utime(path, (stamp, stamp))


def write(path: Path, size: int = 100, days: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    age(path, days)
    age(path.parent, days)          # a folder's own time is what scan() reports for it
    return path


def job(root, status, days=0.0, **fields) -> str:
    job_id = uuid.uuid4().hex
    record = {"id": job_id, "status": status, "stage": "Failed" if status == "failed" else "Working",
              "progress": 0.08, "class_code": "PSYC 1300", "start": "2026-09-30", "end": "2026-10-05",
              "created_at": time.time() - days * DAY, "pid": 1234, **fields}
    path = root / ".jobs" / f"{job_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    write(root / ".jobs" / f"{job_id}.log", 40, days)
    age(path, days)
    return job_id


def lesson(root, lesson_id=LESSON_ID, days=0.0, **fields) -> Path:
    folder = root / lesson_id
    record = {"id": lesson_id, "class_code": "CHEM 1450", "title": "Functional groups",
              "start": "2026-09-09", "end": "2026-09-09", "duration_seconds": 725.0,
              "created_at": time.time() - days * DAY, "slides": [], **fields}
    write(folder / "lesson.json", 0)
    (folder / "lesson.json").write_text(json.dumps(record), encoding="utf-8")
    write(folder / "slide-01.ogg", 5000)
    return folder


def by_key(result):
    return {(item["kind"], item["id"]): item for item in result["items"]}


def junction(link: Path, target: Path):
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError, AttributeError) as error:
        pytest.skip(f"can't make a junction here: {error}")


class TestScan:
    def test_nothing_to_clean_in_an_empty_or_missing_folder(self, root, tmp_path, monkeypatch):
        assert lesson_cleanup.scan() == {"running_build": False, "items": [],
                                         "totals": {"suggested_bytes": 0, "all_bytes": 0}}
        monkeypatch.setattr(lessons, "LESSONS_DIR", tmp_path / "never-made")
        assert lesson_cleanup.scan()["items"] == []

    def test_leftover_build_folders_are_suggested_and_described(self, root, alive):
        write(root / f"{LESSON_ID}.partial" / "slide-01.webp", 2000, days=3)
        item = by_key(lesson_cleanup.scan())[("partial", f"{LESSON_ID}.partial")]
        assert item["suggested"] and not item["protected"] and item["protected_reason"] is None
        assert item["bytes"] == 2000
        assert "CHEM 1450, Sep 9" in item["label"] and item["detail"].startswith("Unfinished build folder from ")

    def test_a_plan_is_suggested_only_once_it_is_too_old_to_reuse(self, root, alive):
        write(root / ".plans" / PLAN_NAME, 300, days=8)
        write(root / ".plans" / "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json", 300, days=2)
        items = by_key(lesson_cleanup.scan())
        stale, recent = items[("plan", PLAN_NAME)], items[("plan", "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json")]
        assert stale["suggested"] and "too old" in stale["detail"]
        assert not recent["suggested"] and "reuses it" in recent["detail"]

    @pytest.mark.parametrize("status, days, suggested", [
        ("ready", 8, True), ("ready", 6, False),
        ("failed", 2, True), ("failed", 0.5, False),
        ("cancelled", 2, True), ("cancelled", 0.5, False),
    ])
    def test_old_build_records_are_suggested_by_status_and_age(self, root, alive, status, days, suggested):
        job_id = job(root, status, days)
        item = by_key(lesson_cleanup.scan())[("job", job_id)]
        assert item["suggested"] is suggested and not item["protected"]
        assert item["bytes"] > 40            # the record and its log are counted together

    def test_finished_lessons_are_listed_with_their_details_but_never_suggested(self, root, alive):
        folder = lesson(root, days=30)
        item = by_key(lesson_cleanup.scan())[("lesson", LESSON_ID)]
        assert item["label"] == "Functional groups · CHEM 1450 · Sep 9 · 12 min"
        assert not item["suggested"] and not item["protected"]
        assert item["bytes"] == sum(f.stat().st_size for f in folder.iterdir())

    def test_unknown_folders_and_dot_folders_are_never_offered(self, root, alive):
        write(root / "notes-backup" / "a.txt")
        write(root / ".hidden" / "lesson.json")
        write(root / "not a lesson id!" / "lesson.json")
        write(root / "BROKEN_1000_20260101-20260101_20260101000000" / "x.ogg")     # no lesson.json: not a lesson
        lesson(root)
        assert list(by_key(lesson_cleanup.scan())) == [("lesson", LESSON_ID)]

    def test_a_lesson_whose_json_cannot_be_read_is_listed_as_not_playable(self, root, alive):
        write(root / LESSON_ID / "lesson.json", 5).write_text("{broken", encoding="utf-8")
        item = by_key(lesson_cleanup.scan())[("lesson", LESSON_ID)]
        assert item["label"] == LESSON_ID and "can't be played" in item["detail"] and not item["suggested"]

    def test_ids_are_plain_names_never_paths(self, root, alive):
        write(root / f"{LESSON_ID}.partial" / "a")
        write(root / ".plans" / PLAN_NAME)
        job(root, "failed", 3)
        lesson(root)
        for item in lesson_cleanup.scan()["items"]:
            assert not any(mark in item["id"] for mark in ("/", "\\", ":", ".."))
            assert not os.path.isabs(item["id"])

    def test_items_come_grouped_by_kind_newest_first(self, root, alive):
        lesson(root)
        job(root, "failed", 3)
        write(root / ".plans" / PLAN_NAME)
        write(root / f"{LESSON_ID}.partial" / "a", days=2)
        write(root / "PSYC_1300_20260101-20260101_20260101000001.partial" / "a", days=1)
        items = lesson_cleanup.scan()["items"]
        assert [item["kind"] for item in items] == ["partial", "partial", "plan", "job", "lesson"]
        assert items[0]["id"].startswith("PSYC") and items[1]["id"].startswith("CHEM")

    def test_totals_count_what_is_suggested_and_everything_listed(self, root, alive):
        write(root / f"{LESSON_ID}.partial" / "a", 1000, days=1)
        write(root / ".plans" / PLAN_NAME, 500, days=1)        # recent: listed, not suggested
        lesson(root)
        scan = lesson_cleanup.scan()
        assert scan["totals"]["suggested_bytes"] == 1000
        assert scan["totals"]["all_bytes"] == sum(item["bytes"] for item in scan["items"])

    def test_a_build_that_is_running_protects_its_files(self, root, alive):
        running = job(root, "running", 0, start="2026-09-09", end="2026-09-09", class_code="CHEM 1450")
        alive.add(running)
        write(root / f"{LESSON_ID}.partial" / "a", days=1)
        write(root / "PSYC_1300_20260101-20260101_20260101000001.partial" / "a", days=9)
        write(root / ".plans" / PLAN_NAME, days=9)
        finished = job(root, "failed", 3)
        scan = lesson_cleanup.scan()
        items = by_key(scan)
        assert scan["running_build"] is True
        own = items[("partial", f"{LESSON_ID}.partial")]
        assert own["protected"] and own["protected_reason"] == "Used by the lesson build running now" and not own["suggested"]
        assert items[("plan", PLAN_NAME)]["protected_reason"] == "Used by the lesson build running now"
        other = items[("partial", "PSYC_1300_20260101-20260101_20260101000001.partial")]
        assert other["protected"] and "wait until it finishes" in other["protected_reason"]
        assert items[("job", running)]["protected"] and not items[("job", finished)]["protected"]

    def test_scan_changes_nothing_even_when_a_builder_has_died(self, root, alive):
        dead = job(root, "running", 0)               # says running, but its builder is gone
        before = {p.name: (p.read_bytes(), p.stat().st_mtime) for p in (root / ".jobs").iterdir()}
        scan = lesson_cleanup.scan()
        assert scan["running_build"] is False
        assert by_key(scan)[("job", dead)]["detail"].startswith("Failed build")
        assert {p.name: (p.read_bytes(), p.stat().st_mtime) for p in (root / ".jobs").iterdir()} == before

    def test_the_real_builder_check_decides_what_counts_as_running(self, root):
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        live = job(root, "running", 0, pid=0)                     # the pid is set once the stand-in is started
        status_file = root / ".jobs" / f"{live}.json"
        # a live builder is one whose command line is "... --job <this build's status file>", so the stand-in is that
        builder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "--job", str(status_file)])
        try:
            record = json.loads(status_file.read_text(encoding="utf-8"))
            record["pid"] = builder.pid
            status_file.write_text(json.dumps(record), encoding="utf-8")
            dead = job(root, "running", 0, pid=gone.pid)
            items = by_key(lesson_cleanup.scan())
            assert items[("job", live)]["protected"] and not items[("job", dead)]["protected"]
        finally:
            builder.kill()
            builder.wait()

    def test_a_record_that_cannot_be_read_is_still_listed(self, root, alive):
        broken = uuid.uuid4().hex
        write(root / ".jobs" / f"{broken}.json", 20, days=3).write_text("{not json", encoding="utf-8")
        age(root / ".jobs" / f"{broken}.json", 3)
        item = by_key(lesson_cleanup.scan())[("job", broken)]
        assert item["label"] == "Incomplete build record" and item["suggested"]

    def test_links_are_not_followed_or_offered(self, root, alive, tmp_path):
        outside = tmp_path / "outside"
        write(outside / "precious.txt", 10)
        junction(root / "LINK_1000_20260101-20260101_20260101000000.partial", outside)
        write(root / f"{LESSON_ID}.partial" / "a")
        assert list(by_key(lesson_cleanup.scan())) == [("partial", f"{LESSON_ID}.partial")]

    def test_a_folder_holding_a_link_is_protected(self, root, alive, tmp_path):
        outside = tmp_path / "outside"
        write(outside / "precious.txt", 10)
        folder = lesson(root)
        junction(folder / "linked", outside)
        item = by_key(lesson_cleanup.scan())[("lesson", LESSON_ID)]
        assert item["protected"] and "link" in item["protected_reason"]


def ref(kind, name):
    return {"kind": kind, "id": name}


def request(*items, confirm=True):
    return {"items": list(items), "confirm": confirm}


class TestDeleteRules:
    @pytest.mark.parametrize("confirm", [False, None, "true", 1, "yes", [True]])
    def test_nothing_is_deleted_without_confirm_being_exactly_true(self, root, alive, confirm):
        plan = write(root / ".plans" / PLAN_NAME, days=9)
        with pytest.raises(ValueError):
            lesson_cleanup.delete({"items": [ref("plan", PLAN_NAME)], "confirm": confirm})
        with pytest.raises(ValueError):
            lesson_cleanup.delete({"items": [ref("plan", PLAN_NAME)]})
        assert plan.exists()

    @pytest.mark.parametrize("payload", [{"confirm": True}, {"confirm": True, "items": "all"},
                                         {"confirm": True, "items": {"kind": "plan"}}, [1], None])
    def test_a_request_that_is_not_a_list_of_items_is_refused(self, root, payload):
        with pytest.raises(ValueError):
            lesson_cleanup.delete(payload)

    def test_an_empty_selection_deletes_nothing(self, root, alive):
        assert lesson_cleanup.delete(request()) == {"deleted": [], "skipped": [], "freed_bytes": 0}

    def test_deletes_each_kind_and_reports_what_was_freed(self, root, alive):
        partial = write(root / f"{LESSON_ID}.partial" / "a", 1000, days=1).parent
        plan = write(root / ".plans" / PLAN_NAME, 300, days=9)
        job_id = job(root, "failed", 3)
        record_bytes = lesson_cleanup.scan()
        expected = sum(item["bytes"] for item in record_bytes["items"])
        result = lesson_cleanup.delete(request(ref("partial", partial.name), ref("plan", PLAN_NAME), ref("job", job_id)))
        assert result["skipped"] == [] and result["freed_bytes"] == expected
        assert result["deleted"] == [ref("partial", partial.name), ref("plan", PLAN_NAME), ref("job", job_id)]
        assert not partial.exists() and not plan.exists()
        assert not (root / ".jobs" / f"{job_id}.json").exists() and not (root / ".jobs" / f"{job_id}.log").exists()
        assert lesson_cleanup.scan()["items"] == []

    def test_deleting_a_finished_lesson_also_removes_its_quiz_answers(self, root, alive):
        keep = lesson(root, "CHEM_1450_20260916-20260916_20260916105209")
        folder = lesson(root)
        write(root / ".quiz" / f"{LESSON_ID}.json", 50)
        write(root / ".quiz" / "CHEM_1450_20260916-20260916_20260916105209.json", 50)
        result = lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))
        assert result["deleted"] == [ref("lesson", LESSON_ID)]
        assert not folder.exists() and not (root / ".quiz" / f"{LESSON_ID}.json").exists()
        assert keep.exists() and (root / ".quiz" / "CHEM_1450_20260916-20260916_20260916105209.json").exists()

    def test_deleting_a_lesson_without_quiz_answers_is_fine(self, root, alive):
        lesson(root)
        assert lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))["deleted"] == [ref("lesson", LESSON_ID)]

    def test_only_the_named_items_are_deleted(self, root, alive):
        keep = write(root / ".plans" / "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json", days=9)
        gone = write(root / ".plans" / PLAN_NAME, days=9)
        lesson(root)
        lesson_cleanup.delete(request(ref("plan", PLAN_NAME)))
        assert keep.exists() and not gone.exists() and (root / LESSON_ID).exists()

    def test_an_item_listed_twice_is_deleted_once(self, root, alive):
        write(root / ".plans" / PLAN_NAME, days=9)
        result = lesson_cleanup.delete(request(ref("plan", PLAN_NAME), ref("plan", PLAN_NAME)))
        assert result["deleted"] == [ref("plan", PLAN_NAME)] and result["skipped"] == []

    def test_an_item_that_is_already_gone_is_skipped_not_an_error(self, root, alive):
        result = lesson_cleanup.delete(request(ref("plan", PLAN_NAME), ref("job", "a" * 32)))
        assert result["deleted"] == [] and len(result["skipped"]) == 2
        assert all("already gone" in skip["reason"] for skip in result["skipped"])

    def test_recent_plans_and_lessons_can_still_be_deleted_when_ticked_by_hand(self, root, alive):
        write(root / ".plans" / PLAN_NAME, days=1)
        lesson(root)
        result = lesson_cleanup.delete(request(ref("plan", PLAN_NAME), ref("lesson", LESSON_ID)))
        assert len(result["deleted"]) == 2

    def test_the_request_is_capped_at_500_items(self, root, alive):
        items = [ref("plan", f"P{n}.json") for n in range(501)]
        with pytest.raises(ValueError):
            lesson_cleanup.delete(request(*items))
        result = lesson_cleanup.delete(request(*items[:500]))
        assert len(result["skipped"]) == 500 and result["deleted"] == []


class TestDeleteNamesNoPaths:
    @pytest.fixture
    def bait(self, root, tmp_path):
        """Real files a path trick could reach: a folder next to lessons/, one inside it, and notes/."""
        notes_file = write(tmp_path / "notes" / "CHEM_1450.md", 50)
        outside = write(tmp_path / "secret" / "keep.txt", 50)
        inside = write(root / "decoy" / "keep.txt", 50)
        state = write(tmp_path / "state" / "BIOL_1440_20260910_121712.wav", 50)
        return [notes_file, outside, inside, state]

    @pytest.mark.parametrize("kind, name", [
        ("lesson", "../notes"), ("lesson", "..\\notes"), ("lesson", ".."), ("lesson", "."), ("lesson", ".jobs"),
        ("lesson", ".quiz"), ("lesson", "decoy"), ("lesson", "C:\\Windows"), ("lesson", "C:"),
        ("lesson", "/etc"), ("lesson", "\\\\server\\share"), ("lesson", ""), ("lesson", "a/b"),
        ("partial", "../secret.partial"), ("partial", "..\\secret.partial"), ("partial", "C:\\x.partial"),
        ("plan", "../CHEM_1450.json"), ("plan", "..\\..\\x.json"), ("plan", "C:\\x.json"), ("plan", "/x.json"),
        ("job", "../" + "a" * 29), ("job", "a" * 31), ("job", "A" * 32), ("job", "a" * 32 + "\\.."),
        ("notes", "CHEM_1450.md"), ("state", "x"), ("", ""), ("lesson", "x" * 500),
    ])
    def test_a_path_or_unknown_kind_is_skipped_and_nothing_outside_changes(self, root, alive, bait, kind, name):
        lesson(root)
        result = lesson_cleanup.delete(request(ref(kind, name)))
        assert result["deleted"] == [] and len(result["skipped"]) == 1
        assert all(path.exists() for path in bait) and (root / LESSON_ID).exists()

    @pytest.mark.parametrize("entry", [None, 7, "lesson", ["lesson", LESSON_ID], {"kind": "lesson"},
                                       {"id": LESSON_ID}, {"kind": 5, "id": LESSON_ID}, {"kind": "lesson", "id": 5},
                                       {"kind": ["lesson"], "id": [LESSON_ID]}])
    def test_malformed_entries_are_skipped(self, root, alive, entry):
        folder = lesson(root)
        result = lesson_cleanup.delete({"items": [entry], "confirm": True})
        assert result["deleted"] == [] and len(result["skipped"]) == 1 and folder.exists()

    def test_the_reply_never_echoes_an_unbounded_or_absolute_name(self, root, alive):
        result = lesson_cleanup.delete(request(ref("lesson", "x" * 5000)))
        assert len(result["skipped"][0]["item"]["id"]) <= 80
        json.dumps(result)

    def test_a_valid_looking_name_that_was_not_scanned_is_not_deleted(self, root, alive, tmp_path):
        outside = write(tmp_path / "BAIT_1000_20260101-20260101_20260101000000" / "a", 5)
        folder = write(root / ".plans" / "folder.json" / "a")      # a folder where a plan file is expected
        result = lesson_cleanup.delete(request(ref("lesson", "BAIT_1000_20260101-20260101_20260101000000"),
                                               ref("plan", "folder.json")))
        assert result["deleted"] == [] and outside.exists() and folder.exists()

    def test_a_junction_is_never_deleted_or_followed(self, root, alive, tmp_path):
        outside = tmp_path / "outside"
        precious = write(outside / "precious.txt", 10)
        name = "LINK_1000_20260101-20260101_20260101000000.partial"
        junction(root / name, outside)
        result = lesson_cleanup.delete(request(ref("partial", name)))
        assert result["deleted"] == [] and precious.exists()

    def test_a_folder_holding_a_junction_is_left_whole(self, root, alive, tmp_path):
        outside = tmp_path / "outside"
        precious = write(outside / "precious.txt", 10)
        folder = lesson(root)
        junction(folder / "linked", outside)
        result = lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))
        assert result["deleted"] == [] and "link" in result["skipped"][0]["reason"]
        assert precious.exists() and (folder / "lesson.json").exists()

    def test_a_symlink_is_never_deleted_or_followed(self, root, alive, tmp_path):
        outside = tmp_path / "outside"
        precious = write(outside / "precious.txt", 10)
        name = "LINK_1000_20260101-20260101_20260101000000.partial"
        try:
            os.symlink(outside, root / name, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"can't make a symlink here: {error}")
        result = lesson_cleanup.delete(request(ref("partial", name)))
        assert result["deleted"] == [] and precious.exists()

    def test_the_plan_cache_folder_being_a_link_leaves_its_target_alone(self, root, alive, tmp_path):
        outside = tmp_path / "other-plans"
        precious = write(outside / PLAN_NAME, 10, days=9)
        (root / ".plans").rmdir()
        junction(root / ".plans", outside)
        result = lesson_cleanup.delete(request(ref("plan", PLAN_NAME)))
        assert result["deleted"] == [] and precious.exists()


class TestLastLineOfDefence:
    """_remove is only reached for scanned items, but it checks the path again before acting."""

    def test_a_link_is_refused_even_if_it_somehow_reached_the_remover(self, root, tmp_path):
        outside = tmp_path / "outside"
        precious = write(outside / "precious.txt", 10)
        junction(root / "X.partial", outside)
        with pytest.raises(lesson_cleanup._Refused):
            lesson_cleanup._remove(root, "partial", "X.partial")
        assert precious.exists()

    def test_a_path_that_leaves_the_folder_is_refused(self, root, tmp_path):
        outside = write(tmp_path / "outside" / "a.txt", 10)
        with pytest.raises(lesson_cleanup._Refused):
            lesson_cleanup._remove(root, "lesson", "../outside")
        assert outside.exists()


class TestDeleteWhileBuilding:
    def test_partials_and_plans_are_refused_while_a_build_runs(self, root, alive):
        alive.add(job(root, "running", 0))
        partial = write(root / f"{LESSON_ID}.partial" / "a", days=9).parent
        plan = write(root / ".plans" / PLAN_NAME, days=9)
        for item in (ref("partial", partial.name), ref("plan", PLAN_NAME)):
            with pytest.raises(lesson_cleanup.InUse) as caught:
                lesson_cleanup.delete(request(item))
            assert caught.value.http_status == 409 and "wait until it finishes" in str(caught.value)
        assert partial.exists() and plan.exists()

    def test_a_refused_request_deletes_nothing_at_all(self, root, alive):
        alive.add(job(root, "running", 0))
        old = job(root, "failed", 3)
        partial = write(root / f"{LESSON_ID}.partial" / "a", days=9).parent
        with pytest.raises(lesson_cleanup.InUse):
            lesson_cleanup.delete(request(ref("job", old), ref("partial", partial.name)))
        assert (root / ".jobs" / f"{old}.json").exists()

    def test_the_running_builds_own_record_is_protected(self, root, alive):
        running = job(root, "running", 0)
        alive.add(running)
        result = lesson_cleanup.delete(request(ref("job", running)))
        assert result["deleted"] == [] and result["skipped"][0]["reason"] == "This build is running right now"
        assert (root / ".jobs" / f"{running}.json").exists()

    def test_old_records_and_lessons_can_go_while_a_build_runs(self, root, alive):
        alive.add(job(root, "running", 0))
        old = job(root, "failed", 3)
        lesson(root)
        result = lesson_cleanup.delete(request(ref("job", old), ref("lesson", LESSON_ID)))
        assert len(result["deleted"]) == 2 and result["skipped"] == []

    def test_deleting_holds_the_lock_a_build_is_started_under(self, root, alive, monkeypatch):
        events = []

        class Recording:
            def __enter__(self):
                events.append("locked")

            def __exit__(self, *exc):
                events.append("unlocked")

        monkeypatch.setattr(lessons, "_jobs_lock", Recording())
        monkeypatch.setattr(lesson_cleanup, "scan", lambda: events.append("scan") or {"running_build": False, "items": []})
        lesson_cleanup.delete(request())
        assert events == ["locked", "scan", "unlocked"]

    def test_a_build_cannot_start_in_the_middle_of_a_delete(self, root, alive, monkeypatch):
        write(root / ".plans" / PLAN_NAME, days=9)
        order, builds = [], []
        original = lesson_cleanup._remove

        def start_like_a_build():
            with lessons._jobs_lock:
                order.append("build started")

        def slow(*args):
            builds.append(threading.Thread(target=start_like_a_build))
            builds[0].start()
            time.sleep(0.2)                # long enough for the build to be waiting on the lock
            order.append("removing")
            return original(*args)

        monkeypatch.setattr(lesson_cleanup, "_remove", slow)
        lesson_cleanup.delete(request(ref("plan", PLAN_NAME)))
        builds[0].join(2)
        assert order == ["removing", "build started"]


class TestPartialFailures:
    @pytest.fixture
    def locked(self, monkeypatch):
        """Makes deleting any file called locked.bin fail the way an open file does on Windows."""
        real = os.unlink

        def unlink(path, *args, **kwargs):
            if Path(path).name == "locked.bin":
                raise PermissionError(13, "The process cannot access the file because it is being used by another process",
                                      str(path), 32)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", unlink)

    def test_a_locked_file_is_skipped_with_a_reason_and_the_rest_still_goes(self, root, alive, locked):
        stuck = lesson(root, "CHEM_1450_20260916-20260916_20260916105209")
        write(stuck / "locked.bin", 700)
        plan = write(root / ".plans" / PLAN_NAME, days=9)
        result = lesson_cleanup.delete(request(ref("lesson", stuck.name), ref("plan", PLAN_NAME)))
        assert result["deleted"] == [ref("plan", PLAN_NAME)] and not plan.exists()
        [skipped] = result["skipped"]
        assert skipped["item"] == ref("lesson", stuck.name)
        assert "open in another program" in skipped["reason"] and "couldn't remove all of it" in skipped["reason"].lower()
        assert str(stuck) not in skipped["reason"]
        assert (stuck / "locked.bin").exists() and not (stuck / "slide-01.ogg").exists()   # the rest of it did go
        assert (stuck / "lesson.json").exists()        # kept: the lesson is still listed, so it can be tried again
        assert ("lesson", stuck.name) in by_key(lesson_cleanup.scan())
        assert result["freed_bytes"] > 300

    def test_a_failure_in_the_first_item_does_not_stop_the_others(self, root, alive, locked):
        first = write(root / f"{LESSON_ID}.partial" / "locked.bin", 10).parent
        second = write(root / "PSYC_1300_20260101-20260101_20260101000001.partial" / "a", 10).parent
        third = write(root / ".plans" / PLAN_NAME, days=9)
        result = lesson_cleanup.delete(request(ref("partial", first.name), ref("partial", second.name), ref("plan", PLAN_NAME)))
        assert [d["id"] for d in result["deleted"]] == [second.name, PLAN_NAME]
        assert len(result["skipped"]) == 1 and first.exists() and not second.exists() and not third.exists()

    def test_a_locked_lesson_keeps_its_quiz_answers(self, root, alive, locked):
        folder = lesson(root)
        write(folder / "locked.bin", 5)
        write(root / ".quiz" / f"{LESSON_ID}.json", 50)
        lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))
        assert (root / ".quiz" / f"{LESSON_ID}.json").exists()

    def test_a_stuck_lesson_can_be_deleted_again_once_its_file_is_free(self, root, alive, locked, monkeypatch):
        stuck = lesson(root)
        write(stuck / "locked.bin", 5)
        lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))
        monkeypatch.setattr(os, "unlink", REAL_UNLINK)         # the other program has let go of it
        result = lesson_cleanup.delete(request(ref("lesson", LESSON_ID)))
        assert result["deleted"] == [ref("lesson", LESSON_ID)] and result["skipped"] == [] and not stuck.exists()

    def test_a_link_inside_a_lesson_is_a_stuck_item_not_a_silent_skip(self, root, alive, tmp_path):
        outside = write(tmp_path / "outside" / "precious.txt", 10)
        folder = lesson(root)
        junction(folder / "linked", tmp_path / "outside")
        _, reason = lesson_cleanup._remove_lesson(folder)
        assert "link" in reason and (folder / "lesson.json").exists() and outside.exists()

    def test_a_locked_job_log_leaves_the_record_whole(self, root, alive, monkeypatch):
        job_id = job(root, "failed", 3)
        real = Path.unlink

        def unlink(self, *args, **kwargs):
            if self.suffix == ".log":
                raise PermissionError(13, "in use", str(self), 32)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink)
        result = lesson_cleanup.delete(request(ref("job", job_id)))
        assert result["deleted"] == [] and "open in another program" in result["skipped"][0]["reason"]
        assert (root / ".jobs" / f"{job_id}.json").exists() and (root / ".jobs" / f"{job_id}.log").exists()

    def test_a_read_only_file_is_deleted_anyway(self, root, alive):
        partial = write(root / f"{LESSON_ID}.partial" / "readonly.bin", 10).parent
        os.chmod(partial / "readonly.bin", 0o444)
        result = lesson_cleanup.delete(request(ref("partial", partial.name)))
        assert result["deleted"] == [ref("partial", partial.name)] and not partial.exists()

    def test_an_unexpected_error_on_one_item_is_a_skip_not_a_crash(self, root, alive, monkeypatch):
        write(root / ".plans" / PLAN_NAME, days=9)
        write(root / ".plans" / "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json", days=9)
        real = lesson_cleanup._remove_file
        calls = []

        def flaky(path):
            calls.append(path.name)
            if len(calls) == 1:
                raise OSError(5, "Access is denied")
            return real(path)

        monkeypatch.setattr(lesson_cleanup, "_remove_file", flaky)
        result = lesson_cleanup.delete(request(ref("plan", "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json"), ref("plan", PLAN_NAME)))
        assert len(result["deleted"]) == 1 and len(result["skipped"]) == 1


class Vanished:
    """A listed entry that is gone by the time it is looked at: its stat() fails the way it would."""

    def __init__(self, entry):
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry, name)

    def stat(self, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", self._entry.path)


class TestEntriesThatVanish:
    """Another tab's delete, or a build pruning its old records, can remove something mid-scan. That is
    a normal event, not an error: the scan has to finish, and the page must not read it as an old backend."""

    @pytest.fixture
    def vanish(self, monkeypatch):
        listed = lesson_cleanup._entries

        def make_vanish(*names):
            monkeypatch.setattr(lesson_cleanup, "_entries",
                                lambda directory: [Vanished(entry) if entry.name in names else entry
                                                   for entry in listed(directory)])
        return make_vanish

    def test_a_build_record_removed_mid_scan_is_left_out(self, root, alive, monkeypatch):
        gone = job(root, "failed", 3)
        survivor = job(root, "failed", 2)
        real_is_file = Path.is_file

        def vanish_after_the_check(path):
            if path.name == f"{gone}.json" and real_is_file(path):
                path.unlink()                  # deleted between the check and the stat
                return True
            return real_is_file(path)

        monkeypatch.setattr(Path, "is_file", vanish_after_the_check)
        items = by_key(lesson_cleanup.scan())
        assert ("job", gone) not in items and ("job", survivor) in items

    @pytest.mark.parametrize("kind", ["partial", "plan", "lesson"])
    def test_an_entry_removed_mid_scan_is_left_out_and_the_rest_are_listed(self, root, alive, vanish, kind):
        if kind == "partial":
            gone, other = f"{LESSON_ID}.partial", "PSYC_1300_20260101-20260101_20260101000001.partial"
            write(root / gone / "a")
            write(root / other / "a")
        elif kind == "plan":
            gone, other = PLAN_NAME, "PSYC_1300_20260930-20261005_aaaaaaaaaaaa.json"
            write(root / ".plans" / gone)
            write(root / ".plans" / other)
        else:
            gone, other = LESSON_ID, "CHEM_1450_20260916-20260916_20260916105209"
            lesson(root)
            lesson(root, other)
        vanish(gone)
        items = by_key(lesson_cleanup.scan())
        assert (kind, gone) not in items and (kind, other) in items

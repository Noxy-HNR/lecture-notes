"""One `## <date>` header per day in the notes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storage import dedupe_day_headings

DAY = "## Monday, September 14, 2026"


def test_repeated_day_header_is_dropped_and_both_sessions_are_kept():
    text = (
        "# Class (CODE)\n\n"
        "<!-- session:A -->\n" + DAY + "\n\n### One\n- a\n<!-- /session:A -->\n\n"
        "<!-- session:B -->\n" + DAY + "\n\n### Two\n- b\n<!-- /session:B -->\n"
    )
    out = dedupe_day_headings(text)
    assert out.count(DAY) == 1
    assert "### One" in out and "### Two" in out
    assert "<!-- session:B -->" in out and "<!-- /session:B -->" in out
    assert "\n\n\n" not in out


def test_day_that_comes_back_later_keeps_its_header():
    text = (
        "## Monday, September 14, 2026\nx\n"
        "## Wednesday, September 16, 2026\ny\n"
        "## Monday, September 14, 2026\nz\n"
    )
    assert dedupe_day_headings(text) == text


def test_dedupe_is_idempotent():
    text = "## Monday, September 14, 2026\n\n### A\n\n## Monday, September 14, 2026\n\n### B\n"
    once = dedupe_day_headings(text)
    assert dedupe_day_headings(once) == once
    assert once.count(DAY) == 1


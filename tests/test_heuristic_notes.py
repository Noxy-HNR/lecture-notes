"""The no-model notes formatter (local_formatter): readable notes from a raw transcript when
neither Claude nor the local model is available."""
import re
import time

import pytest

import local_formatter as lf


@pytest.fixture
def class_notes(tmp_path, monkeypatch):
    """A notes folder of this test's own: CHEM's earlier Claude notes bold a few terms."""
    monkeypatch.setattr(lf, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(lf.vocab_module, "class_only_terms", lambda code: [])
    monkeypatch.setattr(lf, "correct_with_glossary", lambda text, code: text)
    (tmp_path / "CHEM_1450.md").write_text(
        "# Chemistry\n\n## Monday, November 9, 2026\n\n- **Partial pressure** adds up. **Mole fraction** too. "
        "**Vapor pressure** of water. **Stoichiometry** and **energy**.\n\n"
        "## Wednesday, November 11, 2026\n\n*(auto-formatted locally — Claude CLI/API unavailable)*\n\n"
        "- **things** and **through** were bolded by word counts here.\n", encoding="utf-8")
    (tmp_path / "BIOL_1440.md").write_text("## Day\n\n- **Energy** flows. **Example** here.\n", encoding="utf-8")
    (tmp_path / "PSYC_1300.md").write_text("## Day\n\n- **Energy** again. **Example** too.\n", encoding="utf-8")
    return tmp_path


# --- cleaning ------------------------------------------------------------------------------------

def test_filler_sentences_are_dropped_and_leading_filler_stripped():
    cleaned = lf.clean_sentences(["Right?", "So, this is it.", "Okay, so, the pressure doubles here.",
                                  "Um, you know, gases mix completely.", "Any questions?", "Yep."])
    assert cleaned == ["The pressure doubles here.", "Gases mix completely."]


def test_a_sentence_cut_by_a_stray_period_is_joined():
    cleaned = lf.clean_sentences(["We collect gases over water as our application in.", "The laboratory."])
    assert cleaned == ["We collect gases over water as our application in the laboratory."]


def test_short_sentences_with_a_key_term_are_kept():
    assert lf.clean_sentences(["But partial pressure."], terms=["partial pressure"]) == ["But partial pressure."]


def test_repeated_words_and_spoken_math_are_tidied():
    assert lf._clean_sentence("the the ideal gas law is P equals nRT") == "The ideal gas law is P = nRT"
    assert lf._clean_sentence("add 2 plus 3 to get it") == "Add 2 + 3 to get it"


# --- key terms ---------------------------------------------------------------------------------

def test_class_terms_come_from_claudes_bold_not_heuristic_sections(class_notes):
    terms = lf.class_key_terms("CHEM 1450")
    assert {"partial pressure", "mole fraction", "vapor pressure", "stoichiometry"} <= terms
    assert "things" not in terms and "through" not in terms, "heuristic sections' bold is ignored"
    assert "energy" not in terms, "a single word bolded in other classes too isn't distinctive"


def test_repeated_phrases_become_terms():
    sentences = ["The mole fraction is small.", "Find the mole fraction first.", "Mole fraction times total.",
                 "Practice practice practice.", "Practice practice again.", "Practice practice more."]
    assert lf._phrases(sentences) == {"mole fraction"}


def test_each_term_is_bolded_once_longest_first():
    bolded = set()
    first = lf._bold_first_mentions("Partial pressure and pressure.", ["partial pressure", "pressure"], bolded)
    assert first == "**Partial pressure** and **pressure**."
    again = lf._bold_first_mentions("More partial pressure.", ["partial pressure", "pressure"], bolded)
    assert again == "More partial pressure."


# --- structure ---------------------------------------------------------------------------------

def test_sentences_are_grouped_into_short_bullets():
    sentences = ["Gases mix in a tank.", "The tank holds oxygen gas.", "Oxygen tanks help patients.",
                 "Volcanoes erupt magma.", "Magma cools into rock."]
    bullets = lf._bulletize(sentences)
    assert len(bullets) < len(sentences)
    assert all(len(b.split()) <= lf.MAX_BULLET_WORDS * 1.4 for b in bullets)
    assert " ".join(bullets) == " ".join(sentences), "nothing lost, nothing reordered"


def test_a_question_stays_with_its_answer():
    bullets = lf._bulletize(["Why does the gas bubble through water?", "Because water vapor adds pressure."])
    assert bullets == ["Why does the gas bubble through water? Because water vapor adds pressure."]


def topic(word, other, n):
    return [f"The {word} changes the {other} in this {word} system number {i}." for i in range(n)]


def test_sections_break_where_the_vocabulary_changes(monkeypatch):
    monkeypatch.setattr(lf, "WORDS_PER_SECTION", 120)
    sentences = topic("enzyme", "substrate", 14) + topic("glacier", "valley", 14)
    sections = lf._segment(sentences)
    assert len(sections) == 2 and sections[0] == sentences[:14]


def test_short_input_is_one_section():
    sentences = topic("enzyme", "substrate", 5)
    assert lf._segment(sentences) == [sentences]


def test_definitions_and_announcements_go_to_the_top():
    sentences = ["Partial pressure is the pressure one gas would exert alone.",
                 "That's why partial pressure is important.",
                 "The exam is next Tuesday, so study the slides.",
                 "A lot of tanks hold mixtures."]
    lines, moved = lf._front_matter(sentences, ["partial pressure"])
    text = "\n".join(lines)
    assert "**Announcements & reminders**" in text and "- The exam is next Tuesday" in text
    assert "- **Partial pressure** — Partial pressure is the pressure one gas would exert alone." in text
    assert "important" not in text, "a remark about a term is not its definition"
    assert moved == {"The exam is next Tuesday, so study the slides."}


def test_a_definition_is_labelled_with_the_whole_phrase():
    found = lf._definitions(["Because partial pressure means more than one gas is present."], ["pressure"])
    assert found == [("partial pressure", "Because partial pressure means more than one gas is present.")]


# --- the whole thing -------------------------------------------------------------------------------

LECTURE = (
    "Okay. So, today we wrap up gas laws. The exam is on Tuesday after Thanksgiving. Right? "
    "Partial pressure is the pressure one gas would exert alone in the container. "
    "The partial pressures of all the gases add up to the total pressure. "
    "You add the partial pressure of oxygen and the partial pressure of nitrogen. "
    "Scuba tanks hold a mixture, so partial pressure matters for divers. Okay? "
    "Now the mole fraction. The mole fraction is moles of one gas over total moles. "
    "Multiply the mole fraction by the total pressure to get the partial pressure. "
    "Um, so, the mole fraction has no units. Any questions? "
)


def test_whole_lecture_notes(class_notes):
    notes = lf.format_transcript("General Chemistry", "Monday, November 16, 2026", LECTURE * 3, "CHEM 1450",
                                 reason="Claude's usage limit was reached")
    lines = notes.splitlines()
    assert lines[0] == "## Monday, November 16, 2026"
    assert lines[2] == "*(auto-formatted locally — Claude's usage limit was reached)*"
    assert not [l for l in lines[1:] if l.startswith("## ") or l.startswith("# ")], "no extra top-level heading"
    assert "**Announcements & reminders**" in notes and "exam is on Tuesday" in notes
    assert "- **Partial pressure** — Partial pressure is the pressure one gas would exert alone" in notes
    assert "Right?" not in notes and "Any questions?" not in notes and "Um" not in notes
    blank_line = "\n\n"
    below_key_terms = blank_line.join(notes.split("**Key terms**")[1].split(blank_line)[2:])
    assert below_key_terms.lower().count("**mole fraction**") == 1, "bolded once, at its first mention"


def test_no_speech_and_heuristic_only_mode():
    assert "no speech detected" in lf.format_transcript("T", "Day", "Okay. Right? Um.", None)
    assert "*(auto-formatted locally — Claude CLI/API unavailable)*" in lf.format_transcript(
        "T", "Day", "Enzymes lower the activation energy of reactions in cells.", None)


def test_speaker_tagged_transcripts_keep_their_q_and_a(class_notes):
    transcript = ("[Speaker 1] Partial pressure is the pressure of one gas alone. Okay? The total adds up.\n"
                  "[Speaker 2] Does water vapor count as a gas?\n"
                  "[Speaker 1] Yes, water vapor has a partial pressure too.\n"
                  "[Speaker 1] So you subtract it from the total pressure when collecting gas over water.\n"
                  "[Speaker 1] Then you get the dry gas pressure.")
    notes = lf.format_transcript("Chemistry", "Day", transcript, "CHEM 1450")
    assert "### Q&A" in notes and "**Student 1:** Does water vapor count as a gas?" in notes
    assert "Okay?" not in notes


def test_a_full_lecture_formats_in_well_under_a_few_seconds(class_notes):
    started = time.perf_counter()
    lf.format_transcript("General Chemistry", "Day", LECTURE * 60, "CHEM 1450")
    assert time.perf_counter() - started < 5

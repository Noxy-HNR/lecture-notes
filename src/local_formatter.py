"""Heuristic, no-LLM transcript -> structured markdown notes.

Used when neither the Claude CLI nor the API is available. Can't summarize or
"understand" the lecture the way Claude does, but gets closer to a real notes
format than a wall of paragraphs by:
  - splitting the transcript into sentence-level bullets
  - detecting spoken topic-transition cues ("we're going to talk about X",
    "next let's discuss Y", ...) and turning them into subheadings
  - bolding terms that recur often enough to likely be key vocabulary
  - fuzzy-correcting words against the class's vocab.json glossary (helps with
    Latin/technical terms Whisper mangled)
  - if given a diarized, speaker-tagged transcript ("[Speaker 1] ...", one
    speaker per line), detecting back-and-forth as Q&A and setting it apart
    from continuous lecture content
"""
import difflib
import re
from collections import Counter

import vocab as vocab_module

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "has", "are", "was",
    "were", "will", "would", "could", "should", "about", "there", "their",
    "they", "them", "then", "than", "into", "these", "those", "here", "what",
    "when", "where", "which", "while", "your", "just", "like", "going", "gonna",
    "kind", "sort", "really", "actually", "basically", "okay", "right", "yeah",
    "know", "think", "want", "some", "such", "each", "also", "very", "more",
    "most", "much", "many", "over", "into", "onto", "because", "again", "back",
}

_TRANSITION_PATTERNS = [
    r"we(?:'re| are) going to (?:talk about|discuss|cover|look at|go over)",
    r"we(?:'ll| will) (?:now )?(?:talk about|discuss|cover|look at|go over)",
    r"let'?s (?:talk about|discuss|move on to|look at|go over)",
    r"now (?:we'?re going to|let'?s|we'?ll)",
    r"we(?:'ll| will) now (?:talk about|discuss|cover|look at|go over)",
    r"next(?:,)? (?:we'?re going to|we'?ll|let'?s|up (?:is|we))",
    r"moving on(?:,)?\s*(?:to)?",
    r"today we(?:'re| are) (?:going to )?(?:talk(?:ing)? about|discuss(?:ing)?|cover(?:ing)?)",
]
_TRANSITION_RE = re.compile(r"\b(?:" + "|".join(_TRANSITION_PATTERNS) + r")\b\s*(.*)", re.IGNORECASE)
_STOP_AT = re.compile(r"\b(?:and then|and also|but|so that|which|because|that)\b", re.IGNORECASE)
_SPEAKER_LINE_RE = re.compile(r"^\[(Speaker \d+)\]\s*(.*)$")


# ---------------------------------------------------------------------------
# Glossary correction
# ---------------------------------------------------------------------------

def _correct_with_glossary(text: str, class_code: str | None) -> str:
    """Fuzzy-matches individual words against the class's vocab.json glossary
    and fixes near-misses (e.g. Whisper hearing "amigdala" for "amygdala")."""
    if not class_code:
        return text
    terms = vocab_module.terms_for_class(class_code)
    single_word_terms = [t for t in terms if " " not in t]
    if not single_word_terms:
        return text

    lower_lookup = {t.lower(): t for t in single_word_terms}

    def repl(match):
        word = match.group(0)
        lw = word.lower()
        if lw in lower_lookup:
            return word  # already correct
        close = difflib.get_close_matches(lw, lower_lookup.keys(), n=1, cutoff=0.82)
        if close:
            corrected = lower_lookup[close[0]]
            return corrected.capitalize() if word[0].isupper() else corrected
        return word

    return re.sub(r"[A-Za-z][A-Za-z\-]{3,}", repl, text)


# ---------------------------------------------------------------------------
# Speaker-tagged input parsing
# ---------------------------------------------------------------------------

def _parse_speaker_entries(transcript: str):
    """If `transcript` is diarized ("[Speaker 1] ..." lines), returns a flat list
    of (speaker_label, sentence) pairs. Returns None if no speaker tags are present
    (plain transcript - caller should fall back to the unlabeled path)."""
    lines = transcript.split("\n")
    matches = [(_SPEAKER_LINE_RE.match(line), line) for line in lines]
    if not any(m for m, _ in matches):
        return None

    entries = []
    for m, line in matches:
        if m:
            speaker, text = m.group(1), m.group(2)
        else:
            speaker, text = None, line
        for sentence in _split_sentences(text):
            entries.append((speaker, sentence))
    return entries


def _label_by_role(entries):
    """Relabels raw "Speaker N" tags as Instructor / Student 1 / Student 2 / ...
    based on who talks the most (assumed to be the instructor)."""
    counts = Counter(sp for sp, _ in entries if sp)
    if not counts:
        return [("Instructor", s) for _, s in entries]

    primary = counts.most_common(1)[0][0]
    student_labels = {}
    next_student = 1
    labeled = []
    for speaker, sentence in entries:
        if speaker is None or speaker == primary:
            labeled.append(("Instructor", sentence))
        else:
            if speaker not in student_labels:
                student_labels[speaker] = f"Student {next_student}"
                next_student += 1
            labeled.append((student_labels[speaker], sentence))
    return labeled


def _segment_qa(labeled_entries):
    """Splits into ("lecture", [...]) / ("qa", [...]) blocks. A block becomes "qa"
    as soon as a non-instructor speaks, and stays "qa" through the instructor's
    reply until either the instructor has spoken twice in a row uninterrupted, or
    the instructor's reply itself opens a new topic (e.g. "moving on, let's discuss
    X") - that's a lecture resuming, not part of the answer, so it closes the QA
    block immediately and starts fresh instead of being swallowed into it."""
    blocks = []
    i, n = 0, len(labeled_entries)
    while i < n:
        label, sentence = labeled_entries[i]
        if label != "Instructor":
            items = [(label, sentence)]
            i += 1
            consecutive_instructor = 0
            while i < n:
                label2, sentence2 = labeled_entries[i]
                if label2 == "Instructor" and _extract_heading(sentence2) is not None:
                    break  # new topic starting - leave it for the next lecture block
                items.append((label2, sentence2))
                i += 1
                consecutive_instructor = consecutive_instructor + 1 if label2 == "Instructor" else 0
                if consecutive_instructor >= 2:
                    break
            blocks.append(("qa", items))
        else:
            items = [(label, sentence)]
            i += 1
            while i < n and labeled_entries[i][0] == "Instructor":
                items.append(labeled_entries[i])
                i += 1
            blocks.append(("lecture", items))
    return blocks


# ---------------------------------------------------------------------------
# Sentence splitting / topic detection / key terms (unlabeled lecture content)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_heading(sentence: str) -> str | None:
    m = _TRANSITION_RE.search(sentence)
    if not m:
        return None
    tail = m.group(1).strip(" .,:;")
    if not tail:
        return None
    cut = _STOP_AT.search(tail)
    if cut:
        tail = tail[:cut.start()].strip(" .,:;")
    words = tail.split()
    if not words:
        return None
    heading = " ".join(words[:8])
    return heading[:1].upper() + heading[1:]


def _group_into_topics(sentences: list[str]):
    """Returns (groups, all_headings_mentioned). See module docstring for rationale."""
    groups = []
    all_headings = []
    current_heading = None
    current_sentences = []

    for sentence in sentences:
        heading = _extract_heading(sentence)
        if heading:
            all_headings.append(heading)
            if current_sentences:
                groups.append((current_heading, current_sentences))
                current_sentences = []
            current_heading = heading
            continue
        current_sentences.append(sentence)

    if current_sentences:
        groups.append((current_heading, current_sentences))
    return groups, all_headings


def _key_terms(sentences: list[str], max_terms: int = 10) -> set[str]:
    counts = {}
    for sentence in sentences:
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", sentence):
            lw = word.lower()
            if lw in _STOPWORDS:
                continue
            counts[lw] = counts.get(lw, 0) + 1
    ranked = sorted((w for w, c in counts.items() if c >= 2), key=lambda w: -counts[w])
    return set(ranked[:max_terms])


def _bold_terms(sentence: str, terms: set[str]) -> str:
    if not terms:
        return sentence

    def repl(match):
        word = match.group(0)
        return f"**{word}**" if word.lower() in terms else word

    return re.sub(r"[A-Za-z][A-Za-z\-]{4,}", repl, sentence, count=0)


def _render_lecture_sentences(sentences: list[str]) -> list[str]:
    """Returns markdown lines (headings/bullets) for a block of pure lecture content."""
    if not sentences:
        return []
    terms = _key_terms(sentences)
    groups, all_headings = _group_into_topics(sentences)
    lines = []
    if len(all_headings) >= 2:
        lines.append("**Topics covered:** " + "; ".join(all_headings))
        lines.append("")
    for heading, group_sentences in groups:
        if heading:
            lines.append(f"### {heading}")
        for sentence in group_sentences:
            lines.append(f"- {_bold_terms(sentence, terms)}")
        lines.append("")
    return lines


def _render_qa_block(items) -> list[str]:
    lines = ["### Q&A", ""]
    for label, sentence in items:
        lines.append(f"- **{label}:** {sentence}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def format_transcript(class_title: str, session_date: str, transcript: str,
                       class_code: str | None = None) -> str:
    transcript = _correct_with_glossary(transcript, class_code)
    speaker_entries = _parse_speaker_entries(transcript)

    header = [f"## {session_date}", "", "*(auto-formatted locally — Claude CLI/API unavailable)*", ""]

    if speaker_entries is None:
        # No diarization data - plain lecture transcript, original behavior.
        sentences = _split_sentences(transcript)
        if not sentences:
            return f"## {session_date}\n\n*(auto-formatted locally — no speech detected)*\n"
        body = _render_lecture_sentences(sentences)
        return "\n".join(header + body).rstrip() + "\n"

    labeled = _label_by_role(speaker_entries)
    blocks = _segment_qa(labeled)

    body = []
    for block_type, items in blocks:
        if block_type == "qa":
            body.extend(_render_qa_block(items))
        else:
            body.extend(_render_lecture_sentences([s for _, s in items]))

    if not body:
        return f"## {session_date}\n\n*(auto-formatted locally — no speech detected)*\n"

    return "\n".join(header + body).rstrip() + "\n"

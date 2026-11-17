"""Heuristic, no-LLM transcript -> structured markdown notes.

Used when neither the Claude CLI nor the API is available. Can't summarize or
"understand" the lecture the way Claude does, but gets closer to a real notes
format than a wall of paragraphs by:
  - splitting the transcript into sentence-level bullets
  - detecting spoken topic-transition cues ("we're going to talk about X",
    "next let's discuss Y", ...) and turning them into subheadings
  - bolding terms that recur often enough to likely be key vocabulary
  - fuzzy-correcting words against the class's vocab.json glossary (helps with
    Latin/technical terms transcription mangled)
  - if given a diarized, speaker-tagged transcript ("[Speaker 1] ...", one
    speaker per line), detecting back-and-forth as Q&A and setting it apart
    from continuous lecture content
"""
import difflib
import re
from collections import Counter
from pathlib import Path

import spelling
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

def correct_with_glossary(text: str, class_code: str | None) -> str:
    """Fixes near-miss spellings of the class's vocab.json glossary terms (e.g. "amigdala"
    transcribed for "amygdala").

    A word only changes when Windows' spell checker says it is misspelled, the term starts with
    the same letter, and the word isn't just the term with a prefix or suffix. Real words are
    never swapped for a look-alike term: "action" is not "cation", "dehydration" is not
    "hydration", and "molecules" stays plural. Without a spell checker nothing changes."""
    if not class_code:
        return text
    terms = vocab_module.terms_for_class(class_code)
    single_word_terms = [t for t in terms if " " not in t]
    if not single_word_terms:
        return text

    lower_lookup = {t.lower(): t for t in single_word_terms}
    decided = {}  # one decision per distinct word

    def correction_for(word):
        lw = word.lower()
        if lw in lower_lookup:
            return None  # already correct
        close = difflib.get_close_matches(lw, lower_lookup.keys(), n=1, cutoff=0.82)
        if not close:
            return None
        term_lower = close[0]
        if term_lower[0] != lw[0] or term_lower in lw or lw in term_lower:
            return None
        if spelling.is_english_word(word) is not False:
            return None  # a real word, or no way to tell
        term = lower_lookup[term_lower]
        return term[0].upper() + term[1:] if word[0].isupper() else term

    def repl(match):
        word = match.group(0)
        if word not in decided:
            decided[word] = correction_for(word)
        return decided[word] or word

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
    # Guards against a real, observed bug: naive sentence-splitting on "." treats an
    # abbreviation/initial (e.g. a mis-transcribed name like "A.") as its own sentence,
    # so a transition cue right before one ("...we'll talk about A.") extracts just that
    # single letter as the "heading" - a single short word is essentially never a real
    # topic heading, so require either 2+ words or one reasonably long word.
    if len(words) < 2 and len(words[0]) < 4:
        return None
    heading = " ".join(words[:8])
    return heading[:1].upper() + heading[1:]


def _group_into_topics(sentences: list[str]):
    """Splits at spoken transitions ("let's talk about X"). Kept for reference and tests; the
    notes now find topic breaks from the vocabulary itself (_segment)."""
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


# ---------------------------------------------------------------------------
# Cleaning: filler, disfluencies, broken sentences, spoken math
# ---------------------------------------------------------------------------

# Words that say nothing about the topic. Used for content-word counts, topic breaks and names.
_FUNCTION_WORDS = _STOPWORDS | {
    "a", "an", "as", "at", "be", "been", "being", "but", "by", "can", "did", "do", "does", "doing",
    "done", "for", "get", "gets", "getting", "got", "had", "he", "her", "him", "his", "how", "i",
    "if", "in", "is", "it", "its", "it's", "let", "let's", "me", "my", "no", "not", "of", "on",
    "one", "or", "our", "out", "she", "so", "to", "too", "two", "up", "us", "we", "who", "why",
    "you", "all", "any", "anyone", "anything", "everybody", "everyone", "everything", "guys",
    "something", "someone", "thing", "things", "stuff", "way", "lot", "lots", "bit", "little",
    "say", "said", "says", "see", "seen", "look", "looking", "make", "makes", "made", "need",
    "needs", "use", "used", "using", "go", "goes", "went", "come", "comes", "take", "takes", "put",
    "give", "gives", "tell", "talk", "talking", "mean", "means", "meaning", "call", "called",
    "first", "second", "next", "last", "now", "today", "well", "still", "even", "only", "same",
    "other", "another", "different", "good", "great", "bad", "big", "new", "sure", "yes", "yep",
    "nope", "um", "uh", "hmm", "alright", "ok", "can't", "don't", "doesn't", "didn't", "isn't",
    "aren't", "won't", "wouldn't", "i'm", "you're", "we're", "they're", "that's", "there's",
    "what's", "here's", "i'll", "we'll", "you'll", "i've", "we've", "you've", "gotta", "wanna",
    "maybe", "probably", "always", "never", "every", "both", "few", "less", "least", "lots",
    "able", "try", "trying", "work", "works", "time", "times", "question", "questions", "answer",
    "example", "problem", "problems", "point", "part", "number", "three", "four", "five", "ten",
    "remember", "understand", "whatever", "anyway", "already", "enough", "around", "through",
    "down", "off", "after", "before", "between", "under", "same", "own", "where's", "whole",
    # Generic classroom and everyday words: never what a section is about, never worth bold.
    "definition", "bottom", "top", "side", "left", "equation", "concept", "idea", "topic", "slide",
    "slides", "textbook", "chapter", "page", "class", "lecture", "exam", "test", "quiz", "homework",
    "practice", "step", "steps", "unit", "value", "values", "data", "result", "results", "reason",
    "case", "cases", "type", "types", "form", "level", "amount", "process", "start", "end", "rest",
    "reading", "notes", "scenario", "setup", "people", "person", "world", "day", "week", "year",
    "minute", "minutes", "hour", "hours", "words", "word", "picture", "video", "book", "list",
    "board", "term", "terms", "fact", "sense", "kinds", "important", "real", "true", "false",
    "basic", "simple", "easy", "hard", "high", "low", "long", "short", "whatnot", "might", "must",
    "reliable", "doe", "john", "jane", "mr", "mrs", "dr", "somewhere", "somebody", "anywhere",
    "everywhere", "nowhere", "average", "human", "fall", "falls", "above", "below",
}

_LEADING_FILLER = re.compile(
    r"^(?:(?:so|okay|ok|alright|all right|um+|uh+|well|yeah|yep|right|now|and so|anyway|"
    r"you know|i mean)\s*,?\s+)+", re.IGNORECASE)
_DISFLUENCY = re.compile(r"(?:,\s*)?\b(?:um+|uh+|erm|you know)\b,?\s*", re.IGNORECASE)
_REPEATED_WORD = re.compile(r"\b(the|a|an|i|to|and|of|is|it|we|you|that|this|in|so)(?:\s+\1\b)+", re.IGNORECASE)
_PURE_FILLER = re.compile(
    r"^(?:right|okay|ok|yes|yeah|yep|no|alright|all right|good|great|cool|perfect|awesome|"
    r"any questions|questions|does that make sense|make sense|got it|you got it|okay so|"
    r"and so on|et cetera|thank you|thanks|sorry|so yeah|there we go|here we go|that's it)\W*$",
    re.IGNORECASE)
# A period that ended a sentence on one of these words almost always cut a sentence in two
# ("...as our application in. The laboratory.").
_JOIN_AFTER = {"in", "on", "at", "of", "to", "the", "a", "an", "and", "or", "for", "with", "from",
               "by", "as", "into", "our", "your", "their", "its", "is", "are", "was", "were", "this",
               "that", "these", "those", "my", "his", "her", "about", "than", "but", "because"}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z'\-]+", text.lower())
            if len(w) >= 3 and w not in _FUNCTION_WORDS]


def _stem(word: str) -> str:
    w = word.lower().removesuffix("'s")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _spoken_math(text: str) -> str:
    text = re.sub(r"\b([A-Za-z]{1,4}|\d+(?:\.\d+)?)\s+equals\s+", r"\1 = ", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+plus\s+(\d)", r"\1 + \2", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+minus\s+(\d)", r"\1 − \2", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s+times\s+(\d)", r"\1 × \2", text)
    return text


def _clean_sentence(sentence: str) -> str:
    s = _DISFLUENCY.sub(" ", sentence)
    s = _REPEATED_WORD.sub(r"\1", s)
    s = _LEADING_FILLER.sub("", s.strip())
    s = re.sub(r"\s+([,.?!;:])", r"\1", re.sub(r"\s{2,}", " ", s)).strip(" ,")
    s = _spoken_math(s)
    return s[:1].upper() + s[1:] if s else s


def _join_fragments(sentences: list[str]) -> list[str]:
    joined = []
    for sentence in sentences:
        if joined:
            last_word = re.findall(r"[A-Za-z']+", joined[-1][-30:])
            if (joined[-1].endswith(".") and last_word and last_word[-1].lower() in _JOIN_AFTER) \
                    or sentence[:1].islower():
                joined[-1] = joined[-1].rstrip(".") + " " + sentence[:1].lower() + sentence[1:]
                continue
        joined.append(sentence)
    return joined


def _is_filler(sentence: str, terms) -> bool:
    if not sentence or _PURE_FILLER.match(sentence):
        return True
    if terms and any(t in sentence.lower() for t in terms):
        return False
    content = _content_words(sentence)
    if not content:
        return True
    return len(sentence.split()) <= 4 and len(content) <= 1 and not re.search(r"\d", sentence)


def clean_sentences(sentences: list[str], terms=()) -> list[str]:
    """Joins sentences a stray period cut in two, removes disfluencies and leading filler
    ("So, okay, ..."), and drops sentences that carry no content ("Right?", "So, this is it.")."""
    cleaned = [s for s in (_clean_sentence(s) for s in _join_fragments(sentences)) if re.search(r"[A-Za-z0-9]", s)]
    kept = [s for s in cleaned if not _is_filler(s, terms)]
    # Losing what was said is worse than keeping filler: when the filter would remove most of
    # the words (a short save, or speech it misjudges), everything said is kept instead.
    said = sum(len(s.split()) for s in cleaned)
    if said and sum(len(s.split()) for s in kept) < 0.4 * said and not all(_PURE_FILLER.match(s) for s in cleaned):
        return cleaned
    return kept


# ---------------------------------------------------------------------------
# Key terms: what to bold, name sections after, and define
# ---------------------------------------------------------------------------

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"
_notes_cache = {}
_BOLD_RE = re.compile(r"\*\*([^*\n]{3,48})\*\*")


def _bold_terms_in(path: Path) -> set[str]:
    """The terms Claude bolded in a class's notes, ignoring sections the heuristic formatter
    wrote (its bold is word frequency, not judgement)."""
    try:
        stat = path.stat()
    except OSError:
        return set()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _notes_cache:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = re.split(r"(?m)^## ", text)
        found = set()
        for chunk in chunks:
            if "*(auto-formatted locally" in chunk:
                continue
            for raw in _BOLD_RE.findall(chunk):
                term = re.sub(r"\s+", " ", raw.strip(" .,:;()'\"").lower())
                words = term.split()
                if not 1 <= len(words) <= 4 or not re.search(r"[a-z]", term):
                    continue
                if words[0] in _FUNCTION_WORDS or words[-1] in _FUNCTION_WORDS:
                    continue
                if len(words) == 1 and (len(term) < 4 or term in _FUNCTION_WORDS):
                    continue
                found.add(term)
        _notes_cache.clear() if len(_notes_cache) > 64 else None
        _notes_cache[key] = found
    return _notes_cache[key]


def class_key_terms(class_code: str | None) -> set[str]:
    """Key terms for a class: what Claude bolded in its notes so far, plus the class's own
    glossary. A single word bolded in two or more other classes too ("energy", "example") is
    left out: it isn't what distinguishes this class's material."""
    if not class_code:
        return set()
    own_path = NOTES_DIR / f"{class_code.replace(' ', '_')}.md"
    terms = set(_bold_terms_in(own_path))
    try:
        terms |= {t.lower() for t in vocab_module.class_only_terms(class_code)}
    except Exception:
        pass
    others = [_bold_terms_in(p) for p in NOTES_DIR.glob("*.md")
              if p != own_path and re.fullmatch(r"[A-Z]{2,5}_\d{3,4}\.md", p.name)]
    return {t for t in terms if " " in t or (len(t) >= 5 and t not in _FUNCTION_WORDS and not t.endswith("ly")
                                              and not any(t in o for o in others))}


def _phrases(sentences: list[str], min_count: int = 3) -> set[str]:
    """Two- and three-word phrases of content words said at least `min_count` times
    ("partial pressure", "mole fraction")."""
    counts = Counter()
    for sentence in sentences:
        words = re.findall(r"[a-z][a-z'\-]+", sentence.lower())
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if len(set(gram)) == n and all(len(w) >= 3 and w not in _FUNCTION_WORDS for w in gram):
                    counts[" ".join(gram)] += 1
    found = {p for p, c in counts.items() if c >= min_count}
    return {p for p in found if not any(p != q and p in q and counts[q] >= counts[p] for q in found)}


def lecture_terms(sentences: list[str], class_code: str | None) -> list[str]:
    """The key terms this lecture actually used, longest first (so "partial pressure" wins
    over "pressure")."""
    text = " " + " ".join(sentences).lower() + " "
    said = {t for t in class_key_terms(class_code)
            if re.search(r"(?<![a-z])" + re.escape(t) + r"(?:s|es)?(?![a-z])", text)}
    said |= _phrases(sentences)
    return sorted(said, key=lambda t: (-len(t.split()), -len(t), t))


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z*])(" + re.escape(term) + r"(?:s|es)?)(?![A-Za-z*])", re.IGNORECASE)


def _count(term: str, text: str) -> int:
    return len(_term_pattern(term).findall(text))


# ---------------------------------------------------------------------------
# Topic breaks from the vocabulary (a simplified TextTiling)
# ---------------------------------------------------------------------------

WORDS_PER_SECTION = 650     # about five minutes of speech
MIN_SECTION_SENTENCES = 8
_WINDOW = 6


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm = (sum(v * v for v in a.values()) * sum(v * v for v in b.values())) ** 0.5
    return dot / norm if norm else 0.0


def _segment(sentences: list[str]) -> list[list[str]]:
    """Splits where the vocabulary changes most: compares the words of the sentences before
    and after each gap, and cuts at the deepest dips in similarity. About one section per
    WORDS_PER_SECTION words, none shorter than MIN_SECTION_SENTENCES sentences."""
    n = len(sentences)
    words = sum(len(s.split()) for s in sentences)
    wanted = max(1, round(words / WORDS_PER_SECTION))
    if wanted == 1 or n < 2 * MIN_SECTION_SENTENCES:
        return [sentences]
    bags = [Counter(_stem(w) for w in _content_words(s)) for s in sentences]
    sims = []
    for gap in range(1, n):
        left, right = Counter(), Counter()
        for bag in bags[max(0, gap - _WINDOW):gap]:
            left.update(bag)
        for bag in bags[gap:gap + _WINDOW]:
            right.update(bag)
        sims.append(_cosine(left, right))
    depths = []
    for i, sim in enumerate(sims):
        lpeak = sim
        for j in range(i - 1, -1, -1):
            if sims[j] < lpeak:
                break
            lpeak = sims[j]
        rpeak = sim
        for j in range(i + 1, len(sims)):
            if sims[j] < rpeak:
                break
            rpeak = sims[j]
        depths.append((lpeak - sim) + (rpeak - sim))
    cuts = []
    for depth, gap in sorted(((d, i + 1) for i, d in enumerate(depths)), reverse=True):
        if len(cuts) >= wanted - 1 or depth <= 0:
            break
        if gap < MIN_SECTION_SENTENCES or n - gap < MIN_SECTION_SENTENCES:
            continue
        if all(abs(gap - c) >= MIN_SECTION_SENTENCES for c in cuts):
            cuts.append(gap)
    bounds = [0] + sorted(cuts) + [n]
    return [sentences[a:b] for a, b in zip(bounds, bounds[1:])]


def _section_name(section: list[str], everything: str, terms: list[str]) -> str | None:
    """Names a section after the key terms it uses most, relative to the whole lecture."""
    text = " ".join(section)
    scored = []
    for term in terms:
        here = _count(term, text)
        if here >= 2:
            scored.append((here * here / max(1, _count(term, everything)), len(term), term))
    if not scored:
        stems = Counter(_stem(w) for w in _content_words(text) if len(w) >= 6)
        surface = {}
        for w in _content_words(text):
            surface.setdefault(_stem(w), w)
        scored = [(c, 0, surface[s]) for s, c in stems.most_common(2) if c >= 3]
    ranked = sorted(scored, reverse=True)
    best = ranked[0][0] if ranked else 0
    # A phrase that scores at least half as well as the best single word names the section
    # better ("Mole fraction", not "Mole · Fraction").
    ranked.sort(key=lambda s: (-(s[0] * (2 if " " in s[2] and s[0] >= best / 2 else 1)), -s[1]))
    chosen = []
    for _, _, term in ranked:
        if not any(term in c or c in term for c in chosen):
            chosen.append(term)
        if len(chosen) == 2:
            break
    if len(chosen) == 2 and " " not in chosen[0] and " " not in chosen[1] \
            and _count(chosen[0] + " " + chosen[1], text):
        chosen = [chosen[0] + " " + chosen[1]]
    if not chosen:
        return None
    return " · ".join(t[:1].upper() + t[1:] for t in chosen)


# ---------------------------------------------------------------------------
# Bullets, bold, definitions, announcements
# ---------------------------------------------------------------------------

MAX_BULLET_WORDS = 55
_DEFINITION_CUE = r"\s+(?:is|are|means|meaning|refers to|is called|is defined as|is when|stands for|describes)\b"
_ANNOUNCEMENT = re.compile(
    r"\b(?:exams?|quiz|quizzes|midterm|final exam|test on|on the test|due|homework|assignment|"
    r"deadline|office hours|extra credit|don't forget|reminder|turn in|submit|study guide|canvas|"
    r"syllabus|next class|next week|this week)\b", re.IGNORECASE)


def _bulletize(sentences: list[str]) -> list[str]:
    """Two or three related sentences per bullet: a new bullet starts when the next sentence
    shares no content word with the current one, or the bullet is long enough. A question the
    lecturer asks stays with the sentence that answers it."""
    bullets, current, current_words = [], [], set()
    for sentence in sentences:
        words = {_stem(w) for w in _content_words(sentence)}
        if current:
            size = sum(len(s.split()) for s in current)
            ends_question = current[-1].endswith("?")
            unrelated = not (words & current_words)
            too_long = size + len(sentence.split()) > MAX_BULLET_WORDS
            if size > MAX_BULLET_WORDS * 1.4 or not ends_question and (
                    too_long or len(current) >= 3 or (unrelated and size >= 12)):
                bullets.append(" ".join(current))
                current, current_words = [], set()
        current.append(sentence)
        current_words |= words
    if current:
        bullets.append(" ".join(current))
    return bullets


def _bold_first_mentions(text: str, terms: list[str], bolded: set[str]) -> str:
    """Bolds each key term the first time it appears in the notes (bolded is shared across the
    whole lecture). Longer terms go first, and nothing inside an existing **...** changes."""
    for term in terms:
        if term in bolded:
            continue
        pattern = _term_pattern(term)
        pos = 0
        while True:
            match = pattern.search(text, pos)
            if not match:
                break
            inside = text[:match.start()].count("**") % 2 == 1
            if not inside:
                text = text[:match.start()] + "**" + match.group(1) + "**" + text[match.end():]
                bolded.add(term)
                break
            pos = match.end()
    return text


_NOT_A_DEFINITION = re.compile(r"^\s*(?:is|are)\s+(?:important|pretty|really|very|so|not|also|just|"
                               r"going|gonna|kind of|sort of|a lot|here|there|the same|why|what|how)\b",
                               re.IGNORECASE)


def _label_ending_at(sentence: str, end: int) -> str:
    """The noun phrase that ends at `end`: the term plus content words right before it, so
    "...because partial pressure means..." is labelled "partial pressure", not "pressure"."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sentence[:end])
    label = []
    for word in reversed(words):
        if word.lower() in _FUNCTION_WORDS or len(label) == 3:
            break
        label.insert(0, word.lower())
    return " ".join(label)


def _definitions(sentences: list[str], terms: list[str], limit: int = 6) -> list[tuple[str, str]]:
    """Sentences that say what a key term is: the term opens the sentence ("Partial pressure
    is...", "The mole fraction means..."), says so unmistakably anywhere ("...so personality is
    essentially..."), or the sentence names it ("...is called a buffer"). One entry per idea:
    "partial pressures" is not listed again after "partial pressure"."""
    found, used = [], []
    for sentence in sentences:
        if len(sentence.split()) > 40:
            continue
        for term in terms:
            stem = " ".join(_stem(w) for w in term.split())
            if any(stem in u or u in stem for u in used):
                continue
            word = r"(?<![a-z])(" + re.escape(term) + r"(?:s|es)?)"
            opening = re.match(r"(?:the |a |an )?" + word + r"(" + _DEFINITION_CUE + r".*)", sentence, re.IGNORECASE)
            strong = re.search(word + r"\s+(?:(?:is|are)\s+(?:essentially|basically|defined as|simply)|means|refers to)\b",
                               sentence, re.IGNORECASE)
            naming = re.search(r"\b(?:is|are) (?:called|known as)\s+(?:an?\s+|the\s+)?" + re.escape(term) + r"\b",
                               sentence, re.IGNORECASE)
            if opening and not _NOT_A_DEFINITION.match(opening.group(2)):
                label = _label_ending_at(sentence, opening.end(1))
            elif strong:
                label = _label_ending_at(sentence, strong.end(1))
            elif naming:
                label = term
            else:
                continue
            label = label if term.split()[-1] in label else term
            label_stem = " ".join(_stem(w) for w in label.split())
            found.append((label, sentence))
            used.extend({stem, label_stem})
            break
        if len(found) >= limit:
            break
    return found


def _render_lecture(sentences: list[str], terms: list[str], bolded: set[str]) -> tuple[list[str], list[str]]:
    """Markdown lines for a stretch of lecture, and the names of its sections."""
    everything = " ".join(sentences)
    lines, names = [], []
    sections = _segment(sentences)
    named = []
    for section in sections:
        name = _section_name(section, everything, terms) if len(sections) > 1 or len(everything.split()) >= 150 else None
        if named and name and name == named[-1][0]:
            named[-1] = (name, named[-1][1] + section)
        else:
            named.append((name, section))
    for name, section in named:
        if name:
            lines.append(f"### {name}")
            names.append(name)
        for bullet in _bulletize(section):
            lines.append("- " + _bold_first_mentions(bullet, terms, bolded))
        lines.append("")
    return lines, names


def _render_qa_block(items) -> list[str]:
    lines = ["### Q&A", ""]
    for label, sentence in items:
        lines.append(f"- **{label}:** {sentence}")
    lines.append("")
    return lines


def _front_matter(sentences: list[str], terms: list[str]) -> tuple[list[str], set[str]]:
    """The announcements box and key-terms list that go above the notes, and the sentences
    moved into the box (so they aren't repeated below)."""
    lines, moved = [], set()
    notices = [s for s in sentences if _ANNOUNCEMENT.search(s) and len(s.split()) >= 4][:8]
    if notices:
        lines += ["**Announcements & reminders**", ""] + [f"- {s}" for s in notices] + [""]
        moved = set(notices)
    frequent = Counter(w for s in sentences for w in _content_words(s) if len(w) >= 6 and not w.endswith("ly"))
    candidates = terms + [w for w, c in frequent.most_common(15) if c >= 5 and w not in terms]
    definitions = _definitions([s for s in sentences if s not in moved], candidates)
    if definitions:
        lines += ["**Key terms**", ""]
        lines += [f"- **{t[:1].upper() + t[1:]}** — {s}" for t, s in definitions] + [""]
    return lines, moved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def format_transcript(class_title: str, session_date: str, transcript: str,
                       class_code: str | None = None, reason: str | None = None) -> str:
    """Notes without any language model: cleaned-up sentences grouped into short bullets,
    sections where the vocabulary changes (named after their key terms), the class's key terms
    bolded at first mention, plus announcements and definitions pulled to the top."""
    transcript = correct_with_glossary(transcript, class_code)
    speaker_entries = _parse_speaker_entries(transcript)
    why = (reason or "Claude CLI/API unavailable").replace("*", "").replace("\n", " ").strip()
    header = [f"## {session_date}", "", f"*(auto-formatted locally — {why})*", ""]
    empty = f"## {session_date}\n\n*(auto-formatted locally — no speech detected)*\n"

    if speaker_entries is None:
        raw = _split_sentences(transcript)
        terms = lecture_terms(raw, class_code)
        sentences = clean_sentences(raw, terms)
        if not sentences:
            return empty
        front, moved = _front_matter(sentences, terms)
        bolded = set()
        body, names = _render_lecture([s for s in sentences if s not in moved], terms, bolded)
        topics = ["**Topics covered:** " + "; ".join(names), ""] if len(names) >= 2 else []
        return "\n".join(header + front + topics + body).rstrip() + "\n"

    labeled = _label_by_role(speaker_entries)
    blocks = _segment_qa(labeled)
    lecture_sentences = [s for block_type, items in blocks if block_type == "lecture" for _, s in items]
    terms = lecture_terms(lecture_sentences, class_code)
    cleaned_blocks = []
    for block_type, items in blocks:
        if block_type == "qa":
            kept = [(label, _clean_sentence(s)) for label, s in items]
            kept = [(label, s) for label, s in kept if s and not _PURE_FILLER.match(s)]
            if kept:
                cleaned_blocks.append(("qa", kept))
        else:
            sentences = clean_sentences([s for _, s in items], terms)
            if sentences:
                cleaned_blocks.append(("lecture", sentences))
    if not cleaned_blocks:
        return empty
    front, moved = _front_matter([s for kind, items in cleaned_blocks if kind == "lecture" for s in items], terms)
    bolded, body, names = set(), [], []
    for block_type, items in cleaned_blocks:
        if block_type == "qa":
            body.extend(_render_qa_block(items))
        else:
            lines, block_names = _render_lecture([s for s in items if s not in moved], terms, bolded)
            body.extend(lines)
            names.extend(block_names)
    topics = ["**Topics covered:** " + "; ".join(names), ""] if len(names) >= 2 else []
    return "\n".join(header + front + topics + body).rstrip() + "\n"

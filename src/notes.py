"""Turns a raw transcript into clean notes and appends them to that class's notes file.

Formatting is tried in this order:
  1. Claude Code CLI (`claude -p`) - uses your logged-in Pro/Max subscription,
     no per-token API billing.
  2. Claude API (`ANTHROPIC_API_KEY`) - only used if the CLI isn't available/logged in.
  3. Local, dependency-free formatter - used if we're offline or both of the above fail.
"""
import difflib
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import docx_export
import gpu_formatter
import local_formatter
import vocab as vocab_module

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"
NOTES_DIR.mkdir(exist_ok=True)

CLAUDE_MODEL = "claude-sonnet-5"
TAIL_CONTEXT_CHARS = 3000  # how much of the existing notes file we show Claude for continuity
CLI_TIMEOUT_SECONDS = 180

# Native installer (irm https://claude.ai/install.ps1 | iex) puts it here; also check PATH.
_FALLBACK_CLI_PATH = Path.home() / ".local" / "bin" / "claude.exe"

# Real, observed bug: without this, `claude -p` sometimes decides to act as a full agentic
# coding session instead of a plain text-completion call - exploring the project with
# Read/Glob to figure out "the right file" and attempting to Write directly to it, rather
# than returning the notes as its response text (which is all _try_claude_cli_format
# actually asked for - the whole prompt is self-contained, no exploration is ever needed).
# Headless (-p) mode can't interactively grant that Write permission, so it prints an
# "I need your approval to write to <file>" explanation as its answer instead of the
# notes - which then gets saved verbatim as if it were real content. Denying every tool
# forces a plain text response every time, which is the only thing this call ever wanted.
_CLI_NO_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
                 "NotebookEdit", "Task", "TodoWrite", "Agent", "ExitPlanMode")


def _find_claude_cli() -> str | None:
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    if _FALLBACK_CLI_PATH.exists():
        return str(_FALLBACK_CLI_PATH)
    return None


def cli_available() -> bool:
    return _find_claude_cli() is not None


def notes_path(class_code: str) -> Path:
    safe = class_code.replace(" ", "_")
    return NOTES_DIR / f"{safe}.md"


def _read_existing(class_code: str) -> str:
    p = notes_path(class_code)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _build_proofread_prompt(class_title: str, class_code: str, transcript: str) -> str:
    terms = vocab_module.terms_for_class(class_code)
    glossary_line = ("Known vocabulary for this class (fix mis-transcriptions toward these "
                      f"when it's clearly the intended word): {', '.join(terms)}\n\n") if terms else ""
    return f"""You are proofreading a raw speech-to-text transcript of a college lecture
for {class_title} ({class_code}). This is a cleanup pass only, not a rewrite.

{glossary_line}Fix ONLY:
- spelling and punctuation
- obvious speech-to-text mis-hearings (e.g. a mangled technical/Latin term that clearly
  should be a real word given context)
- grammar, without changing meaning

Do NOT:
- summarize, shorten, or reword for style
- remove content, even filler-sounding statements, unless it's pure stutter/repetition noise
- add any information that wasn't said

If a sentence contains a claim that seems factually off (likely because a word was
mis-transcribed into something that doesn't make sense, e.g. a garbled term produced a
statement that contradicts basic facts about the subject), do NOT silently correct the
claim itself - leave the sentence as transcribed but append an inline flag right after it:
" [FLAG: possible transcription error - please verify]". Only use this flag when something
looks like a transcription artifact, not just because a claim is debatable or you're unsure.

Transcript (may contain "[Speaker N]" tags marking different speakers - keep those tags
exactly as-is, do not remove or move them):
---
{transcript}
---

Output ONLY the corrected transcript text, same structure/tags, no preamble or explanation.
"""


def _build_notes_prompt(class_title: str, class_code: str, session_date: str,
                         transcript: str, prior_tail: str) -> str:
    return f"""You are turning a cleaned-up speech-to-text lecture transcript into clean study notes.

Class: {class_title} ({class_code})
Date: {session_date}

Here is the END of the notes file from previous lectures, for context and continuity
(do not repeat this material, just use it to stay consistent and to note connections
when today's material clearly builds on it):
---
{prior_tail if prior_tail else "(no previous notes yet for this class)"}
---

Here is today's transcript to turn into notes. It may contain "[Speaker N]" tags if multiple
speakers were detected (diarization) - if present, treat the speaker who talks the most as
the instructor, and any back-and-forth with other speakers as Q&A: pull those into their own
"### Q&A" section formatted as "**Q (Student):** ..." / "**A (Instructor):** ...", separate
from the main lecture content, rather than interleaving them into the regular notes. If no
speaker tags are present, treat it as one continuous lecture.

It may also contain inline "[FLAG: possible transcription error - please verify]" markers from
an earlier proofreading pass - preserve these flags on the relevant point in the notes (e.g. as
a trailing "⚠️ *verify*" note) rather than silently dropping them; don't invent flags of your own.
---
{transcript}
---

Write clean, well-organized markdown notes for TODAY's lecture only. These are STUDY
NOTES a student reviews later to learn the material - not a recap/summary of the class
session itself. Requirements:
- Start with a level-2 heading: "## {session_date}"
- State the actual content directly and factually: "**Amygdala**: part of the limbic
  system, handles fear responses" - NOT "The professor discussed the amygdala and its
  role in fear responses" or "We covered how the amygdala relates to emotion." Avoid
  any "recap" framing ("today we learned...", "the lecture covered...", "discussion
  of..."). If the instructor themselves recaps or summarizes something mid-lecture,
  still extract and state the underlying facts directly - don't write a summary of a
  summary.
- Use bullet points and sub-bullets for concepts, bold key terms
- Fix obvious speech-to-text errors and filler words, but don't invent content that wasn't said
- Group related points under short level-3 headings if the lecture covered multiple topics
- Where a technical term or concept is mentioned but not fully explained in the transcript,
  you may add a brief one-line background definition from general knowledge of the subject
  to make the notes more self-contained and useful for studying - but clearly mark any such
  addition as supplementary, e.g. "*(background: ...)*", so it's never confused with
  something the instructor actually said. Only add these where they'd genuinely help
  comprehension of an under-explained term, not for every term that appears.
- If something clearly connects to previous material, add a brief note like "*(builds on ...)*"
- Do not include a preamble or explanation, output only the markdown notes section
"""


def _try_claude_cli_format(prompt: str) -> str | None:
    """Returns clean markdown via the Claude Code CLI (subscription usage), or None if unusable."""
    cli = _find_claude_cli()
    if not cli:
        return None
    try:
        result = subprocess.run(
            [cli, "-p", "--model", CLAUDE_MODEL, "--disallowedTools", " ".join(_CLI_NO_TOOLS)],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=CLI_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None
        text = result.stdout.strip()
        return text or None
    except Exception:
        # CLI not logged in, timed out, offline, etc. -> fall back.
        return None


def _try_claude_api_format(prompt: str) -> str | None:
    """Returns clean markdown via the Anthropic API, or None if the API isn't usable right now."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return text.strip() or None
    except Exception:
        # Offline, invalid key, rate-limited, etc. -> fall back to local formatting.
        return None


def _local_format(class_title: str, session_date: str, transcript: str, class_code: str) -> str:
    """Dependency-free fallback: heuristic topic grouping, bullets, key-term bolding,
    and fuzzy glossary correction (no proofreading pass - that needs an LLM)."""
    return local_formatter.format_transcript(class_title, session_date, transcript, class_code)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _detect_corrections(original: str, corrected: str) -> set[str]:
    """Word-level diff between the raw and proofread transcript, returning the set of
    "corrected-to" words where the change looks like a mis-transcription fix (the old
    and new words are similar enough to plausibly be the same word) rather than a
    genuine reword/edit - candidates to persist into vocab.json so the heuristic/local
    formatters catch the same term next time without needing the CLI/API. Deliberately
    conservative (single-word replacements only, similarity-gated, min length 5) to
    avoid learning junk from ordinary proofreading edits."""
    orig_words = _WORD_RE.findall(original)
    corr_words = _WORD_RE.findall(corrected)
    matcher = difflib.SequenceMatcher(a=[w.lower() for w in orig_words], b=[w.lower() for w in corr_words])

    candidates = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        old_word, new_word = orig_words[i1], corr_words[j1]
        if len(new_word) < 5:
            continue
        similarity = difflib.SequenceMatcher(a=old_word.lower(), b=new_word.lower()).ratio()
        if 0.6 <= similarity < 1.0:
            candidates.add(new_word)
    return candidates


def _build_combined_prompt(class_title: str, class_code: str, session_date: str,
                            transcript: str, prior_tail: str) -> str:
    """Same end result as running _build_proofread_prompt then _build_notes_prompt as
    two separate CLI/API calls, but as ONE call/prompt instead. Used for the expensive
    full-session-diarization save at Ctrl+C (a large, latency-sensitive transcript,
    where two sequential round-trips over the whole thing was the dominant cost -
    measured ~200s vs. a much smaller per-chunk save where the extra round-trip barely
    matters). Trades away the separate proofread-only output that vocab-learning
    diffs against - that's fine here since it's a one-shot, whereas the smaller
    per-chunk path (which keeps the two-call version) already gets plenty of chances
    to learn vocabulary over the course of a session."""
    terms = vocab_module.terms_for_class(class_code)
    glossary_line = ("Known vocabulary for this class (fix mis-transcriptions toward these "
                      f"when it's clearly the intended word): {', '.join(terms)}\n\n") if terms else ""
    return f"""You are turning a raw speech-to-text lecture transcript into clean study notes.
Class: {class_title} ({class_code})
Date: {session_date}

{glossary_line}First, silently account for transcription noise as you read the transcript below -
fix obvious spelling/mis-hearings (e.g. a mangled technical/Latin term that clearly should
be a real word given context) and grammar in your head, without changing meaning - but do
NOT output a corrected transcript; go straight to writing the final notes as described below.
If a specific claim in the transcript seems factually off in a way that looks like it's
caused by a transcription error (a garbled term producing something that contradicts basic
facts about the subject), keep your best-guess correction in the notes but append "⚠️
*verify*" right after that point - only for likely transcription artifacts, not because a
claim is merely debatable.

Here is the END of the notes file from previous lectures, for context and continuity
(do not repeat this material, just use it to stay consistent and to note connections
when today's material clearly builds on it):
---
{prior_tail if prior_tail else "(no previous notes yet for this class)"}
---

Here is today's transcript. It may contain "[Speaker N]" tags if multiple speakers were
detected (diarization) - if present, treat the speaker who talks the most as the instructor,
and any back-and-forth with other speakers as Q&A: pull those into their own "### Q&A"
section formatted as "**Q (Student):** ..." / "**A (Instructor):** ...", separate from the
main lecture content, rather than interleaving them into the regular notes. If no speaker
tags are present, treat it as one continuous lecture.
---
{transcript}
---

Write clean, well-organized markdown notes for TODAY's lecture only. These are STUDY
NOTES a student reviews later to learn the material - not a recap/summary of the class
session itself. Requirements:
- Start with a level-2 heading: "## {session_date}"
- State the actual content directly and factually: "**Amygdala**: part of the limbic
  system, handles fear responses" - NOT "The professor discussed the amygdala and its
  role in fear responses" or "We covered how the amygdala relates to emotion." Avoid
  any "recap" framing ("today we learned...", "the lecture covered...", "discussion
  of..."). If the instructor themselves recaps or summarizes something mid-lecture,
  still extract and state the underlying facts directly - don't write a summary of a
  summary.
- Use bullet points and sub-bullets for concepts, bold key terms
- Don't invent content that wasn't said, and don't drop content just because it's long -
  this may be a full lecture's worth of material, cover all of it
- Group related points under short level-3 headings for each topic covered
- Where a technical term or concept is mentioned but not fully explained in the transcript,
  you may add a brief one-line background definition from general knowledge of the subject
  to make the notes more self-contained and useful for studying - but clearly mark any such
  addition as supplementary, e.g. "*(background: ...)*", so it's never confused with
  something the instructor actually said. Only add these where they'd genuinely help
  comprehension of an under-explained term, not for every term that appears.
- If something clearly connects to previous material, add a brief note like "*(builds on ...)*"
- Do not include a preamble or explanation, output only the markdown notes section
"""


def _proofread(class_title: str, class_code: str, transcript: str) -> str:
    """Runs the transcript through a proofread/flag pass via CLI or API. Returns the
    original transcript unchanged if neither is available (local-only mode). Also
    detects likely vocabulary corrections and persists them to vocab.json (best-effort,
    never lets a failure here affect the actual proofreading result)."""
    prompt = _build_proofread_prompt(class_title, class_code, transcript)
    cleaned = _try_claude_cli_format(prompt) or _try_claude_api_format(prompt)
    if cleaned:
        try:
            vocab_module.save_learned_terms(class_code, _detect_corrections(transcript, cleaned))
        except Exception:
            pass
    return cleaned or transcript


FORMATTING_MODES = ("auto", "local", "heuristic")
# auto:      CLI -> API -> GPU model -> heuristic (default - best available each step)
# local:     GPU model -> heuristic only (no network calls - CLI/API never contacted)
# heuristic: heuristic only (no LLM anywhere - fastest, fully deterministic)


def format_and_save(class_code: str, class_title: str, transcript: str,
                     session_date: str | None = None, mode: str = "auto",
                     combine_proofread: bool = False) -> Path:
    """Formats `transcript` into notes and appends them to notes/<CODE>.md. Returns the file path.
    `mode` controls which formatting tiers are allowed - see FORMATTING_MODES above.
    `combine_proofread`: do proofreading + notes-formatting as one CLI/API call instead
    of two sequential ones (see _build_combined_prompt) - worth it for a large,
    latency-sensitive transcript (e.g. the full-session-diarization save at Ctrl+C),
    not worth the lost vocab-learning signal for the normal small per-chunk saves."""
    if mode not in FORMATTING_MODES:
        raise ValueError(f"Unknown formatting mode {mode!r}, expected one of {FORMATTING_MODES}")
    if not session_date:
        now = datetime.now()
        session_date = f"{now.strftime('%A, %B')} {now.day}, {now.strftime('%Y')}"
    transcript = transcript.strip()
    if not transcript:
        return notes_path(class_code), "none"

    existing = _read_existing(class_code)
    prior_tail = existing[-TAIL_CONTEXT_CHARS:] if existing else ""

    section = None
    method = None

    if mode == "auto":
        if combine_proofread:
            prompt = _build_combined_prompt(class_title, class_code, session_date, transcript, prior_tail)
        else:
            proofread_transcript = _proofread(class_title, class_code, transcript)
            prompt = _build_notes_prompt(class_title, class_code, session_date, proofread_transcript, prior_tail)
        section = _try_claude_cli_format(prompt)
        method = "cli"
        if section is None:
            section = _try_claude_api_format(prompt)
            method = "api"

    # An NPU tier (Phi-3.5-mini via OpenVINO) was tried here first and rolled back: on
    # the real pipeline it produced degenerate repetition with default decoding, and
    # hallucinated/incoherent rambling once repetition_penalty was added to fix that.
    # This GPU tier (Qwen2.5-3B via llama.cpp/CUDA, actually reaching the discrete GPU
    # unlike OpenVINO's Intel-only "GPU" device) has been reliable in testing for this
    # per-chunk task specifically - see gpu_formatter.py for the full story.
    if section is None and mode in ("auto", "local"):
        section = gpu_formatter.format_transcript(class_title, session_date, transcript)
        if section is not None:
            # Smaller model than Claude, inconsistently catches mis-transcribed
            # vocabulary - run the same fuzzy glossary fix-up the heuristic formatter
            # uses, as a safety net.
            section = local_formatter.correct_with_glossary(section, class_code)
        method = "gpu"

    if section is None:
        section = _local_format(class_title, session_date, transcript, class_code)
        method = "local"

    path = notes_path(class_code)
    if not existing:
        header = f"# {class_title} ({class_code})\n\n"
        path.write_text(header + section + "\n\n", encoding="utf-8")
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + section + "\n\n")

    try:
        docx_export.append_section(class_code, class_title, section)
    except Exception:
        # .docx is a nice-to-have mirror of the .md; never let it block saving notes.
        pass

    return path, method


def _build_condense_prompt(class_title: str, class_code: str, session_markdown: str) -> str:
    return f"""You are reviewing the notes generated during ONE lecture recording session for
{class_title} ({class_code}) and cleaning them up before they're finalized. During the session,
notes may have been saved multiple times (autosaves), which can result in:
- multiple separate date headings for what is really the same lecture
- duplicate or near-duplicate bullet points repeated across those saves
- formatting inconsistencies (inconsistent heading levels, stray bullets, broken markdown)
- leftover raw/unformatted transcript fragments that never got cleaned up
- inline "[FLAG: possible transcription error - please verify]" or "⚠️ verify" markers

Your job: merge everything below into ONE clean, well-organized section for this lecture.

Requirements:
- Produce exactly ONE "## <date>" heading (reuse the date already present, don't invent one)
- Merge duplicate/overlapping points into a single clean bullet - don't just concatenate them
- Fix any broken markdown (unclosed bold/italic, inconsistent bullet indentation/nesting,
  stray or misapplied headings)
- If you find text that still looks like an unformatted raw transcript fragment (run-on
  sentences, filler words, no bullet structure), reformat it into proper notes rather than
  leaving it as-is
- Preserve any "⚠️ verify" / "[FLAG: ...]" markers on the specific point they were attached
  to - don't drop them, but you may reword the surrounding sentence for clarity
- Keep an existing "### Q&A" section as its own section if present, cleaned up the same way
- Don't invent content that wasn't in the original notes below
- Output ONLY the final markdown for this section, no preamble or explanation

Notes to clean up:
---
{session_markdown}
---
"""


def study_guide_path(class_code: str) -> Path:
    safe = class_code.replace(" ", "_")
    return NOTES_DIR / f"{safe}_study_guide.md"


def _build_study_guide_prompt(class_title: str, class_code: str, all_notes: str) -> str:
    return f"""You are building a consolidated exam-prep study guide for {class_title} ({class_code})
from that class's full set of per-lecture notes below, spanning the whole semester so far.

This is NOT a lecture-by-lecture recap - a student should be able to read only this
document and have everything they need to review for an exam, without needing to
revisit the individual lecture notes it was built from.

Requirements:
- Organize by TOPIC/THEME across the whole semester, not by date or lecture session -
  merge related material from different lectures under the same heading (e.g. if
  "memory" came up across three separate lectures, it should appear once, combined)
- Use level-2 headings ("## <topic>") for major topics, with bullet points and
  sub-bullets underneath; bold key terms
- State facts directly, the way a study guide would ("**Amygdala**: part of the limbic
  system, handles fear responses"), not as a recap of what was taught or when
- Where the same concept was covered more than once across lectures (recapped, revisited,
  or built upon), merge those into one clean, complete treatment rather than repeating it -
  if a later lecture added nuance or corrected/extended an earlier point, reflect the most
  complete/current understanding
- Preserve any "⚠️ verify" / "[FLAG: ...]" transcription-uncertainty markers that are
  attached to specific points, so the student knows to double check those - don't
  invent new ones
- Keep Q&A content only where it adds information not already covered in the main
  material - fold a genuinely informative Q&A point into the relevant topic section
  rather than keeping a separate Q&A section
- Do not invent content that isn't present in the notes below
- Do not include a preamble, explanation, or meta-commentary about the notes themselves -
  output ONLY the final study guide markdown, starting with a level-1 heading:
  "# {class_title} - Study Guide"

Full notes to consolidate:
---
{all_notes}
---
"""


def generate_study_guide(class_code: str, class_title: str,
                          mode: str = "auto") -> tuple[bool, str | None, Path | None]:
    """Reads all of a class's accumulated per-lecture notes and asks Claude to produce
    one consolidated, topic-organized review document (notes/<CODE>_study_guide.md),
    good for exam prep - distinct from condense_session, which only cleans up ONE
    session's notes without cross-lecture synthesis. CLI/API only, same reasoning as
    condense_session: this is an even harder multi-section merge task than condensing,
    and the local GPU model's unreliability on that smaller task makes it a bad fit
    here too. Returns (True, None, path) on success, or (False, reason, None)."""
    if mode not in FORMATTING_MODES:
        raise ValueError(f"Unknown formatting mode {mode!r}, expected one of {FORMATTING_MODES}")
    if mode != "auto":
        return False, f"no LLM available in '{mode}' mode", None

    path = notes_path(class_code)
    if not path.exists():
        return False, "no notes yet for this class", None

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return False, "no notes yet for this class", None

    prompt = _build_study_guide_prompt(class_title, class_code, content)
    guide = _try_claude_cli_format(prompt) or _try_claude_api_format(prompt)
    if not guide:
        return False, "Claude CLI/API unavailable", None

    out_path = study_guide_path(class_code)
    out_path.write_text(guide.strip() + "\n", encoding="utf-8")

    try:
        docx_export.save_study_guide(class_code, class_title, guide.strip())
    except Exception:
        pass  # .docx is a mirror of the .md; never let it block saving the study guide

    return True, None, out_path


def condense_session(class_code: str, class_title: str, before_length: int,
                      mode: str = "auto") -> tuple[bool, str | None]:
    """Re-reviews everything written to this class's notes since `before_length`
    (the file's length when this recording session started), collapsing duplicate/
    multi-autosave sections into one clean section. Only runs in "auto" mode (CLI/API)
    - deliberately does NOT fall back to the local GPU model (gpu_formatter) the way
    format_and_save does. Testing showed the GPU model handles single-chunk formatting
    reliably but not this harder multi-section merge/dedup task: across repeated runs
    it would inconsistently drop one genuinely distinct bullet (which one varied by
    generation params) while satisfying the other instructions - a real content-loss
    risk that matters more here since condensing rewrites/replaces existing notes,
    unlike format_and_save which only appends. So "local"/"heuristic" modes are always
    a no-op for condensing for now. Returns (True, None) on success, or (False, reason)
    if there was nothing to do or the CLI/API aren't available right now."""
    if mode not in FORMATTING_MODES:
        raise ValueError(f"Unknown formatting mode {mode!r}, expected one of {FORMATTING_MODES}")
    if mode != "auto":
        return False, f"no LLM available in '{mode}' mode"

    path = notes_path(class_code)
    if not path.exists():
        return False, "no notes file yet"

    content = path.read_text(encoding="utf-8")

    # The top-level "# Class Title (CODE)" heading is written once by the very first
    # save ever made for this class and isn't part of any single lecture's content -
    # always keep it out of what gets handed to the condense pass, even when this is
    # that first-ever session (before_length == 0, so it would otherwise fall inside
    # the "session" range and the condense pass would have no reason to keep it).
    title_end = 0
    if content.startswith("# "):
        split_at = content.find("\n\n")
        if split_at != -1:
            title_end = split_at + 2
    preserve_length = max(before_length, title_end)

    if preserve_length >= len(content):
        return False, "nothing new to condense"

    header_part = content[:preserve_length]
    session_part = content[preserve_length:]
    if not session_part.strip():
        return False, "nothing new to condense"

    prompt = _build_condense_prompt(class_title, class_code, session_part)
    condensed = _try_claude_cli_format(prompt) or _try_claude_api_format(prompt)
    if not condensed:
        return False, "Claude CLI/API unavailable"

    new_content = header_part.rstrip("\n") + "\n\n" + condensed.strip() + "\n"
    path.write_text(new_content, encoding="utf-8")

    try:
        docx_export.rebuild(class_code, class_title, new_content)
    except Exception:
        pass  # .docx is a mirror of the .md; never let it block the condensed save

    return True, None

# Lecture Notes App

## Starting Notes from diagnostics

Open the dashboard's **Diagnostics** page and click **Start Notes**. It opens the
existing recorder startup window, where you choose the action, class, audio source,
and formatting mode. Recording begins only after that setup and the normal preflight
checks. Diagnostics shows "Notes window open" during startup, then live session
status once capture starts. Use **Save now** and **Stop recording** there as usual.

The Start button is disabled while a recorder process is open, including a recorder
launched outside the dashboard. Closing that window makes Start available again.
The dashboard reports launch/command errors rather than silently ignoring them.
The dashboard and recorder remain separate processes, so restarting or closing the
dashboard does not stop a recording. Restart the dashboard after installing this update.

## Correcting a lecture

Open **Corrections** in the dashboard, or **Correct this lecture** in Recordings.

1. Choose a recording. Click a transcript segment to seek the audio, then press Play.
2. Edit the transcript text. Optionally enter a corrected technical term to remember
   in that class's glossary, then select **Save transcript correction**.
3. Choose the precise note section to update. Check the transcript segments that
   support it; use the filter to find relevant passages in a long recording.
4. Select a regeneration method and generate a preview. Local mode is the default;
   Claude mode sends only the selected section and excerpt to the CLI/API. The
   no-model option produces lossless transcript bullets. The preview identifies
   which method actually succeeded.
5. Review the before/after text, edit the proposed content if needed, and select
   **Apply this section only**. Other sections remain unchanged. Keep the original
   section heading when editing the preview.

Corrections are stored in `state/<recording>_corrections.json`; original audio,
raw text, and timed segments remain intact. Search and recovery use reviewed
wording. Notes, glossary changes, and correction replacements preserve revisions.
The existing revision restoration command can restore the prior notes file.

Legacy recordings without precise timing still support text correction and
whole-recording playback. Legacy notes without recording markers require you to
choose their section manually. Finish recording a class before editing its notes.
Concurrent edits are checked when saving and when applying previews; reload after
a conflict. Unsaved transcript edits have a **Discard edit** control.

Restart the dashboard server after updating to load the new API routes. Generation
runs in the background; previews expire after one hour or a dashboard restart.

## Lessons: narrated TL;DR slideshows

Open **Lessons** in the dashboard, pick a class and a range of lecture dates (or a quick
range like "Last lecture" / "Past week"), choose a narrator voice and select **Build
lesson**. A lesson takes a few minutes to build in the background and then replays
instantly from **Your lessons**.

- **What it teaches**: the notes in that date range decide what's covered - everything
  testable in them is taught - but the explanations go well beyond them: plain-language
  re-explanations, analogies, worked examples, context and common mistakes, written by
  Claude (CLI/API, same model as note formatting). The notes come from speech
  transcription, so where they look garbled or wrong the slide teaches the correct version
  and shows a **Heads up** saying what the notes said. Lessons end with a recap and quick
  check questions (answers hidden until you reveal them).
- **Visuals**: diagrams (processes, cycles, comparisons, timelines) are drawn with Mermaid,
  bundled in `dashboard/vendor/` so the player works offline. Pictures are searched on
  Wikipedia (article images), Wikimedia Commons and Openverse, and Claude picks the best
  candidate for each slide from their descriptions - or none, if nothing fits. Molecules
  are drawn as sharp vector images with RDKit from PubChem's structure data (every
  hydrogen shown for small molecules like H₂O), falling back to PubChem's own drawing.
  Transparent textbook figures are placed on white so they stay readable on the dark
  theme. Anything that fails validation is left out rather than shown wrong.
- **Narration**: generated offline on the CPU with Kokoro-82M (`src/speech.py`), one clip
  per slide - about half a second of work per second of speech on this laptop, so a
  12-minute lesson narrates in about 6 minutes. The player auto-advances with
  the narration, and has speed control (0.85-1.5x), captions, a clickable slide bar,
  fullscreen, and keys: ←/→ slides, Space play/pause, C captions, A auto-advance, F
  fullscreen.
- **Storage**: `lessons/<id>/` holds `lesson.json`, the pictures and the narration. Builds
  happen in a `.partial` folder that's only renamed into place when complete, so a failed
  build never leaves a broken lesson. One lesson builds at a time.
- **Setup**: `pip install -r requirements.txt` (adds `kokoro-onnx` and `pillow`), then
  download the two voice model files listed at the top of `src/speech.py` into
  `state/kokoro/`. Without them, lessons are built without narration. Restart the
  dashboard after updating to load the Lessons page.

## Recording, recovery, and playback improvements

- Each captured audio block is flushed to the WAV **before** entering the
  transcription queue. Failed inference preserves the source audio; backlog
  entries store disk offsets rather than retaining whole audio arrays.
- New sessions save `_segments.jsonl` beside the recording, with audio-relative
  start/end times. Search results expose **Play this moment**, and Recordings
  provides an audio player with seeking. Older recordings retain whole-recording
  playback; their legacy wall-clock timestamps are not used for inaccurate seeks.
  Recovering an old WAV creates precise segment offsets after transcription succeeds.
- Autosaves, final diarization, and recovery use session IDs inside Markdown
  comments. Repeating recovery replaces that session's contribution. Earlier
  same-day sessions are preserved. Legacy unmarked notes are left intact because
  their ownership cannot be reconstructed reliably; the first recovery may coexist
  with those old notes, while subsequent recoveries replace the marked section.
- Every replaced notes file is saved with an atomic write and a prior-version
  snapshot under `notes/.revisions/<filename>/`. Formatting failures no longer
  truncate the old notes before replacement content is ready. The Word mirror is
  rebuilt after saves and excludes the internal session comments.
- Recovery reads WAV data in chunks, and cleanup closes the recording even when
  final processing fails. A disk write failure stops capture and reports the error.

To list revisions and restore one (stop recording before restoring):

```powershell
venv\Scripts\python.exe tools/restore_revision.py --class "BIOL 1440"
venv\Scripts\python.exe tools/restore_revision.py --class "BIOL 1440" --revision <filename-from-list>
```

Restoring preserves the current version too. Revisions are not automatically
deleted. Run offline regressions with `venv\Scripts\python.exe -m pytest tests -q`.
Physical microphone/GPU behavior should still be smoke-tested before a lecture.

Auto-detects which class you're in (from `schedule.json`, based on the day/time),
records + transcribes the lecture fully locally (Cohere Transcribe), and turns the transcript into clean notes
appended to that class's ongoing notes file.

Note formatting is tried in this order, each falling back to the next if unavailable:
1. **Claude Code CLI** (`claude -p`) — uses your logged-in Pro/Max subscription, no per-token billing
2. **Claude API** (`ANTHROPIC_API_KEY`) — only used if the CLI isn't installed/logged in
3. **Local GPU model** (Qwen2.5-3B via llama.cpp, on the discrete GPU) — real LLM formatting, fully offline, used if neither the CLI nor API is available
4. **Heuristic local formatter** — regex-based topic/bullet formatting + glossary correction, last resort if the local model isn't set up either

Also:
- Notes are written as **direct study content, not a recap of the lecture** —
  e.g. "**Amygdala**: part of the limbic system, handles fear responses," not
  "The professor discussed the amygdala and its role in fear." Where a term is
  mentioned but not fully explained, the CLI/API tier may add a brief
  **background definition** from general subject knowledge to make notes more
  self-contained - always clearly marked `*(background: ...)*` so it's never
  confused with something the instructor actually said.
- Whenever the CLI/API is used, a **proofreading pass** runs first — fixes
  spelling/grammar/mis-heard technical terms without changing what was actually
  said, and flags (never silently "corrects") any statement that looks like a
  transcription artifact producing something factually odd.
- **When you stop recording (Ctrl+C)**, if the CLI is available it re-reviews
  everything saved during that session (across any autosaves) in one pass —
  merging duplicate/repeated sections from multiple autosaves into one clean
  section, fixing formatting bugs, and cleaning up anything that still looks
  like unformatted raw transcript. This pass is CLI/API-only by design (see
  "Local GPU note formatting" below for why the local model isn't used here).
- Output is written as both **Markdown** (`notes/<CODE>.md`) and **Word** (`notes/<CODE>.docx`),
  kept in sync, appended lecture by lecture.
- If speaker diarization is set up (see below), **Q&A exchanges get their own section**,
  separated from the main lecture content.
- The live console view is color-coded: cyan timestamps, white transcript text,
  green for successful saves, yellow/red for fallback or error states, magenta
  for autosave markers.
- **One transcription pass**: the terminal and dashboard show the accurate output
  used for notes, normally after each two-minute Cohere window plus processing time.
  Lines use blue timestamps and light text without pass headings. Audio keeps
  recording between updates; no extra preview inference runs.

## One-time setup

### Performance and recovery checks

Transcription retries an inference failure once, restoring rolling context before
retrying. Result-writing callbacks are never automatically replayed. Unresolved
windows are recorded beside the audio as `.failed.json`, and shutdown reports the
existing `--resume` command. Resume still recovers the session from its backup;
the ledger identifies gaps rather than introducing another recovery format.

Stage durations (without transcript text) are appended to `state/performance.jsonl`.
They include audio reads, transcription windows, Cohere preprocessing/inference,
proofreading, formatting calls, and Word export.

`tools/quality_benchmark.py` supports `format` comparisons on a saved transcript
and `accuracy` comparisons on a JSON list of audio cases. It refuses to run while
the recorder is open. Pass `--output` with a new report filename. Formatting uses
Claude CLI and writes only the report, never class notes. Accuracy cases provide
`audio`, human-checked `reference`, and optional `start`, `seconds`, and
`quiet_padding` (seconds added on each side). Audio must be mono 16 kHz.

Combined formatting remains opt-in pending quality review. Real-model quiet-audio
testing and the duplicate-splitting optimization are pending measurement; tuned
recognition settings and timestamp splitting have not been changed.

```bash
cd C:/AI/Projects/lecture-notes
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Transcription model (one-time download, ~4GB):** the default model is
[Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).
Its repo is gated, so accept the terms on that page with the Hugging Face account
behind `HUGGINGFACE_TOKEN`, then run the download command in `requirements.txt`. It
goes into this project's `state/cohere_transcribe/` rather than the shared Hugging Face
cache, so a general cache cleanup can't delete it. It's the only transcription model
(the Whisper fallback was removed): if it's missing or there's no CUDA GPU, preflight
says why and won't start the recording. After that everything runs fully offline.

**Recommended:** the Claude Code CLI is already installed
(`C:\Users\braxt\.local\bin\claude.exe`, on PATH as `claude`). Log in once with
your Pro/Max account so note formatting uses your subscription instead of
paid API credits:

```powershell
claude login
```

That's a one-time interactive step (opens a browser). After that, every
lecture's notes get formatted via the CLI automatically — no API key needed.

**Optional fallback:** if you'd rather use metered API billing instead of (or
in addition to) the CLI, set an API key — it's only used when the CLI isn't
logged in:

```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```

(Restart your terminal after `setx` so the variable takes effect. Without
either the CLI login or a key, the app still works — notes just get lightly
cleaned up locally instead of intelligently written/merged.)

**Optional: speaker diarization (for Q&A sections).** Off by default. To
enable real speaker detection so questions/answers get pulled into their own
section:

```powershell
./venv/Scripts/python.exe -m pip install torch pyannote.audio
```

**Also needs the *shared-library* build of FFmpeg** (pyannote.audio 4.x uses
`torchcodec` internally for audio decoding, which loads FFmpeg's DLLs directly -
a static/CLI-only FFmpeg build does NOT work, even though `ffmpeg` still runs
fine from the terminal with one installed):
```powershell
winget install --id Gyan.FFmpeg.Shared -e
```
(If you have the plain `Gyan.FFmpeg` static build installed, uninstall it first
so `ffmpeg`/PATH aren't ambiguous: `winget uninstall --id Gyan.FFmpeg -e`.)

Then:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the terms on all three gated models the pipeline depends on:
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), and
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   (the third one isn't obvious from pyannote's own docs - it only surfaces as a
   `GatedRepoError` the first time the pipeline tries to load, since it's an internal
   dependency of the top-level pipeline)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   logged into the **same account** that accepted the terms above
4. `setx HUGGINGFACE_TOKEN "hf_..."` (restart your terminal after)

This is a heavy install (~2GB, mostly PyTorch) and diarization adds noticeable
CPU time per save (runs once each time notes are saved, on just the newly
recorded audio since the last save). If it's not installed or the token isn't
set, the app runs exactly as before — no speaker labels, nothing breaks. If it
IS set up but something's wrong (wrong FFmpeg build, terms not accepted, a
pyannote.audio API change), `[diarize]`-prefixed errors print to the console
instead of silently doing nothing - if you see one, that's the actual problem
to fix, not something to ignore.

## Running it

```bash
./venv/Scripts/python.exe src/main.py
```

- Before anything else, it runs **preflight checks**: opens your audio device
  and confirms it's actually picking up sound (not silent/muted), checks disk
  space, and confirms the transcription model loads - so a broken mic or a dead GPU shows up now, not
  silently mid-lecture. A genuinely broken audio device or a transcription model that
  can't load at all stops the app here rather than starting a doomed session;
  other issues are just warnings and don't block starting.
- It checks `schedule.json` against the current day/time and tells you which
  class it thinks you're in (with a 10-minute grace window before/after, so
  starting the app slightly early or late still picks the right class).
- If nothing matches (e.g. off-schedule study session), it lists all classes
  so you can pick one manually.
- It then asks whether to record from your **microphone** (in-person lecture)
  or **system audio** (online lecture, e.g. Zoom/Teams playing through your
  speakers).
- It also asks which **note formatting mode** to use this session:
  1. **Auto** (default) — Claude CLI/API when available, falls back to the
     local GPU model then the heuristic formatter
  2. **Local only** — local GPU model + heuristic only, *no network calls at
     all* (CLI/API are never contacted this session) — useful for privacy,
     exam review, or working fully offline
  3. **Heuristic only** — no LLM anywhere, fastest and fully deterministic
- Talk/listen normally. Live transcript prints to the console as it goes
  (color-coded — see above).
- Press **`s`** anytime (no Enter needed) to **save immediately** instead of
  waiting for the next autosave — useful right before a class ends, or if
  you just want to be sure something important is captured. This also resets
  the 5-minute autosave timer, so it doesn't immediately trigger another save
  right after. Doesn't wait for a full chunk to finish collecting first — it
  sends the audio so far to transcription, then saves once that text is ready.
- Press **Ctrl+C** to stop — you'll see a detailed, timestamped play-by-play
  of the shutdown sequence (stopping capture, transcribing any final buffered
  audio, diarizing if enabled, saving, condensing) rather than a silent pause,
  since some of these steps can take a while on a long/complex final segment.
  Pressing Ctrl+C again during shutdown doesn't cancel it; it prints "Already
  stopping" and keeps saving. Closing the window instead can lose the final notes
  (the audio backup is kept for `--resume`).
  Notes are formatted and appended to `notes/<CLASS_CODE>.md`. Long sessions
  also autosave every 5 minutes so nothing is lost if the app closes
  unexpectedly.

Useful flags (each skips its corresponding prompt):

```bash
python src/main.py --class "BIOL 1440"     # skip auto-detection, force a class
python src/main.py --source mic            # skip the audio-source prompt
python src/main.py --source system         # capture system audio (loopback)
python src/main.py --formatting auto       # skip the formatting-mode prompt
python src/main.py --formatting local      # local only - no CLI/API calls this session
python src/main.py --formatting heuristic  # heuristic only - no LLM anywhere
python src/main.py --chunk 30              # transcribe in 30s chunks for more frequent output (default: 120)
python src/main.py --list                  # show all classes from schedule.json
python src/main.py --list-sessions         # list recoverable session backups (see below)
python src/main.py --resume PATH           # recover a crashed session (see below)
python src/main.py --prune-backups         # clean up old state/ backups (see below)
```

## Recovering a crashed/interrupted session

If the app dies unexpectedly (not a clean Ctrl+C - a crash, a power loss),
the final formatting/condense step never runs, the persisted audio and transcript remain available:
the raw transcript and full audio are written incrementally throughout the
session (`state/*_raw.txt` and `state/*.wav`), not just at the end.

```bash
python src/main.py --list-sessions              # see what's recoverable
python src/main.py --resume "state/PSYC_1300_20260902_091439.wav"
```

Point `--resume` at either the `.wav` or `_raw.txt` backup (it finds the
matching pair automatically). If the audio backup exists, it's **re-transcribed
from scratch** (not just replayed from the raw log) so diarization can run on
it too - safe to do now since recording has already stopped, unlike during a
live session. It's then formatted, saved, and condensed exactly like a normal
final save, replacing the same session section written by earlier autosaves. Other sessions,
including recordings from the same date, are left untouched. Falls back to the raw transcript log alone (no diarization
possible) if only that backup survived.

## Full-session diarization at Ctrl+C

If diarization is set up, stopping a live recording re-diarizes the **entire**
session's audio (not just whatever's pending since the last autosave) and
replaces this session's whole contribution to the notes file with one clean,
fully speaker-labeled section — so Q&A exchanges get pulled out across the
*whole* lecture, not just the last few minutes before you stopped. This
reuses the transcript already produced live (no re-transcription needed,
unlike `--resume`) and just runs diarization fresh against the full WAV.

**The tradeoff**: this reformats the entire lecture transcript through the
CLI/API at shutdown instead of just the small tail chunk, so on a long
lecture that's a real wait. The detailed, timestamped shutdown messages exist
specifically so this doesn't look hung. Falls back automatically to the
normal tail-only save (fast, no diarization) if diarization isn't set up,
fails, or nothing was ever transcribed that session.

**Optimized**: this path uses `combine_proofread=True` - proofreading and
notes-formatting run as ONE CLI/API call instead of two sequential ones (the
normal per-chunk save still uses two, since that's where vocab-learning's
before/after diff comes from, and the extra round-trip barely matters on a
small chunk anyway). Measured **60% faster** on a real transcript (19.0s →
7.5s, two calls vs. one) with no quality loss.

Diarization itself was also checked for GPU under-utilization by testing
pyannote's `embedding_batch_size`/`segmentation_batch_size` above their
default of 32. First pass: `64` was noise-level faster (23.1s vs. 22.5s) and
`128` measured 17x slower (386.5s) - but the machine then crashed
(`CLOCK_WATCHDOG_TIMEOUT`, a hardware/driver-level BSOD, not an app bug) while
running that same test with the laptop poorly ventilated (in a bag). A
retest with proper airflow, deliberately skipping `128`, found `32` and `64`
statistically indistinguishable (~51-59s both, same audio) - noisier than
the first pass in absolute terms, but no batch-size effect either way. Net
conclusion: **batch size isn't a real lever here** - left at the library
default (32). The `128` result specifically should not be trusted as a clean
measurement given what happened during that run; it wasn't safe to retest.

## Pruning old backups

`state/*.wav` files are large (tens to ~170MB+ for a long lecture) and
accumulate with no expiry — they're only there so `--resume` can recover a
crashed session, so once you're confident a lecture's notes are solid, the
backup can go.

```bash
python src/main.py --prune-backups                    # dry run, default 30+ days old
python src/main.py --prune-backups --older-than 14     # dry run, custom threshold
python src/main.py --prune-backups --older-than 14 --confirm   # actually delete
```

Dry-run by default — lists exactly what would be deleted and the total space
freed; nothing is actually removed until you add `--confirm`.

## Where things live

- `schedule.json` — your class schedule (edit this each semester; see format
  in the file — day, start/end time in 24h, location, type).
- `vocab.json` — per-class vocabulary hints (Latin/technical terms) used to
  fuzzy-correct mis-transcriptions during proofreading/local formatting. Add your own terms per class code - or let it
  grow on its own: whenever the CLI/API proofreading pass fixes a mis-heard
  term (e.g. "amigdala" → "amygdala"), that correction is automatically
  detected and saved into `vocab.json` for that class, so the heuristic/local
  formatters catch the same term next time without needing an LLM at all.
  Deliberately conservative about what it learns (word-level, similarity-gated)
  to avoid picking up ordinary rewording as if it were a vocabulary term.
- `notes/<CLASS_CODE>.md` / `.docx` — the running notes file per class, kept
  in sync. New lectures are appended as dated sections, so each class builds
  one continuous notes doc across the semester.
- `state/*_raw.txt` — raw timestamped transcript backups per session.
- `state/*.wav` — full audio backup per session (also what diarization runs
  against, if enabled).

## Updating your schedule

Edit `schedule.json`. Each class has a `sessions` list; each session has
`day` (full weekday name), `start`/`end` (24h `HH:MM`), `location`, and `type`.
Add/remove classes or sessions as your schedule changes each term.

## Meaning search

The dashboard's search box has two modes. **Keyword search** needs the words you type to appear
in the line. **Meaning search** matches by meaning, so "why does sweating cool you down" finds
the evaporative-cooling notes even though they never use those words.

It runs offline on all-MiniLM-L6-v2 through the shared service in `C:\AI\Tools
pu-services`,
on the **Intel iGPU**. Your notes and transcripts are embedded once and cached, so the first
search after new lectures takes a while (~50s for a semester's worth) and later ones take a
second or two.

Measured on 6155 passages of real notes and transcripts (2026-09-18), all four devices returned
identical answers, so the device choice is only about speed and CPU load:

| | index | CPU time | per passage |
|---|---|---|---|
| Intel iGPU, batches of 32 | 35.2s | 11.1s | 5.7 ms |
| NPU, one at a time | 55.6s | 13.8s | 9.0 ms |
| CPU, dynamic shapes, batches of 16 | 28.4s | 153.6s | 4.6 ms |
| CPU, one at a time | 199.5s | 1022.3s | 32.4 ms |

Quality, on 32 paraphrased study questions written before any results were seen
(`tests/test_semantic_search.py`): the right section came first for 23 of 32, was in the top 5
for 30, and the top 10 for all 32. Keyword search found none of them, since it needs every word
to match; a generous word-overlap ranking got 14 first and 23 in the top 10.

```bash
python tools/semantic_search_benchmark.py --freeze-corpus   # refresh the test copy of your notes
python tools/semantic_search_benchmark.py                   # re-run the device comparison
```

The frozen copy of your notes that the test uses is not committed, for the same reason `notes/`
isn't. The benchmark skips the discrete GPU while a recording is live.

## Improving accuracy

The app is currently tuned for accuracy over live-update frequency:

- **Model**: Cohere Transcribe 03-2026, chosen by measurement. In a three-way test on
  real PSYC 1300 and BIOL 1440 lectures it disagreed least with the other two models
  on both clips, ran several times faster than Whisper large-v3 or Qwen3-ASR, and got
  technical terms right with no vocabulary prompt. It's the only model - the Whisper
  fallback and its tuning tools were removed after Cohere ran every lecture without
  failing (results kept in `state/model_ab/`). Cohere returns no word timings; each piece it decodes (up
  to 35s) is one timed segment, so search jumps to the right half-minute of a
  recording, not the exact word.
- **Cohere settings** (`COHERE_*` in `src/transcribe.py`): tuned by
  `tools/cohere_tuning.py` against human-verified TED-LIUM transcripts, not other
  models. The app hands Cohere 2-minute windows, which its own splitter cuts at the
  quietest moment near each 35s mark - versus fixed 20s cuts that land mid-word, WER
  went 2.88% → 2.44% on the tuning talks and 4.33% → 3.73% on held-out talks. Beam search (2/4/8), no dither, and full 32-bit precision were all
  within noise of greedy bf16 while running 2-8× slower, so the defaults stay. Even
  so it transcribes 2 minutes of audio in ~2-4 seconds.
- **Cohere failure guards** (`looks_degenerate` / `is_silent_audio` in
  `src/transcribe.py`): two failures seen live in CHEM 1450 (2026-09-14), both
  reproducible. A muted or dropped-out mic (pure digital silence) came back as
  invented sentences ("The world is a very important part of the world" ×6), so
  pieces with no sound get no text. Very quiet room audio (class working) made the
  decoder loop until its length cap ("the other one is the other one…", 770 words
  from 33s), so a piece that repeats a phrase 4+ times, or has more words than
  anyone can say in that time, is re-decoded in 10s sub-pieces and anything still
  looping is dropped. Pieces that pass both checks keep exactly the normal output:
  re-running that session changed only its 6 failed pieces out of 63. A repetition
  penalty was tried first and rejected because it also changed correct words. The
  diagnostics dashboard's **Transcript guard** tile counts both kinds for the session,
  and each time the guard acts it logs an event naming the minute of audio affected.
- **Frozen-audio watchdog** (`CaptureWatchdog` in `src/main.py`): after the laptop slept
  during HLTH 1320 (2026-09-16), the Windows audio call never returned and the recorder
  ran for four hours capturing nothing, with no warning. Now, if no audio arrives for 10
  seconds it warns (terminal and dashboard), keeps whatever audio was already captured,
  and reconnects the microphone - retrying every 30 seconds while it stays silent and
  saying when audio is flowing again. A gap of 45+ seconds between checks is reported as
  "the computer was asleep or unresponsive from X to Y", so a missing stretch of lecture
  is explained. Note that recording only prevents *idle* sleep: closing the lid or a dead
  battery still sleeps the laptop.
- **Mic signal-loss warning** (`DropoutMonitor` in `src/main.py`): a live mic never
  produces exact zeros, even in a silent room, so 3 seconds of exact zeros means the
  signal is gone (muted or disconnected) and nothing is being recorded. It warns in
  the terminal and dashboard within seconds, says whether Windows has the mic muted
  (a read-only check; it never unmutes mid-recording), and reports when the signal
  returns and for how long it was lost. The dashboard's **Mic signal** tile turns
  red while it's out.
- **Chunk size** (`--chunk`): defaults to Cohere's tuned 120s window. Accurate text arrives in 2-minute batches,
  without an additional preview pass. A window is only
  skipped as silent if every 20s slice of it is silent.
- **No chunk overlap**: Cohere has no word timings to drop repeated words with, so
  windows run back to back - exactly how it was tested - and its splitter cuts each
  window at quiet points.
- **Latin/technical vocabulary**: add terms to `vocab.json` under your class's code
  (or `"_global"` for terms that apply everywhere). The proofreading pass
  fuzzy-corrects toward them (Cohere has no prompt input to prime). Terms are no longer learned
  automatically - that filled the list with ordinary words and mis-corrections that
  steered transcription toward words nobody said - so add them by hand or through the
  corrections review.
- **Proofreading + fact flags**: whenever the CLI/API is available, transcripts
  get proofread and possible mis-transcription-driven factual oddities get
  flagged inline (`⚠️ verify`) rather than silently changed — always double
  check flagged lines against your own memory of the lecture.

## GPU acceleration

This machine has an RTX 5070 Ti. Cohere Transcribe needs it (it's too slow on CPU),
and speaker diarization uses it too if enabled. The CUDA build of PyTorch ships its
own CUDA/cuDNN libraries, so it's the only GPU install needed:
```bash
./venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Preflight prints the device the model loaded on (`cuda/bfloat16`).

## Local GPU note formatting

Set up as the third formatting tier (`src/gpu_formatter.py`) — kicks in
automatically when neither the Claude CLI nor API is available, before
falling back further to the pure heuristic formatter. Runs a local LLM
(**Qwen2.5-3B-Instruct**, GGUF Q8_0) via **llama.cpp's `llama-server`**, fully
offline, actually reaching the discrete GPU (RTX 5070 Ti).

This reuses the already-installed llama.cpp build at
`C:/AI/Tools/llama-native/bin/llama-server.exe` **read-only, as a completely
separate process on its own port (8090)** — it does not touch, reconfigure,
or share anything with any other personal llama.cpp/model setup on this
machine. The model file lives in this project's own `state/llama_model/`,
never in a shared models folder. The app starts its own server automatically
on first use each run and shuts it down on exit (via `atexit`), so it doesn't
sit in the background holding ~3.6GB of VRAM between lecture sessions.

Setup (one-time, ~3.6GB download into this project only):
```python
from huggingface_hub import hf_hub_download
hf_hub_download("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q8_0.gguf",
                 local_dir="state/llama_model")
```
(If the Hugging Face download fails with a Xet/CDN error, retry with
`HF_HUB_DISABLE_XET=1` set - a more reliable plain-HTTP fallback.)

**Why this backend, not OpenVINO/NPU:** an NPU-based tier (Phi-3.5-mini via
OpenVINO GenAI) was built and evaluated first. It was rolled back after two
real reliability failures on the actual pipeline (not hand-picked test
prompts): default (greedy) decoding produced degenerate repetition - 9
near-duplicate paraphrased bullets for one simple two-sentence transcript -
and adding a `repetition_penalty` to fix that instead caused incoherent
rambling that invented content never in the transcript. A "GPU" comparison
via OpenVINO was also tried, but OpenVINO's GPU plugin only targets Intel
graphics (oneAPI/Level Zero) - it silently ran on the integrated GPU, never
the RTX 5070 Ti, and was no faster than the NPU. Switching to llama.cpp (which
does reach NVIDIA GPUs via CUDA) fixed all three problems at once: **~90-110
tok/s** vs. NPU's effective ~1-2 tok/s, no repetition (llama.cpp's sampling
defaults + explicit `repeat_penalty`/`temperature` tuning), and no
hallucination in per-chunk formatting testing.

**Why the Ctrl+C condense pass doesn't use this tier:** per-chunk formatting
tested reliably, but the harder multi-section merge/dedup task didn't -
across repeated test runs, it would inconsistently drop one genuinely
distinct bullet (which one varied by generation parameters) while satisfying
the notes' other instructions. That's a real content-loss risk that matters
more for condensing (which rewrites/replaces existing notes) than for
per-chunk formatting (which only appends), so `condense_session()` stays
CLI/API-only - see the comment in `src/notes.py` for the full reasoning.

Notes:
- Cold start (server spawn + model load) takes a few seconds; the app reuses
  the same server process for the rest of that run.
- Still noticeably less capable than Claude (3B params vs. a frontier model) -
  it inconsistently catches mis-transcribed vocabulary, so the same
  `vocab.json` glossary fix-up the heuristic formatter uses is applied to its
  output too.
- If the llama.cpp binary or the model file isn't found, this tier is
  silently skipped and the app falls straight to the heuristic formatter -
  nothing else breaks.
- VRAM check: `nvidia-smi --query-gpu=memory.used --format=csv` should show
  ~0MiB before a run and ~3.6GB while `llama-server.exe` is running for this
  app; it should return to ~0MiB after the app exits.

## Running unattended (minimized / long sessions)

Several things address transcription silently stopping or losing audio when
the app is left running in the background for a while - two of these were
found and fixed from real, live failures during an actual lecture, not just
theoretical hardening:

- **Threaded audio capture** (`src/capture.py`) - the biggest one. Audio
  capture runs on its own dedicated background thread, continuously draining
  the microphone/system-audio buffer into an in-memory queue, completely
  decoupled from transcription and saving. This replaced an earlier
  single-threaded design where any slow step in the main loop (a stuck API
  call, and especially the diarization pass, which used to run every autosave)
  blocked the next audio read for however long that step took - and WASAPI's
  hardware capture buffer is small enough (a fraction of a second) that this
  silently **dropped** audio rather than just delaying it. Confirmed live: a
  slow diarization pass caused a real ~2 minute gap of lost lecture audio.
  With capture on its own thread, however long processing takes, it only adds
  latency to when segments show up in the live view - audio itself can no
  longer be silently lost this way.
- **Diarization only runs once, at the very end** (on Ctrl+C), never during
  autosaves. It was originally run on every autosave to keep Q&A labels
  reasonably fresh, but pyannote isn't a real-time/incremental process anyway
  (it needs a complete clip to compute speaker segments), so there was no
  actual live benefit being traded away by moving it to the end - only
  autosaves being pointlessly slow. This was the direct cause of the ~2 minute
  gap mentioned above, and is fixed independently of (in addition to) the
  threaded-capture change.
- **Sleep prevention** (`src/keep_awake.py`): blocks *system* sleep for the
  duration of a recording session (released automatically on Ctrl+C or exit) -
  without this, an idle timeout can suspend the whole process, not just dim
  the screen. Deliberately does NOT force the display to stay on - that would
  waste real battery for a 50+ minute lecture for no benefit, since the app
  doesn't need the screen on to keep recording in the background.
- **Silence detection** (`audio.is_silent`, used in `RollingTranscriber`): if
  a whole window is at/near total silence, it's skipped before ever reaching the
  model; silent pieces inside a window are dropped by the Cohere guard above.
  Feeding a model silence is a known way to get it to hallucinate repeated
  punctuation/filler (`...`, `you`) or whole invented sentences instead of just
  emitting nothing. If silence continues for about 45 seconds, you'll get a
  one-time console and dashboard warning suggesting you check
  whether your mic is muted/disconnected (or, on system audio, whether
  anything's actually playing) - the app keeps running either way, but this
  flags a real audio-source problem instead of silently producing garbage.
- **Repetition collapse** (`transcribe._collapse_repeated_segments`): on
  ambiguous/overlapping audio (several people answering quietly at once,
  seen live in an actual lecture), a model can get stuck emitting the same
  short segment over and over as separate consecutive segments - caught and
  capped, since per-segment checks only look within one segment's text and
  don't catch repetition spread across many.
- The capture thread also **auto-recovers from audio-device errors** (a
  dropout after a resume, a USB mic hiccup): it logs the error (surfaced in
  the console) and reopens the recorder instead of capture dying silently.

## Notes on system audio

System-audio capture only picks up what plays through your speakers, so it
works for streamed/online lectures but not for playing back someone else's
copyrighted recording without permission — use it for your own classes.

# Lecture Notes App

Auto-detects which class you're in (from `schedule.json`, based on the day/time),
records + transcribes the lecture fully locally with Whisper, and turns the
transcript into clean notes appended to that class's ongoing notes file.

Note formatting is tried in this order, each falling back to the next if unavailable:
1. **Claude Code CLI** (`claude -p`) — uses your logged-in Pro/Max subscription, no per-token billing
2. **Claude API** (`ANTHROPIC_API_KEY`) — only used if the CLI isn't installed/logged in
3. **NPU local model** (Qwen2.5-1.5B on the Intel NPU via OpenVINO) — real LLM formatting, fully offline, used if neither the CLI nor API is available
4. **Heuristic local formatter** — regex-based topic/bullet formatting + glossary correction, last resort if the NPU model isn't set up either

Also:
- Whenever the CLI/API is used, a **proofreading pass** runs first — fixes
  spelling/grammar/mis-heard technical terms without changing what was actually
  said, and flags (never silently "corrects") any statement that looks like a
  transcription artifact producing something factually odd.
- **When you stop recording (Ctrl+C)**, if the CLI is available it re-reviews
  everything saved during that session (across any autosaves) in one pass —
  merging duplicate/repeated sections from multiple autosaves into one clean
  section, fixing formatting bugs, and cleaning up anything that still looks
  like unformatted raw transcript.
- Output is written as both **Markdown** (`notes/<CODE>.md`) and **Word** (`notes/<CODE>.docx`),
  kept in sync, appended lecture by lecture.
- If speaker diarization is set up (see below), **Q&A exchanges get their own section**,
  separated from the main lecture content.
- The live console view is color-coded: cyan timestamps, white transcript text,
  green for successful saves, yellow/red for fallback or error states, magenta
  for autosave markers.

## One-time setup

```bash
cd C:/AI/lecture-notes
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

The first time you transcribe, `faster-whisper` downloads the model (~150MB
for the default `base.en`) and caches it — after that it runs fully offline.

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

Then:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the terms on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. `setx HUGGINGFACE_TOKEN "hf_..."` (restart your terminal after)

This is a heavy install (~2GB, mostly PyTorch) and diarization adds noticeable
CPU time per save (runs once each time notes are saved, on just the newly
recorded audio since the last save). If it's not installed or the token isn't
set, the app runs exactly as before — no speaker labels, nothing breaks.

## Running it

```bash
./venv/Scripts/python.exe src/main.py
```

- It checks `schedule.json` against the current day/time and tells you which
  class it thinks you're in (with a 10-minute grace window before/after, so
  starting the app slightly early or late still picks the right class).
- If nothing matches (e.g. off-schedule study session), it lists all classes
  so you can pick one manually.
- It then asks whether to record from your **microphone** (in-person lecture)
  or **system audio** (online lecture, e.g. Zoom/Teams playing through your
  speakers).
- It also asks which **note formatting mode** to use this session:
  1. **Auto** (default) — Claude CLI/API when available, falls back to the NPU
     model then the heuristic formatter
  2. **Local only** — NPU model + heuristic only, *no network calls at all*
     (CLI/API are never contacted this session) — useful for privacy, exam
     review, or working fully offline
  3. **Heuristic only** — no LLM anywhere, fastest and fully deterministic
- Talk/listen normally. Live transcript prints to the console as it goes
  (color-coded — see above).
- Press **Ctrl+C** to stop — notes are formatted and appended to
  `notes/<CLASS_CODE>.md`. Long sessions also autosave every 5 minutes so
  nothing is lost if the app closes unexpectedly.

Useful flags (each skips its corresponding prompt):

```bash
python src/main.py --class "BIOL 1440"     # skip auto-detection, force a class
python src/main.py --source mic            # skip the audio-source prompt
python src/main.py --source system         # capture system audio (loopback)
python src/main.py --formatting auto       # skip the formatting-mode prompt
python src/main.py --formatting local      # local only - no CLI/API calls this session
python src/main.py --formatting heuristic  # heuristic only - no LLM anywhere
python src/main.py --chunk 8               # transcribe in 8s chunks for more frequent output (default 15)
python src/main.py --list                  # show all classes from schedule.json
```

## Where things live

- `schedule.json` — your class schedule (edit this each semester; see format
  in the file — day, start/end time in 24h, location, type).
- `vocab.json` — per-class vocabulary hints (Latin/technical terms) used both
  to bias Whisper's recognition and to fuzzy-correct mis-transcriptions during
  proofreading/local formatting. Add your own terms per class code.
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

## Improving accuracy

The app is currently tuned for accuracy over live-update frequency:

- **Model size**: `"large-v3"` (the most accurate Whisper model) in
  `src/transcribe.py` - practical because of the GPU (see below). If you ever
  run this CPU-only, drop it to `"small.en"` or `"medium.en"`, or `large-v3`
  will be painfully slow.
- **Chunk size** (`--chunk`, default 15s): larger chunks give Whisper more
  context per call, which improves accuracy. Live console updates arrive less
  often as a result - lower this if you want faster feedback at some accuracy
  cost.
- **Chunk overlap**: consecutive chunks overlap by 1.5s (`RollingTranscriber`
  in `src/main.py`) so a word split across a chunk boundary doesn't get
  clipped/mis-heard. The overlapping portion's already-emitted text is
  automatically deduplicated.
- **Rolling context prompt**: each chunk is transcribed with the tail of the
  previous chunk's text fed back in as Whisper's `initial_prompt` (alongside
  the vocab hints), so it has continuity across chunks instead of starting
  cold every time.
- **Beam size**: `BEAM_SIZE = 8` in `src/transcribe.py` (Whisper's default is
  5) - a wider decoding search for slightly better accuracy, cheap given the
  GPU headroom on this machine.
- **Latin/technical vocabulary**: add terms to `vocab.json` under your class's
  code (or `"_global"` for terms that apply everywhere). These both prime
  Whisper's recognition and get fuzzy-corrected during proofreading.
- **Proofreading + fact flags**: whenever the CLI/API is available, transcripts
  get proofread and possible mis-transcription-driven factual oddities get
  flagged inline (`⚠️ verify`) rather than silently changed — always double
  check flagged lines against your own memory of the lecture.

## GPU acceleration

This machine has an RTX 5070 Ti, so both Whisper transcription and (if
enabled) speaker diarization are already set up to use it automatically -
`transcribe.get_model()` uses `device="auto"`, which picks CUDA when available
and falls back to CPU otherwise; no code changes needed. Confirmed working:
transcribing an 8s audio chunk takes ~0.3s on GPU (vs. several seconds on CPU).

What's installed for this:
```bash
# CUDA-enabled PyTorch (also used by diarization if you set that up)
./venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# NVIDIA runtime libs faster-whisper/ctranslate2 needs for GPU execution
./venv/Scripts/python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If you ever reinstall from `requirements.txt` on a machine without a
CUDA-capable GPU, skip those two commands - the app still runs fine on CPU,
just slower. Check what Whisper picked with the "Whisper running on: ..."
line printed at startup (`cuda/float16` = GPU, `cpu/int8` = CPU).

## NPU-accelerated local formatting

This machine has an Intel NPU (**Intel(R) AI Boost**), already set up as the
third formatting tier (`src/npu_formatter.py`) — kicks in automatically when
neither the Claude CLI nor API is available, before falling back further to
the pure heuristic formatter. It runs a small local LLM (Qwen2.5-1.5B-Instruct,
INT4-quantized) via OpenVINO GenAI, so even fully offline you get real
LLM-quality note formatting instead of just regex bullet-splitting - confirmed
working: ~5s to format a lecture chunk, running entirely on the NPU (not
competing with the GPU that's busy transcribing).

What's installed for this:
```bash
./venv/Scripts/python.exe -m pip install openvino openvino-tokenizers openvino-genai
```
Model (downloaded once, ~900MB, cached in `state/npu_model/`):
```python
from huggingface_hub import snapshot_download
snapshot_download("OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", local_dir="state/npu_model")
```

Notes:
- First load per run compiles the model for the NPU (~50s); generation itself
  is fast (~5s per chunk) after that.
- It's noticeably less capable than Claude (1.5B params vs. a frontier model) -
  expect occasional rough formatting or a missed correction. The same
  `vocab.json` glossary fix-up the heuristic formatter uses is also applied to
  its output as a safety net.
- If the packages aren't installed or the model isn't downloaded, this tier is
  silently skipped and the app falls straight to the heuristic formatter -
  nothing else breaks.
- Machines without an Intel NPU (`Get-PnpDevice | Where FriendlyName -match
  "AI Boost"` to check) should skip this entirely.

## Notes on system audio

System-audio capture only picks up what plays through your speakers, so it
works for streamed/online lectures but not for playing back someone else's
copyrighted recording without permission — use it for your own classes.

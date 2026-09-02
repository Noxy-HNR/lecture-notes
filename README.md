# Lecture Notes App

Auto-detects which class you're in (from `schedule.json`, based on the day/time),
records + transcribes the lecture fully locally with Whisper, and turns the
transcript into clean notes appended to that class's ongoing notes file.

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
  space, and confirms the Whisper model loads - so a broken mic or a dead GPU
  shows up now, not silently mid-lecture. A genuinely broken audio device or
  Whisper model stops the app here rather than starting a doomed session;
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
  saves whatever's been transcribed so far within about half a second.
- Press **Ctrl+C** to stop — you'll see a detailed, timestamped play-by-play
  of the shutdown sequence (stopping capture, transcribing any final buffered
  audio, diarizing if enabled, saving, condensing) rather than a silent pause,
  since some of these steps can take a while on a long/complex final segment.
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
python src/main.py --chunk 8               # transcribe in 8s chunks for more frequent output (default 15)
python src/main.py --list                  # show all classes from schedule.json
python src/main.py --list-sessions         # list recoverable session backups (see below)
python src/main.py --resume PATH           # recover a crashed session (see below)
python src/main.py --prune-backups         # clean up old state/ backups (see below)
```

## Recovering a crashed/interrupted session

If the app dies unexpectedly (not a clean Ctrl+C - a crash, a power loss),
the final formatting/condense step never runs, but nothing is actually lost:
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
final save, including merging against any earlier autosaves already in the
notes file for that same date - previous lecture dates in the file are left
untouched. Falls back to the raw transcript log alone (no diarization
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
- `vocab.json` — per-class vocabulary hints (Latin/technical terms) used both
  to bias Whisper's recognition and to fuzzy-correct mis-transcriptions during
  proofreading/local formatting. Add your own terms per class code - or let it
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
  a chunk is at/near total silence, it's skipped before ever reaching Whisper.
  Feeding Whisper silence is a known way to get it to hallucinate repeated
  punctuation/filler (`...`, `you`) instead of just emitting nothing - if
  you've seen streams of dots in the output, this is why. If silence continues
  for 2+ minutes, you'll get a one-time console warning suggesting you check
  whether your mic is muted/disconnected (or, on system audio, whether
  anything's actually playing) - the app keeps running either way, but this
  flags a real audio-source problem instead of silently producing garbage.
- **Repetition collapse** (`transcribe._collapse_repeated_segments`): on
  ambiguous/overlapping audio (several people answering quietly at once,
  seen live in an actual lecture), Whisper can get stuck emitting the same
  short segment over and over as separate consecutive segments - caught and
  capped, since Whisper's own anti-hallucination heuristics only look within
  one segment's text and don't catch repetition spread across many.
- The capture thread also **auto-recovers from audio-device errors** (a
  dropout after a resume, a USB mic hiccup): it logs the error (surfaced in
  the console) and reopens the recorder instead of capture dying silently.

## Notes on system audio

System-audio capture only picks up what plays through your speakers, so it
works for streamed/online lectures but not for playing back someone else's
copyrighted recording without permission — use it for your own classes.

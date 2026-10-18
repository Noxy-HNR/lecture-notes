# Recorder priorities: implementation and verification

October 12, 2026. All six accepted priorities are implemented in the requested order.
The raw evidence is under `state/priority_implementation/`. A frozen copy of the
starting working tree is in that directory's `baseline/` folder. Existing dirty
dashboard, lesson, audio, export, and regression-test work was preserved.

## What runs by default

| Priority | Implementation | Default and control |
|---|---|---|
| 1. Bounded bulk loader | Pinned Nano Cohere runtime, original BF16 checkpoint, bounded loading and per-model decoder graphs | `auto`; reference fallback. `--asr-runtime hf` selects Transformers explicitly. |
| 2. Lean formatter invocation | Dedicated fidelity prompt, tools disabled, no session persistence, safe mode | Active in the existing Claude CLI route. Ordinary proofreading and note writing remain two passes. |
| 3. Audio persistence during preparation | The selected source starts writing the WAV while model preparation continues; validation observes that same capture thread | Active. Queued audio is backed by the WAV. |
| 4. Word alignment and speaker turns | Local Qwen forced alignment validates word timings against unchanged source text; diarization uses those boundaries | Automatic when installed. `--word-alignment off` skips alignment. Speaker diarization retains its existing `--diarize` switch. |
| 5. Compiled BF16/FP8 encoder | BF16 compiles single-piece batches; larger batches stay eager. FP8 is a separate experimental mode | Eager by default. `--encoder compile-bf16` or `--encoder compile-fp8` selects a candidate. |
| 6. Native captions and vocabulary retries | An isolated CPU worker supplies provisional captions. Selected glossary near misses receive two local second decodes and a strict acceptance check | Automatic when installed. `--captions off` and `--vocabulary-retry off` skip these paths. |

The alignment, captions, vocabulary, and compiler assets are prepared on this
machine. Recording, raw backups, recovery, autosaves, two-pass formatting,
glossary review, narration, lessons, exports, and the existing menus remain
available. The added word metadata lives in the segment ledger. Provisional
captions have their own Diagnostics display; final notes use the accepted
transcript.

## 1. Bounded loader and dependency preparation

The primary path imports Torch without importing the Transformers ASR stack.
The reference implementation is imported when selected or needed for recovery.
Nano is vendored at commit `f8e1cd94554a812f35d56dbdc42c461c681da774`, with its
Apache 2.0 license and a notice of local changes. This uses the existing model
weights and precision. Prompt tokens, piece splitting, batch limits, silence
guards, repetition repairs, and generated-token telemetry are retained.

Checkpoint metadata, shapes, offsets, dtypes, alignment, and payload size are
validated before GPU allocation. The loader uses two 64 MiB pinned host buffers
to stream the 4,131,608,480-byte payload into one GPU allocation. It requires
checkpoint-sized free VRAM plus 1 GiB. The bulk path also checks available host
RAM for staging plus headroom. Failed bulk resources are released before the
tensor-at-a-time `pread` fallback starts. Runtime construction failure can use
the reference Transformers backend.

Decoder graphs belong to the model instance. The cache holds at most two entries
and 384 MiB of estimated persistent buffers, evicting before allocating a
replacement. That budget covers graph buffers; allocator pools and workspaces
consume additional VRAM. Graph execution failure clears the cache and retries
using the eager decoder.

### Fresh-process startup measurement

Three serial runs used Nano / reference / Nano order with the same short clip.
Each process was new; filesystem caches were warm.

| Measurement | Reference HF | Nano run 1 | Nano run 2 |
|---|---:|---:|---:|
| Dependencies prepared | 9.85 s | 2.52 s | 2.25 s |
| Model ready, from process start | 19.79 s | 8.59 s | 8.26 s |
| First text, from process start | 34.51 s | 21.71 s | 16.14 s |
| Checkpoint bulk transfer | — | 2.88 s | 2.00 s |

Mean Nano model readiness was **57.4% shorter**. First-text time was 37.1–53.2%
shorter. All three returned identical segment text and timestamps. These measure
model preparation and first inference, not cold boot or total interactive
button-to-microphone latency. First-shape CUDA graph capture still costs time.

Evidence: `startup_compare_0_nano.json`, `startup_compare_1_hf.json`,
`startup_compare_2_nano.json`, and `runtime_nano.json`.

### Human-reference fidelity

| Evaluation | Items | Audio | Reference words | Reference and integrated WER |
|---|---:|---:|---:|---:|
| Held-out TED-LIUM talks | 4 | 76.28 min | 14,506 | 3.729% |
| AMI meeting clips | 200 | 10.74 min | 1,913 | 11.291% |

All **204 normalized word sequences, per-item error rates, and piece lengths**
matched the saved reference evaluation. Raw strings matched in 197 cases;
seven differed in punctuation or capitalization. Scoring uses the same Whisper
English normalizer and corpus WER method. Two saved 120-second lecture windows
also retained exact segment text and timestamps. Those windows have no human
transcript and establish compatibility rather than a new accuracy score.

The integrated evaluation processed the talks in 31.84 seconds and the meeting
clips in 33.73 seconds. These are serial evaluation timings, not a controlled
ABBA comparison with HF. Peak CUDA allocation was 4.91 GiB, reserved memory was
5.49 GiB, and process-tree RSS peaked at 2.81 GiB. The graph cache peaked at two
entries and approximately 306 MiB of persistent buffers.

Evidence: `production_validation.json`, compared with the existing
`state/efficiency_experiment/hf_validation.json`.

## 2. Lean formatter invocation

The CLI receives a dedicated source-fidelity system prompt, `--tools ""`,
`--no-session-persistence`, and `--safe-mode`. The prompt preserves quantities,
units, negation, scope, Q&A, quiz cues, and uncertainty markers. Existing timeout
and provider fallback behavior remains in place. Ordinary saves retain separate
proofreading and note-writing passes.

Bare mode failed with `Not logged in` on this subscription setup. A subsequent
authenticated synthetic request succeeded in 7.41 seconds. Safe mode retains
the authenticated route while excluding coding customization. Explicit bare
mode remains available through `NOTE_RECORDER_CLAUDE_BARE=1` for a separately
configured environment. Flag behavior is documented in the
[official Claude CLI reference](https://code.claude.com/docs/en/cli-reference).

A complete new live formatting comparison could not finish because the Claude
account reached its weekly quota. Both safe and ordinary CLI invocations reported
that limit. Argument wiring and the existing proofreading/notes regressions pass;
this report does not claim a measured new formatter speedup or a full semantic
acceptance run. Only synthetic fixtures were used in the cloud probes.

## 3. Capture during preparation and disk-backed queues

After class and source selection, microphone preparation runs, the session WAV
opens, and the dedicated capture thread starts. Model checks and the remaining
formatting prompt can continue while the selected source is being preserved.
Audio validation observes the first three persisted seconds from that capture
thread, avoiding another recorder open and preserving the transcription queue.

Each block is written and flushed before it is queued. Queue entries hold WAV
offsets and frame counts; a delayed model or user prompt therefore accumulates
small references instead of audio arrays. The startup observer retains only
three seconds. Transcription reads the same durable frames in order.

Startup rejection and Ctrl+C close capture, release the sleep-prevention request,
close the WAV, and print the recovery command. Tests verify that every captured
frame remains readable after both failure paths. Device reconnect and normal
shutdown behavior retain their existing recovery logic. No new live microphone
session was opened for this implementation's verification.

## 4. Word timing and speaker attribution

The local Qwen forced aligner runs after transcription and autosaves are drained.
The Cohere model is released first. Audio is read one transcript piece at a time.
Predictions must match the source word sequence and count; times must be finite,
ordered, non-overlapping, and inside the piece. Exact source spans preserve
punctuation and capitalization. Invalid alignment keeps the original segment.

Word timings let speaker attribution split a long piece at a speaker boundary
while retaining questions and exact text. Pieces without word metadata retain
the existing overlap-based attribution. Real probes aligned three public clips
without changing their transcript text. This verifies timing integration; it
does not demonstrate a measured classroom diarization-accuracy gain.

The model is pinned to `Qwen/Qwen3-ForcedAligner-0.6B-hf` revision
`c07281df297b9905d24a508279258cccf987a064`. It requires 3 GiB available RAM,
loads locally, and unloads after alignment. The APIs are documented in the
[Transformers Qwen3 ASR reference](https://huggingface.co/docs/transformers/model_doc/qwen3_asr).
Evidence: `alignment_real.json` and word-turn/failure regression tests.

## 5. Compiled encoder: measured dispatch and experimental FP8

The optional compiler matches the installed CUDA Torch 2.11 build with
`triton-windows==3.6.0.post26`. Compilation runs with one compile worker and
project-owned caches. The supported pairing is documented by
[Triton for Windows](https://github.com/triton-lang/triton-windows).

A fresh BF16 cache took 53.54 seconds on its first short decode; a cached
first decode still took approximately 7.9 seconds in the final comparison.
Therefore compilation remains explicit. Compiler execution failure restores
all layers together and continues eagerly without repeated compiler attempts.

The larger-batch comparison showed that compilation could slow dense windows.
The final BF16 dispatch compiles batches of one and uses eager execution for
larger batches. Alternating eager / candidate / candidate / eager runs measured:

| Warm case | Eager mean | BF16 dispatch mean | Change |
|---|---:|---:|---:|
| Short speech | 198 ms | 144 ms | 27.4% shorter |
| Quiet 120-second window with one speech piece | 289 ms | 241 ms | 16.6% shorter |
| Dense four-piece window | 582 ms | 570 ms | Same eager path; approximately 2% run variation |

All 36 final comparison outputs retained exact segment text and timestamps.
An earlier compile-all BF16 evaluation retained TED-LIUM corpus WER and improved
AMI by one reference word, with 203/204 identical item error rates. BF16
compilation can change floating-point rounding; exact output identity is an
observed result on the stated comparison cases, not a general guarantee.

FP8 keeps reference BF16 weights for recovery and quantizes feed-forward matrix
multiplications. It checks additional weight memory plus 1 GiB free VRAM before
allocating. It showed no consistent speed advantage: the short ABBA case was
23.4% slower, and the quiet case changed recognized text. In the 204-item set,
four item error rates improved and four worsened. TED-LIUM WER was 3.757%, AMI
WER was 11.134%, and peak CUDA allocation rose to 6.08 GiB. FP8 remains an
explicit experimental option.

Evidence: `encoder_bf16_dispatch.json`, `encoder_bf16_abba.json`,
`encoder_fp8_abba.json`, `compiled_bf16_validation.json`, and
`compiled_fp8_validation.json`.

## 6. Native streaming captions and selective vocabulary retries

### Provisional captions

The isolated native worker uses the pinned
[NVIDIA NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) v0.2.0 Windows
release and the Nemotron streaming English Q8 model on CPU. Model revision:
`ebe59e5a817142986528bbbee5dba8db7b38ed50`. The DLL and its model live under
project-owned `state/nemo_speech/`; the release archive checksum is verified.

Capture writes the WAV before submitting optional preview audio. Submission is
nonblocking. The queue holds at most 32 one-second packets. A backlog clears the
preview queue and resets native streaming state across the dropped interval.
The worker has startup and shutdown timeouts, a RAM watchdog, and an owned
Windows kill-on-close job. A startup exit reports failure in Diagnostics.

Real public clips produced useful interim captions. The actual parent/worker
pipe, callbacks, glossary context, completion, and cleanup all passed. Caption
process-tree RSS peaked at approximately 1.02 GiB. The displayed preview holds
four completed utterances plus current text, capped at 2,000 characters. The
final Cohere transcript can revise the preview.

Evidence: `native_real_2.json`, `caption_pipeline.json`, and
`caption_pipeline_vocabulary.json`. The initial `native_real.json` preserves
the rejected streaming-geometry attempt for provenance.

### Vocabulary second passes

Only an eligible near miss to a supplied single-word glossary term is retried.
The local Qwen ASR model decodes the saved audio once without context and once
with glossary context. Acceptance requires matching normalized outputs, equal
word counts, exactly the selected glossary-word replacement, and every other
word unchanged. Numbers, negation, scope, and additional wording changes cause
rejection. Source punctuation is retained; the original and both candidates
are recorded in the ledger. Accepted changes refresh the full session's notes
through ordinary formatting.

The shutdown pass considers at most eight pieces, checks a 60-second soft budget
between pieces, caps each decode at 512 tokens, requires 3 GiB available RAM, and
unloads after use. The time limit is checked between requests and does not
interrupt a decode already running. Model revision:
`Qwen/Qwen3-ASR-0.6B-hf` at `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`.

A real paired decode succeeded, but Qwen introduced a phrase unsupported by
the human reference. The acceptance rule rejects that broader replacement.
A synthetic one-word near miss derived from the paired output was accepted.
This verifies model wiring and the conservative gate; it does not establish
a measured real-speech vocabulary accuracy improvement.

Evidence: `vocabulary_real.json` and quantity/negation/scope rejection tests.

## Python, verification, and reproduction

Python remains **3.13.15**. The installed `kokoro-onnx` declares
`Requires-Python: >=3.10,<3.14`, so this environment retains compatibility with
the narration stack. Startup savings above came from the runtime and dependency
path. No newer-Python speedup is claimed.

The full project suite passed **662 tests in 75.24 seconds**. Final failure-path
review added four checks and improved the caption-startup error report; the
targeted encoder, loader, and caption set then passed **13 tests in 1.38 seconds**.
The full-suite count predates those four added tests and that small error-path
change. Python sources parsed successfully and `git diff --check` passed.

Real GPU/native probes ran as bounded offline children, with RAM/VRAM admission,
hard timeouts, and owned-process cleanup. Initial low-RAM attempts stopped before
model loading; later probes completed after RAM was freed. Existing recorder
state and performance files were isolated from the tests and benchmarks.

Useful controls:

```powershell
# Normal integrated runtime; prepared optional assets are detected automatically.
.\venv\Scripts\python.exe -B src/main.py

# Opt into the measured BF16 dispatch, accepting cache warmup cost.
.\venv\Scripts\python.exe -B src/main.py --encoder compile-bf16

# Explicit reference runtime and controls for the additional processing.
.\venv\Scripts\python.exe -B src/main.py --asr-runtime hf --captions off --word-alignment off --vocabulary-retry off

# Tests use isolated live-state paths and disable real optional models.
.\venv\Scripts\python.exe -B -m pytest -q
```

`tools/prepare_priority_assets.py` prepares the pinned `aligner`, `native`, and
`vocabulary` assets. Optional compiler requirements are in
`requirements-encoder.txt`. The startup, runtime, encoder, alignment, caption,
and vocabulary probe tools require distinct output paths under `state/`.
Run model probes after closing the recorder normally. The preparation tools
and offline probes are separate from ordinary recorder startup.

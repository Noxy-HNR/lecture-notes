# Recorder efficiency results — October 11, 2026

## Changes in the recorder

1. Quiet pieces are excluded **before generation**, using the existing RMS policy and exact processor split. The original feature tensors and padding remain intact. Empty results restore their original positions, so timestamps do not shift. Unexpected split/decoder counts keep the existing fallback behavior. Silence and repetition repair guards remain available.
2. WASAPI discontinuities are counted per capture thread, retained across device reconnects, and exposed through telemetry/dashboard events. Only the native discontinuity warning is routed; other warnings retain normal behavior. Raw blocks are still persisted before inference. Mono remains the capture default.
3. Both notes prompts now explicitly preserve quantity scope and source qualifiers, and reject invented measurement/setup assertions. The combined prompt explicitly retains existing uncertainty markers. Ordinary saves retain the two-pass default; the existing final diarized save can still use its combined path. These prompt constraints are instructions to the formatter, not a deterministic content validator.

VAD and the Nano decoder remain **offline experiments**, as specified by the evaluation priorities. They are not selected by the live recorder. No recording, recovery, menu, diarization, glossary review, or export feature was removed. Existing unrelated dashboard/lesson changes were preserved. No requirements or installed packages were changed.

## 1. Silence filtering

The saved PSYC window beginning at 2160 seconds contains four 30-second pieces; three are quiet under the existing guard. An ABBA comparison used the same loaded weights and processor.

- Baseline mean: **1.290 s**.
- Filtered mean: **0.789 s** — **38.8% less elapsed time**.
- Generated tokens: **349 → 67**.
- Speech segments, their text, and timestamps were exactly equal in all four runs.
- A continuous-speech BIOL window and all nine short reference clips also retained exact segment parity.

This gain applies to windows containing entirely quiet pieces. Continuously spoken windows have little avoidable decoding work. Preprocessing still runs over the original window to preserve feature parity.

Raw report: `state/efficiency_experiment/silence_abba.json`.

## 2. Live capture validation

The default Intel Smart Sound microphone array exposes two channels. Four sequential six-second captures used mono/stereo/stereo/mono order at 16 kHz with the existing 4096-frame block size. All returned the expected 96,000 frames, finite samples, zero clipping, and zero reported discontinuities. Stereo channel correlations were approximately 0.81 and 0.76; downmixing did not show strong cancellation.

Ambient sound varied between samples, especially the first mono capture. This experiment has no spoken human reference, so it cannot establish a speech-accuracy advantage or prove that the single-channel path is defective. Mono stays the default. Audio samples were held in memory; only statistics were saved.

Raw report: `state/efficiency_experiment/capture_abba.json`.

## 3. Conservative VAD

The CPU-only [Silero model](https://github.com/snakers4/silero-vad/tree/1e261b036686cd0017d500ee96acd1c4ba572a9d) was evaluated without trimming or concatenating speech. The candidate keeps a whole piece whenever any 32 ms frame reaches the declared speech threshold. It is more conservative than applying the usual utterance-duration filter. The local NumPy/ONNX state/context wrapper exactly matched upstream probabilities on all nine reference clips.

| Gate | Human-referenced speech clips rejected, out of 843 |
|---|---:|
| Existing RMS | 19 |
| VAD maximum probability ≥ 0.1 | 1 |
| VAD maximum probability ≥ 0.2 | 2 |
| VAD maximum probability ≥ 0.5 | 11 |

The single clip rejected at 0.1 said **“Yeah.”** It passed RMS. All 19 clips rejected by RMS were detected by VAD at 0.1. A hybrid that keeps ordinary RMS-passing speech and uses VAD to rescue low-energy speech would therefore retain all 843 labeled speech clips in this set. That is a classification result, not proof of full classroom accuracy or background-noise specificity.

On nine clips with synthetic 30 dB attenuation, normalized WER was **16.79% without gating**, **87.59% with RMS**, and **16.79% with VAD at 0.1**. RMS rejected eight of those clips; VAD rejected none. This is an amplitude stress test, not a recording of genuinely quiet classroom speech. At synthetic 5 dB noise SNR, ungated/RMS/VAD-0.1 WER were all 34.31%. Digital zero, white noise, and 60 Hz hum controls stayed below probability 0.03.

A VAD-only replacement still risks short answers. A rescue-only hybrid is the strongest next accuracy experiment; it needs real room-noise negatives before selecting a new default.

Raw reports: `vad_843.json`, `vad_wrapper_parity.json` under `state/efficiency_experiment/`.

## 4. Streamed weights and bounded CUDA graphs

The [Nano implementation](https://github.com/Deep-unlearning/nano-cohere-transcribe/tree/f8e1cd94554a812f35d56dbdc42c461c681da774) is pinned to `f8e1cd94554a812f35d56dbdc42c461c681da774`. The experiment adapter:

- Streams original safetensors tensors through CPU `pread` into the GPU, avoiding the large mapped-checkpoint host peak.
- Adds the recorder's leading prompt token, which upstream Nano omits. This removes a known source of word/punctuation drift in the earlier probe.
- Retains the 120-second windows, exact 35-second energy splits, batch limit four, 770-token cap, silence policy, repetition repairs, and piece ordering.
- Limits graphs to **two entries** and **384 MiB of persistent buffers**; evicts before allocating a replacement. CUDA allocator headroom and the process RAM watchdog bound additional graph-pool/model memory.

| Human reference dataset | Audio | Reference words | Current HF | Nano eager | Nano with graphs | WER, all three |
|---|---:|---:|---:|---:|---:|---:|
| Four held-out TED-LIUM talks | 76.28 min | 14,506 | 100.72 s | 56.96 s | 42.23 s | 3.729% |
| 200 AMI meeting clips | 10.74 min | 1,913 | 53.74 s | 44.12 s | 38.44 s | 11.291% |

All **204 normalized word sequences** matched exactly across the three backends. Seven HF/graph raw-text differences were punctuation/capitalization. Piece lengths matched. Normalization uses the existing Whisper EnglishTextNormalizer and corpus-level WER scoring.

End-to-end graph candidate speed relative to current HF was **2.39×** on talks and **1.40×** on meeting clips. Comparing Nano graphs with Nano eager isolates the graph contribution: **25.9% less lecture processing time**, and **12.9% less meeting processing time**. The remaining gain comes from the candidate implementation/frontend, so the whole 2.4× should not be attributed to graphs alone.

The graph validation built 40 shapes and evicted 38. Peak cached persistent buffers were **306 MiB**, peak entries **2**, peak CUDA allocated memory **4.886 GiB**, and peak CUDA reserved memory **5.809 GiB**. The memory budget refers to persistent graph buffers; the separate CUDA/RAM limits also cover allocator pools and model workspaces.

Two actual saved lecture windows also matched **all segment text and timestamps exactly** after matching the prompt. This includes the BIOL window where the earlier unaligned candidate changed “feed” to “heat.” These lecture windows have no human transcript, so parity is a compatibility check, not an accuracy score.

Fresh-process readiness was 21.71 s for HF and 13.37 s for graphs; the first decode took 3.83 s and 4.65 s respectively. Filesystem caches were warm, runs were serial, and this was not an interactive-menu/preflight measurement. Graphs have a first-shape capture cost. Do not treat these observations as guaranteed cold-start savings.

The graph decoder is the strongest next production integration. The current adapter is experimental and does not yet supply the live recorder's generated-token telemetry; lifecycle/startup integration and that telemetry must be preserved when promoting it.

Raw reports: `hf_validation.json`, `nano_matched_validation.json`, `nano_eager_validation.json`, `lecture_parity.json` under `state/efficiency_experiment/`.

## 5. Combined proofreading and notes

The existing configured CLI model was used only with synthetic fixtures; no class transcript or real glossary was sent. The comparison never wrote actual notes. Call order alternated between fixtures.

| Fixture, final shared fidelity rules | Separate calls | Combined call |
|---|---:|---:|
| terms_numbers_qa_flags | 135.12 s | 14.85 s |
| negation_contrast_quiz | 28.69 s | 16.57 s |
| many_distinct_observations | 33.56 s | 13.89 s |

Numbers, all twelve station observations, Q&A, and uncertainty flags were reviewed against the fixture source. The initial combined prompt added “each” to group trial counts. Shared fidelity rules removed that error in the final examples, but did not eliminate all generative errors:

- Final combined station notes retained the comparison concept but omitted its explicit **quiz** cue.
- Final separate station notes invented a condensation setup and an explanation of the experiment that the source did not provide, under a background label.
- All twelve station counts and their timestamps survived both final routes.

Pattern checks are screening aids. One matcher initially crossed a newline and falsely flagged “Each trial lasted 90 seconds”; it was corrected, regression-tested, and the existing outputs were rescored offline. `formatter_review.json` records the rescoring. Original run reports are preserved.

Combined calls were faster in these runs, but latency varied substantially and three short synthetic fixtures do not establish full-lecture fidelity. Ordinary saves retain the existing two-pass default. The new fidelity instructions also apply to that path; they are not a guarantee against unsupported additions.

Raw reports: `formatter_existing.json`, `formatter_fidelity.json`, `formatter_final.json`, `formatter_review.json` under `state/efficiency_experiment/`.

## Reproduce and review

Use the commands in the README's **Recorder efficiency experiments** section. `tools/prepare_recorder_experiments.py` verifies/downloads hash-pinned public assets into ignored state and never installs packages. Original ASR weights and cached human-reference datasets are prerequisites.

The child runner refuses a live recorder, checks available system RAM, enforces a 1.2 GiB RAM floor and a hard timeout, checks CUDA headroom before weights load, and terminates only its own process tree. Future runs also abort if the recorder is opened while a probe is running. Progress and partial failure reports are retained. Performance stages are isolated so these experiments do not write live recorder telemetry.

Tests cover quiet-piece mapping/fail-open behavior, repetition repairs, thread-local native warning routing, capture persistence, reconnect counts, graph eviction/destruction, VAD state/context, matcher scope, and hard-timeout child cleanup, alongside the existing startup, shutdown, notes and recorder-launcher regressions. The focused command is recorded below; this is not a claim that the entire project suite ran.

**135 focused tests passed in 9.09 seconds.** Every real model/formatter probe completed without a watchdog stop. All benchmark model processes exited; the GPU returned to zero reported memory use. The existing live performance log remained at 11,886 rows.

```powershell
.\venv\Scripts\python.exe -B -m pytest tests/test_startup_preparation.py tests/test_transcription_backend.py tests/test_shutdown_controls.py tests/test_day_headings.py tests/test_regressions.py tests/test_recorder_launcher.py tests/test_capture_watchdog.py tests/test_capture_diagnostics.py tests/test_recorder_experiments.py tests/test_notes_flashcards_timeout.py -q
```

"""Meaning search: how good is it, and does the NPU earn its place?

Runs the notes page's "Meaning search" (all-MiniLM-L6-v2, the NPU service's exact embedding and
ranking code from Tools/npu-services/runtime.py) on each OpenVINO device, same weights:

  quality   32 paraphrased study questions (tests/fixtures/semantic_search_cases.json) against a
            frozen copy of the notes. A question scores if a top-k line is in a section that
            answers it. Baselines: the notes page's keyword search (every word must appear) and
            a generous word-overlap ranking.
  parity    cosine between each device's vectors and the CPU fp32 reference.
  speed     compile time (cold, then cached), indexing every passage of the live notes and
            transcripts, per-question latency, and CPU time the process burned doing it.

"service" configs reproduce the NPU service setup (one passage at a time, padded to 256 tokens,
which the NPU requires); "best" configs use what suits that hardware (dynamic shapes on CPU,
batches on the GPUs).

    python tools/semantic_search_benchmark.py                       # every config
    python tools/semantic_search_benchmark.py --configs npu cpu-best
    python tools/semantic_search_benchmark.py --freeze-corpus       # refresh the frozen notes copy
Results: state/eval/semantic_search_benchmark.json. Refuses the discrete GPU while a recording is
live (it would compete with Cohere), unless --force.
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "Tools" / "npu-services"))

import numpy as np

FIXTURES = ROOT / "tests" / "fixtures"
RESULTS = ROOT / "state" / "eval" / "semantic_search_benchmark.json"
STATUS_PATH = ROOT / "state" / "status.json"

# name: (label, device, static_shape, batch)
CONFIGS = {
    "npu":        ("NPU - service setup",            "NPU",   True,  1),
    "npu-batch":  ("NPU - batch 8",                  "NPU",   True,  8),
    "cpu":        ("CPU - service setup",            "CPU",   True,  1),
    "cpu-best":   ("CPU - dynamic shapes, batch 16", "CPU",   False, 16),
    "igpu":       ("Intel iGPU - service setup",     "GPU.0", True,  1),
    "igpu-batch": ("Intel iGPU - batch 32",          "GPU.0", True,  32),
    "dgpu":       ("RTX 5070 Ti - service setup",    "GPU.1", True,  1),
    "dgpu-batch": ("RTX 5070 Ti - batch 32",         "GPU.1", True,  32),
}


def recording_in_progress() -> bool:
    try:
        s = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return bool(s.get("active")) and (time.time() - s.get("updated_at", 0)) < 90
    except Exception:
        return False


def freeze_corpus():
    """Snapshot today's notes as the test corpus. Kept out of git (it's lecture content), so a
    fresh clone runs this once before the meaning-search quality test can run."""
    import search
    rows = [{"class_code": h.class_code, "location": h.location, "section": h.section, "text": h.text}
            for h in search.search_notes("*")]
    path = FIXTURES / "semantic_search_corpus.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {path}")


def load_cases():
    cases = json.loads((FIXTURES / "semantic_search_cases.json").read_text(encoding="utf-8"))
    corpus = json.loads((FIXTURES / "semantic_search_corpus.json").read_text(encoding="utf-8"))
    return cases, corpus


def live_rows():
    """What the notes page searches today: every notes line and transcript segment."""
    import search
    return [asdict(h) for h in search.search_notes("*") + search.search_transcripts("*")]


def evaluate(cases, corpus, embed, limit=30) -> dict:
    """Section-level hit@1/5/10 and mean reciprocal rank, using the service's rank()."""
    from runtime import rank
    ranks = []
    for case in cases:
        results = rank(embed(case["query"]), corpus, embed, limit=limit)
        wanted = {(case["class"], s) for s in case["sections"]}
        position = next((i + 1 for i, r in enumerate(results)
                         if (r["class_code"], r["section"]) in wanted), None)
        ranks.append(position)
    return scores(ranks)


def keyword_baseline(cases, corpus, limit=30) -> dict:
    """The notes page's keyword search (every word must appear), same scoring."""
    ranks = []
    for case in cases:
        terms = case["query"].lower().split()
        results = [r for r in corpus if all(t in r["text"].lower() for t in terms)][:limit]
        wanted = {(case["class"], s) for s in case["sections"]}
        ranks.append(next((i + 1 for i, r in enumerate(results)
                           if (r["class_code"], r["section"]) in wanted), None))
    return scores(ranks)


STOPWORDS = set("a an and are as at be by can do does for from how in is it its of on one or so than that "
                "the their them they this to up used what when where which who why with without you your".split())


def keyword_overlap_baseline(cases, corpus, limit=30) -> dict:
    """A generous keyword search: lines ranked by how many of the question's non-trivial words
    (or their first five letters, to catch plurals) they contain."""
    import re
    ranks = []
    for case in cases:
        stems = {w[:5] for w in re.findall(r"[a-z]+", case["query"].lower()) if w not in STOPWORDS}
        scored = []
        for r in corpus:
            line = {w[:5] for w in re.findall(r"[a-z]+", r["text"].lower())}
            if stems & line:
                scored.append((len(stems & line), r))
        results = [r for _, r in sorted(scored, key=lambda x: -x[0])][:limit]
        wanted = {(case["class"], s) for s in case["sections"]}
        ranks.append(next((i + 1 for i, r in enumerate(results)
                           if (r["class_code"], r["section"]) in wanted), None))
    return scores(ranks)


def scores(ranks) -> dict:
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    return {"hit@1": round(hit(1), 3), "hit@5": round(hit(5), 3), "hit@10": round(hit(10), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3), "ranks": ranks}


def passage_texts(rows):
    from runtime import passages
    seen = dict.fromkeys(t for r in rows for t in passages(r["text"]))
    return list(seen)


def embed_all(embedder, texts, sort_by_length):
    """Vectors for every text, plus wall and process-CPU seconds. Dynamic shapes are sorted by
    length so a batch doesn't pad short lines out to its longest one."""
    order = sorted(texts, key=len) if sort_by_length else texts
    wall, cpu = time.perf_counter(), time.process_time()
    vectors = embedder.embed_many(order)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return dict(zip(order, vectors)), wall, cpu


def run_config(name, core, texts, cases, corpus, reference=None) -> dict:
    from runtime import Embedder
    label, device, static, batch = CONFIGS[name]
    result = {"config": name, "label": label, "device": device, "static_shape": static, "batch": batch}
    print(f"\n== {label}")
    with tempfile.TemporaryDirectory() as cache:
        started = time.perf_counter()
        Embedder(core, device, static, batch, cache_dir=cache)
        result["compile_cold_s"] = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        embedder = Embedder(core, device, static, batch, cache_dir=cache)
        result["compile_cached_s"] = round(time.perf_counter() - started, 2)
    print(f"   compile {result['compile_cold_s']}s cold, {result['compile_cached_s']}s cached")

    embedder.embed_many(texts[:max(batch, 8)])  # first inferences allocate; don't time them
    vectors, wall, cpu = embed_all(embedder, texts, sort_by_length=not static)
    result.update(passages=len(texts), index_s=round(wall, 2), index_cpu_s=round(cpu, 2),
                  ms_per_passage=round(1000 * wall / len(texts), 2))
    print(f"   indexed {len(texts)} passages in {wall:.1f}s ({result['ms_per_passage']} ms each), "
          f"CPU time {cpu:.1f}s")

    query_ms = []
    single = embedder if batch == 1 else Embedder(core, device, static, 1)
    for case in cases:
        started = time.perf_counter()
        single(case["query"])
        query_ms.append(1000 * (time.perf_counter() - started))
    result["query_ms_median"] = round(statistics.median(query_ms), 2)

    lookup = lambda text: vectors[text] if text in vectors else single(text)
    result["quality"] = evaluate(cases, corpus, lookup)
    q = result["quality"]
    print(f"   question embed {result['query_ms_median']} ms | hit@1 {q['hit@1']}  hit@5 {q['hit@5']}  "
          f"hit@10 {q['hit@10']}  MRR {q['mrr']}")
    if reference is not None:
        cos = [float(vectors[t] @ reference[t]) for t in texts]
        result["parity_min_cos"], result["parity_mean_cos"] = round(min(cos), 5), round(float(np.mean(cos)), 5)
        print(f"   vs CPU fp32: min cosine {result['parity_min_cos']}, mean {result['parity_mean_cos']}")
    return result, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--force", action="store_true", help="run the discrete GPU during a recording")
    parser.add_argument("--freeze-corpus", action="store_true", help="refresh the frozen notes copy and exit")
    args = parser.parse_args()

    if args.freeze_corpus:
        return freeze_corpus()

    import openvino as ov
    core = ov.Core()
    configs = [c for c in args.configs if CONFIGS[c][1] in core.available_devices]
    if recording_in_progress() and not args.force:
        skipped = [c for c in configs if CONFIGS[c][1] == "GPU.1"]
        configs = [c for c in configs if c not in skipped]
        if skipped:
            print(f"A recording is live - skipping {', '.join(skipped)} (discrete GPU). Use --force to include.")

    cases, corpus = load_cases()
    rows = live_rows()
    texts = passage_texts(rows + corpus)
    print(f"{len(rows)} live rows -> {len(texts)} unique passages; {len(cases)} questions")

    baseline = {"keyword_page": keyword_baseline(cases, corpus), "keyword_overlap": keyword_overlap_baseline(cases, corpus)}
    for label, q in baseline.items():
        print(f"{label} baseline: hit@1 {q['hit@1']}  hit@5 {q['hit@5']}  hit@10 {q['hit@10']}  MRR {q['mrr']}")

    # Reference vectors: CPU, fp32, dynamic shapes - no padding, no precision reduction.
    reference = None
    results = []
    order = (["cpu-best"] if "cpu-best" in configs else []) + [c for c in configs if c != "cpu-best"]
    for name in order:
        try:
            result, vectors = run_config(name, core, texts, cases, corpus, reference)
            if name == "cpu-best":
                reference = vectors
        except Exception as e:
            result = {"config": name, "label": CONFIGS[name][0], "error": f"{type(e).__name__}: {e}"}
            print(f"   FAILED: {result['error']}")
        results.append(result)

    # Merge, so a later run of one device (e.g. the discrete GPU once a recording ends) keeps
    # the configs measured earlier instead of replacing the file.
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    merged = {r["config"]: r for r in previous.get("results", [])}
    stamp = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "recording_live": recording_in_progress()}
    merged.update({r["config"]: dict(r, **stamp) for r in results})
    RESULTS.write_text(json.dumps({**stamp, "baselines": baseline,
                                   "results": [merged[c] for c in CONFIGS if c in merged]}, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()

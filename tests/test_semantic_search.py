"""Meaning search on the notes page: ranking logic, answer quality and device parity.

Quality runs the real all-MiniLM-L6-v2 on the CPU - the same OpenVINO weights and embedding code
the service runs on the Intel iGPU - against a frozen copy of the notes and 32 paraphrased study
questions, written before any results were seen. tools/semantic_search_benchmark.py runs the same
questions on every device. Skipped when OpenVINO or the model files aren't installed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "Tools" / "npu-services"))
sys.path.insert(0, str(ROOT / "tools"))

ov = pytest.importorskip("openvino")
import runtime  # noqa: E402
import semantic_search_benchmark as bench  # noqa: E402

if not runtime.EMBEDDING_MODEL.exists():
    pytest.skip("embedding model not downloaded (Tools/npu-services/setup_models.py)", allow_module_level=True)

# The corpus is a copy of real lecture notes, so it stays out of git like notes/ itself.
CORPUS = Path(__file__).parent / "fixtures" / "semantic_search_corpus.json"
needs_corpus = pytest.mark.skipif(
    not CORPUS.exists(),
    reason="run: python tools/semantic_search_benchmark.py --freeze-corpus")


def test_long_rows_score_their_best_window():
    words = [f"w{i}" for i in range(300)]
    row = {"text": " ".join(words), "class_code": "X", "section": "S"}
    windows = runtime.passages(row["text"])
    assert [len(w.split()) for w in windows] == [160, 160, 20]
    assert windows[1].split()[0] == "w140"  # 20 words of overlap
    query = np.array([1.0, 0.0])
    embed = lambda t: np.array([1.0, 0.0]) if t == windows[2] else np.array([0.0, 1.0])
    other = {"text": "short line", "class_code": "X", "section": "T"}
    ranked = runtime.rank(query, [other, row], embed)
    assert ranked[0]["section"] == "S" and ranked[0]["semantic_score"] == 1.0


@pytest.fixture(scope="module")
def cpu():
    return runtime.Embedder(ov.Core(), "CPU", static_shape=False, batch=16, cache_dir=None)


@needs_corpus
def test_paraphrased_questions_find_the_right_section(cpu):
    cases, corpus = bench.load_cases()
    vectors, _, _ = bench.embed_all(cpu, bench.passage_texts(corpus), sort_by_length=True)
    meaning = bench.evaluate(cases, corpus, lambda t: vectors[t] if t in vectors else cpu(t))
    keyword = bench.keyword_baseline(cases, corpus)
    # Measured 2026-09-17: hit@1 0.72, hit@5 0.94, hit@10 1.0 - keyword search found none.
    assert meaning["hit@5"] >= 0.85 and meaning["hit@10"] >= 0.95, meaning
    assert meaning["mrr"] > keyword["mrr"] + 0.5


@needs_corpus
def test_unpadded_cpu_vectors_match_the_services_fixed_256_token_input(cpu):
    """Padding to 256 tokens (what the iGPU and NPU want) must not change the answer."""
    _, corpus = bench.load_cases()
    sample = [r["text"] for r in corpus[::97]][:20]
    service_shape = runtime.Embedder(ov.Core(), "CPU", static_shape=True, batch=1, cache_dir=None)
    cos = [float(service_shape(t) @ v) for t, v in zip(sample, cpu.embed_many(sample))]
    assert min(cos) > 0.999, cos


@needs_corpus
def test_batching_gives_the_same_vectors_as_one_at_a_time(cpu):
    """The service embeds a whole search in batches; a padded batch must not change a vector."""
    _, corpus = bench.load_cases()
    sample = [r["text"] for r in corpus[5::211]][:10]
    batched = runtime.Embedder(ov.Core(), "CPU", static_shape=True, batch=8, cache_dir=None)
    one = runtime.Embedder(ov.Core(), "CPU", static_shape=True, batch=1, cache_dir=None)
    cos = [float(a @ b) for a, b in zip(batched.embed_many(sample), one.embed_many(sample))]
    assert min(cos) > 0.9999, cos


def test_search_embeds_each_passage_once_and_then_reads_the_cache():
    """Every unique passage goes to the device exactly once, batched; a repeat search does no
    inference at all. Slow searches were per-row round trips with a commit each."""
    import sqlite3
    service = object.__new__(runtime.Runtime)
    service.lock = __import__("threading").RLock()
    service.cache = sqlite3.connect(":memory:")
    service.cache.execute("CREATE TABLE vectors (key TEXT PRIMARY KEY, value TEXT)")
    calls = []

    class FakeEmbedder:
        def embed_many(self, texts):
            calls.append(list(texts))
            return [np.full(4, float(len(t)), dtype=np.float32) for t in texts]

    service.embedder = FakeEmbedder()
    texts = ["alpha", "beta", "alpha", "gamma"]
    vectors = service.embeddings(texts)
    assert calls == [["alpha", "beta", "gamma"]]
    assert [v[0] for v in vectors] == [5.0, 4.0, 5.0, 5.0]
    assert service.embeddings(texts) and calls == [["alpha", "beta", "gamma"]]

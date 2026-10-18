"""Compare the existing formatter's two-call and one-call paths on synthetic fixtures.

Uses only the configured CLI model, never substitutes a provider, never writes
real notes. Regex checks are screening aids; saved outputs require semantic review.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

from recorder_probe import ROOT, emit, isolate_performance, run_probe


def screen(case, output):
    checks = {c["name"]: (bool(re.search(c["pattern"], output, re.I | re.S)) != c.get("forbidden", False))
              for c in case["checks"]}
    checks["one_date_heading"] = output.splitlines().count("## " + case["date"]) == 1
    checks["starts_with_date"] = output.startswith("## " + case["date"])
    return checks


def worker(args):
    isolate_performance()
    import notes
    # Never read or send a real class glossary in this synthetic benchmark.
    notes.vocab_module.terms_for_class = lambda _: []
    cases = json.loads(args.fixtures.read_text(encoding="utf-8"))
    emit({"model": notes.CLAUDE_MODEL, "provider": "existing Claude CLI",
          "content": "Synthetic fixtures only", "warning": "Pattern checks do not prove semantic parity"})
    for i, case in enumerate(cases):
        order = (True,) if args.combined_only else ((False, True) if i % 2 == 0 else (True, False))
        for combined in order:
            beginning = time.perf_counter()
            stages = {}
            if combined:
                prompt = notes._build_combined_prompt(case["title"], "SYNTHETIC_BENCHMARK", case["date"],
                                                      case["transcript"], case["prior_tail"])
            else:
                started = time.perf_counter()
                proof = notes._try_claude_cli_format(notes._build_proofread_prompt(
                    case["title"], "SYNTHETIC_BENCHMARK", case["transcript"]))
                stages["proofread_seconds"] = time.perf_counter() - started
                if proof is None:
                    raise RuntimeError("Proofread benchmark failed: " + str(notes.last_cli_failure))
                prompt = notes._build_notes_prompt(case["title"], "SYNTHETIC_BENCHMARK", case["date"],
                                                   proof, case["prior_tail"])
            started = time.perf_counter()
            output = notes._try_claude_cli_format(prompt)
            stages["notes_seconds"] = time.perf_counter() - started
            if output is None:
                raise RuntimeError("Formatter benchmark failed: " + str(notes.last_cli_failure))
            emit({"case": case["id"], "combined": combined, "seconds": time.perf_counter() - beginning,
                  "stages": stages, "checks": screen(case, output), "output": output})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tools/fixtures/formatter_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined-only", action="store_true", help="Retest the combined prompt after a fidelity fix")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        worker(args)
    else:
        run_probe(Path(__file__).resolve(), sys.argv[1:], args.output, timeout=1200, minimum_ram=1.5)

"""Measure a sync's embedding memory: peak and final footprint while embedding synthetic chunks.

Each configuration runs in a fresh process so one run's allocator state cannot leak into the
next. Text is generated here (random words and symbol soup), never read from Drive.

    uv run python scripts/embedding_memory.py                 # compare every configuration
    uv run python scripts/embedding_memory.py --only current  # one configuration
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL = "nomic-ai/nomic-embed-text-v1.5"
CACHE = Path(".drive-agent/models")
UPSERT_BATCH = 64  # texts per embed_documents call, as in indexing/sync.py

# "before" reproduces the settings that let a sync reach ~8 GB: the arena kept its peak,
# 16 texts per run, input order. "current" is FastEmbedEmbeddings as shipped.
CONFIGS = ("before", "current")

WORDS = [
    "policy",
    "compliance",
    "report",
    "quarter",
    "revenue",
    "growth",
    "customer",
    "onboarding",
    "risk",
    "control",
    "audit",
    "transaction",
    "settlement",
    "ledger",
    "account",
    "partner",
    "market",
    "review",
    "approval",
    "process",
    "document",
    "summary",
    "analysis",
    "budget",
    "forecast",
    "payment",
    "invoice",
    "contract",
    "vendor",
    "security",
    "incident",
    "team",
]


def _memory_mb() -> tuple[float, float]:
    """(current, lifetime peak) memory in MB. On macOS this is the physical footprint that
    Activity Monitor shows, which still counts pages that were compressed or swapped out (plain
    RSS drops when the machine is under memory pressure). Elsewhere: current and peak RSS."""
    if sys.platform == "darwin":
        import ctypes
        import os

        info = (ctypes.c_uint64 * 48)()  # rusage_info_v4: 16-byte uuid, then uint64 fields
        libc = ctypes.CDLL("libc.dylib")
        if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            raise OSError("proc_pid_rusage failed")
        fields = info[2:]
        return fields[7] / 2**20, fields[28] / 2**20  # ri_phys_footprint, lifetime max
    with open("/proc/self/status") as status:
        values = dict(line.split(":", 1) for line in status)
    return int(values["VmRSS"].split()[0]) / 1024, int(values["VmHWM"].split()[0]) / 1024


def synthetic_chunks(n: int, seed: int = 0) -> list[str]:
    """Mixed lengths: long chunks past the 1,024-token cap, ~800-token chunks, short ones, and
    symbol soup that tokenizes into many tokens per character (like garbled PDF text)."""
    rng = random.Random(seed)  # noqa: S311 - synthetic text, not security
    chunks = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.25:
            chunks.append(" ".join(rng.choices(WORDS, k=rng.randint(1100, 1600))))
        elif kind < 0.35:
            chars = "#@%&*|~^<>{}[]0123456789\u03b1\u03b2\u03b3\u03b4\u6f22\u5b57"
            chunks.append("".join(rng.choices(chars, k=rng.randint(1500, 4000))))
        elif kind < 0.75:
            chunks.append(" ".join(rng.choices(WORDS, k=rng.randint(550, 800))))
        else:
            chunks.append(" ".join(rng.choices(WORDS, k=rng.randint(10, 200))))
    return chunks


def run_one(name: str, n: int) -> dict[str, Any]:
    from drive_agent.embeddings import local

    chunks = synthetic_chunks(n)
    start_mb, _ = _memory_mb()
    if name == "before":
        local._release_arena_after_each_run = lambda model: None
        emb = local.FastEmbedEmbeddings(MODEL, cache_dir=CACHE, batch_size=16)
        model, prefix = emb._model, emb._doc_prefix

        def in_input_order(texts: list[str]) -> list[list[float]]:
            vectors = model.embed([prefix + t for t in texts], batch_size=16)
            return [v.tolist() for v in vectors]

        emb.embed_documents = in_input_order  # type: ignore[method-assign]
    else:
        emb = local.FastEmbedEmbeddings(MODEL, cache_dir=CACHE)
    loaded_mb, _ = _memory_mb()
    trace = []
    started = time.perf_counter()
    for i in range(0, len(chunks), UPSERT_BATCH):
        emb.embed_documents(chunks[i : i + UPSERT_BATCH])
        trace.append(round(_memory_mb()[0]))
    elapsed = time.perf_counter() - started
    final_mb, peak_mb = _memory_mb()
    time.sleep(3)  # let freed pages leave the footprint before the idle reading
    idle_mb, _ = _memory_mb()
    return {
        "config": name,
        "chunks": n,
        "start_mb": round(start_mb),
        "loaded_mb": round(loaded_mb),
        "peak_mb": round(peak_mb),
        "final_mb": round(final_mb),
        "idle_mb": round(idle_mb),
        "seconds": round(elapsed, 1),
        "chunks_per_s": round(n / elapsed, 1),
        "trace_mb": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=CONFIGS)
    parser.add_argument("--chunks", type=int, default=320)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run_one(args.only, args.chunks)))
        return
    for name in [args.only] if args.only else list(CONFIGS):
        out = subprocess.run(  # noqa: S603 - re-runs this script with our own arguments
            [sys.executable, __file__, "--child", "--only", name, "--chunks", str(args.chunks)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(out.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()

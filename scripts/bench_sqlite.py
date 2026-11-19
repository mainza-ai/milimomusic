#!/usr/bin/env python3
"""Measure the SQLite knobs before adopting them (plan 9.8).

The research was blunt: WAL + synchronous=NORMAL + busy_timeout are the wins; cache_size
and mmap_size are "measure it". This script measures, on a throwaway database, the
write and read throughput under each configuration so the decision is a number.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time

ROWS = 20_000


def _run(pragmas: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bench.db")
        conn = sqlite3.connect(path)
        for key, value in pragmas.items():
            conn.execute(f"PRAGMA {key}={value}")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload TEXT)")
        conn.commit()

        started = time.perf_counter()
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO t (payload) VALUES (?)",
            [(f"row-{i}-" + "x" * 64,) for i in range(ROWS)],
        )
        conn.commit()
        write_s = time.perf_counter() - started

        started = time.perf_counter()
        conn.execute("SELECT COUNT(*) FROM t").fetchone()
        conn.execute("SELECT payload FROM t WHERE id = ?", (ROWS // 2,)).fetchone()
        read_s = time.perf_counter() - started

        wal = path + "-wal"
        wal_kb = round(os.path.getsize(wal) / 1024, 1) if os.path.isfile(wal) else 0.0
        conn.close()
    return {
        "write_rows_per_s": round(ROWS / write_s) if write_s else None,
        "read_ms": round(read_s * 1000, 2),
        "wal_kb": wal_kb,
    }


CONFIGS = {
    "defaults": {},
    "wal+normal": {"journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 10000},
    "wal+normal+limit": {
        "journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 10000,
        "journal_size_limit": 67108864,
    },
    "wal+normal+cache64": {
        "journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 10000,
        "cache_size": -65536, "temp_store": "MEMORY",
    },
    "wal+normal+mmap256": {
        "journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 10000,
        "mmap_size": 268435456,
    },
}


def main() -> int:
    print(f"{'config':22s} {'writes/s':>9s} {'point read':>11s} {'wal KB':>8s}")
    results = {}
    for name, pragmas in CONFIGS.items():
        result = _run(pragmas)
        results[name] = result
        print(f"{name:22s} {result['write_rows_per_s'] or 0:>9d} "
              f"{result['read_ms']:>10.2f}ms {result['wal_kb']:>8.1f}")
    base = results["wal+normal"]["write_rows_per_s"] or 1
    print("\nverdict:")
    for name in ("wal+normal+cache64", "wal+normal+mmap256"):
        gain = (results[name]["write_rows_per_s"] or 0) / base
        print(f"  {name}: {gain:.2f}x vs wal+normal on writes")
    print("  Adopt a knob only for a measured win; see wiki/performance-audit-2026-11.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

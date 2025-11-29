#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from typing import List, Set, Tuple


def make_backbone_graph(n: int, extra_edges: int, rng: random.Random) -> List[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for i in range(n - 1):
        edges.add((i, i + 1))

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_pairs)
    for (u, v) in all_pairs:
        if (u, v) in edges:
            continue
        if v == u + 1:
            continue
        edges.add((u, v))
        if len(edges) >= (n - 1) + extra_edges:
            break

    return sorted(edges)


def write_instance(path: str, n: int, edges: List[Tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


def run_solver(ham_script: str, instance_path: str, glucose_cmd: str | None) -> Tuple[float, str, str, int]:
    cmd = [sys.executable, ham_script, "--input", instance_path, "--quiet"]
    if glucose_cmd:
        cmd += ["--glucose-cmd", glucose_cmd]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    return (t1 - t0, p.stdout, p.stderr, p.returncode)


def is_sat(stdout: str) -> bool:
    first = stdout.strip().splitlines()[:1]
    return bool(first) and first[0].strip() == "YES"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=10.0)
    ap.add_argument("--max-time", type=float, default=600.0)
    ap.add_argument("--min-n", type=int, default=70)
    ap.add_argument("--max-n", type=int, default=140)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--tries-per-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="instances/hard_sat_10s.txt")
    ap.add_argument("--glucose-cmd", type=str, default=None)
    ap.add_argument("--ham-script", type=str, default="hamiltonian_sat.py")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    extra_edge_options = [0, 5, 15, 30, 60, 120]

    for n in range(args.min_n, args.max_n + 1, args.step):
        for t in range(args.tries_per_n):
            extra = extra_edge_options[min(t, len(extra_edge_options) - 1)]
            edges = make_backbone_graph(n, extra, rng)

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
                tmp_path = tmp.name
            try:
                write_instance(tmp_path, n, edges)

                dt, out, err, code = run_solver(args.ham_script, tmp_path, args.glucose_cmd)
                sat = (code == 0) and is_sat(out)

                print(f"n={n:3d} extra={extra:3d}  time={dt:8.3f}s  sat={sat}", file=sys.stderr)

                if sat and (dt >= args.target) and (dt <= args.max_time):
                    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                    write_instance(args.out, n, edges)
                    print(f"\nFOUND: wrote {args.out} (n={n}, extra={extra}, time={dt:.3f}s)", file=sys.stderr)
                    return 0
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    print("No instance found in the searched range. Increase --max-n or adjust parameters.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

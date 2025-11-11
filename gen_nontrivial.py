#!/usr/bin/env python3
import argparse, math, os, random, subprocess, time
from collections import deque

def is_connected(n, edges):
    """Simple BFS on undirected graph to skip trivial UNSAT (disconnected) cases."""
    if n == 0: return True
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v); g[v].append(u)
    seen = [False]*n
    q = deque([0]); seen[0]=True; count=1
    while q:
        u = q.popleft()
        for w in g[u]:
            if not seen[w]:
                seen[w]=True; q.append(w); count+=1
    return count == n

def gen_random_graph(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges

def write_edge_instance(path, n, edges):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# n m\n{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")

def run_solver(hp_py, glucose_bin, inst_path, timeout=None):
    """Return (status, seconds). status in {'SAT','UNSAT','ERROR'}."""
    cmd = ["python3", hp_py, "--graph", inst_path, "--format", "edge",
           "--glucose-bin", glucose_bin]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False, encoding="utf-8", errors="replace")
        secs = time.time() - t0
    except subprocess.TimeoutExpired:
        return ("ERROR", None)
    out = proc.stdout + "\n" + proc.stderr
    if "Result: SAT" in out:
        return ("SAT", secs)
    if "Result: UNSAT" in out:
        return ("UNSAT", secs)
    return ("ERROR", secs)

def main():
    ap = argparse.ArgumentParser(description="Search for a ~10s nontrivial SAT Hamiltonian-path instance.")
    ap.add_argument("--glucose-bin", default="glucose", help="Path/name of Glucose binary")
    ap.add_argument("--script", default="hamilton_sat.py", help="Path to hamilton_sat.py")
    ap.add_argument("--out-dir", default="instances", help="Where to save instances")
    ap.add_argument("--target-sec", type=float, default=10.0, help="Target runtime seconds (≈)")
    ap.add_argument("--max-sec", type=float, default=600.0, help="Upper bound seconds (< 10 min)")
    ap.add_argument("--n-min", type=int, default=60)
    ap.add_argument("--n-max", type=int, default=120)
    ap.add_argument("--seeds-per-n", type=int, default=50, help="Tries per n")
    ap.add_argument("--p-scale-min", type=float, default=1.05, help="lower multiplier for ln(n)/n")
    ap.add_argument("--p-scale-max", type=float, default=1.6, help="upper multiplier for ln(n)/n")
    ap.add_argument("--timeout", type=int, default=900, help="Per-run wall timeout (s)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(0xC0FFEE)

    best = None  # (abs(diff), n, p, seed, status, secs, path)

    for n in range(args.n_min, args.n_max + 1):
        base = math.log(n)/n
        for _ in range(args.seeds_per_n):
            seed = rng.randrange(10**9)
            p = rng.uniform(args.p_scale_min, args.p_scale_max) * base
            rlocal = random.Random(seed)
            edges = gen_random_graph(n, p, rlocal)

            # Skip trivial unsat (disconnected)
            if not is_connected(n, edges):
                continue

            tmp_path = os.path.join(args.out_dir, f"cand_n{n}_seed{seed}_p{p:.5f}.txt")
            write_edge_instance(tmp_path, n, edges)

            status, secs = run_solver(args.script, args.glucose_bin, tmp_path, timeout=args.timeout)

            if status == "SAT":
                diff = abs((secs or 0) - args.target_sec)
                if secs is not None:
                    print(f"[SAT] n={n} |E|={len(edges)} p={p:.5f} seed={seed} time={secs:.2f}s -> {tmp_path}")
                    if (best is None) or (diff < best[0]):
                        best = (diff, n, p, seed, status, secs, tmp_path)
                if secs is not None and secs >= args.target_sec and secs <= args.max_sec:
                    final = os.path.join(args.out_dir, f"nontrivial_edge_n{n}_seed{seed}_p{p:.5f}.txt")
                    os.replace(tmp_path, final)
                    print(f"\n=== FOUND nontrivial instance ≈{secs:.2f}s ===")
                    print(final)
                    return
            elif status == "UNSAT":
                print(f"[UNSAT] n={n} |E|={len(edges)} p={p:.5f} seed={seed}")
                os.remove(tmp_path)
            else:
                print(f"[ERROR] n={n} p={p:.5f} seed={seed} (timeout or parse error)")
                try: os.remove(tmp_path)
                except: pass

    if best:
        print("\nCould not hit the 10s target exactly, but best SAT was:")
        _, n, p, seed, _, secs, path = best
        print(f"n={n}, p={p:.5f}, seed={seed}, time={secs:.2f}s, file={path}")
        print("You can include this as your 'nontrivial' instance and report the observed time.")
    else:
        print("\nNo SAT instance found in the search range. Expand n / seeds / p-scale range and retry.")

if __name__ == "__main__":
    main()

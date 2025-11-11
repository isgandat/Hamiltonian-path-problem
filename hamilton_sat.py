#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import subprocess
from typing import List, Tuple, Optional, Set

# --------------- Graph parsing ----------------

def read_graph_edge_list(path: str) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Format A (default): 
      First non-comment line: n m
      Then m lines: u v  (0-based, undirected)
      Lines starting with '#' are comments.
    """
    n = m = None
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if n is None:
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError("First non-comment line must be: n m")
                n, m = map(int, parts)
            else:
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError("Edge line must be: u v")
                u, v = map(int, parts)
                if not (0 <= u < n and 0 <= v < n):
                    raise ValueError(f"Edge out of range: {u} {v}")
                if u != v:
                    edges.append((min(u, v), max(u, v)))
    # deduplicate
    edges = sorted(set(edges))
    return n, edges

def read_graph_dimacs_like(path: str) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Format B (DIMACS-like):
        c comments...
        p edge n m
        e u v    (1-based, undirected)
    We will convert to 0-based internally.
    """
    n = None
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("c"):
                continue
            if s.startswith("p"):
                parts = s.split()
                # p edge n m
                if len(parts) != 4 or parts[1] != "edge":
                    raise ValueError("Expected: p edge n m")
                n = int(parts[2])
            elif s.startswith("e"):
                parts = s.split()
                if len(parts) != 3:
                    raise ValueError("Edge line must be: e u v")
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                if u != v:
                    edges.append((min(u, v), max(u, v)))
    if n is None:
        raise ValueError("Missing 'p edge n m' line")
    edges = sorted(set(edges))
    return n, edges

def read_graph_matrix(path: str) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Format C (adjacency matrix, 0/1, whitespace separated):
      n lines, each has n entries (0/1).
    Symmetric assumed (undirected). Diagonals should be 0.
    """
    mat = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            row = list(map(int, parts))
            mat.append(row)
    if len(mat) == 0 or any(len(r) != len(mat) for r in mat):
        raise ValueError("Adjacency matrix must be square (n x n).")
    n = len(mat)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if mat[i][j] != 0:
                edges.append((i, j))
    return n, edges

# --------------- CNF builder ----------------

class CNF:
    def __init__(self, num_vars: int = 0):
        self.num_vars = num_vars
        self.clauses: List[List[int]] = []

    def add_clause(self, lits: List[int]):
        # 0 not allowed; DIMACS vars are ±1..±num_vars
        self.clauses.append(list(lits))

    def extend_vars(self, required: int):
        if required > self.num_vars:
            self.num_vars = required

    def to_dimacs(self, var_comments: Optional[List[str]] = None) -> str:
        lines = []
        if var_comments:
            for c in var_comments:
                lines.append(f"c {c}")
        lines.append(f"p cnf {self.num_vars} {len(self.clauses)}")
        for cl in self.clauses:
            lines.append(" ".join(str(l) for l in cl) + " 0")
        return "\n".join(lines) + "\n"

# --------------- Hamiltonian Path encoding ----------------

def var_id(v: int, p: int, n: int) -> int:
    """
    Variable x_{v,p}: vertex v is in position p (0..n-1).
    ID in [1..n*n] as v*n + p + 1.
    """
    return v * n + p + 1

def hamiltonian_path_cnf(n: int,
                         edges: List[Tuple[int,int]],
                         start: Optional[int] = None,
                         end: Optional[int] = None,
                         fix_first: Optional[int] = None) -> Tuple[CNF, List[str]]:
    """
    Build CNF for the existence of a Hamiltonian path in an undirected graph:
      - exactly one vertex at each position,
      - each vertex appears in exactly one position,
      - consecutive positions must be adjacent in G,
      - optional: start vertex fixed at position 0, end vertex fixed at position n-1,
      - optional: fix_first for simple symmetry breaking (fix a vertex at position 0).
    """
    edge_set: Set[Tuple[int,int]] = set()
    for (u, v) in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))

    cnf = CNF(num_vars=n*n)
    comments = []
    comments.append("Hamiltonian Path encoding")
    comments.append(f"n={n}, |E|={len(edge_set)}")
    comments.append("Variables x_{v,p} (1..n*n): v in [0..n-1], p in [0..n-1]")
    comments.append("ID(x_{v,p}) = v*n + p + 1")

    # 1) Each position has exactly one vertex
    for p in range(n):
        # at least one
        cnf.add_clause([var_id(v, p, n) for v in range(n)])
        # at most one (pairwise)
        for v1 in range(n):
            for v2 in range(v1+1, n):
                cnf.add_clause([-var_id(v1, p, n), -var_id(v2, p, n)])

    # 2) Each vertex appears in exactly one position
    for v in range(n):
        # at least one
        cnf.add_clause([var_id(v, p, n) for p in range(n)])
        # at most one (pairwise)
        for p1 in range(n):
            for p2 in range(p1+1, n):
                cnf.add_clause([-var_id(v, p1, n), -var_id(v, p2, n)])

    # 3) Adjacency constraints for consecutive positions
    # For every (p, p+1), forbid (u at p) and (w at p+1) if {u,w} not an edge.
    for p in range(n-1):
        for u in range(n):
            for w in range(n):
                if u == w:
                    # Not needed because of "each vertex exactly once" but harmless:
                    cnf.add_clause([-var_id(u, p, n), -var_id(w, p+1, n)])
                else:
                    a, b = (u, w) if u < w else (w, u)
                    if (a, b) not in edge_set:
                        cnf.add_clause([-var_id(u, p, n), -var_id(w, p+1, n)])

    # Optional: fix start or end vertex
    if start is not None:
        if not (0 <= start < n):
            raise ValueError("--start must be in [0..n-1]")
        cnf.add_clause([var_id(start, 0, n)])
        comments.append(f"Start fixed: vertex {start} at position 0")

    if end is not None:
        if not (0 <= end < n):
            raise ValueError("--end must be in [0..n-1]")
        cnf.add_clause([var_id(end, n-1, n)])
        comments.append(f"End fixed: vertex {end} at position {n-1}")

    # Optional: simple symmetry breaking
    if fix_first is not None and start is None:
        if not (0 <= fix_first < n):
            raise ValueError("--fix-first must be in [0..n-1]")
        cnf.add_clause([var_id(fix_first, 0, n)])
        comments.append(f"Symmetry breaking: vertex {fix_first} at position 0")

    return cnf, comments

# --------------- Glucose interface ----------------

def run_glucose(glucose_bin: str, cnf_path: str, timeout: Optional[int], want_model: bool=True) -> Tuple[str, str, int]:
    """
    Call Glucose 4.2 (or compatible) on CNF file.
    - For a model, we pass '-model' so it prints 'v ...' lines.
    Returns (stdout, stderr, returncode).
    """
    cmd = [glucose_bin]
    if want_model:
        cmd.append("-model")
    cmd.append(cnf_path)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace"
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout} seconds", -1
    except FileNotFoundError:
        return "", f"ERROR: Glucose binary not found at '{glucose_bin}'", -2

# --------------- Model decoding ----------------

def decode_model_to_path(model_lits: List[int], n: int) -> Optional[List[int]]:
    """
    model_lits: a list of literals from Glucose 'v ...' lines.
    Return the Hamiltonian path as a sequence of vertices of length n
    (positions 0..n-1), or None if no consistent assignment found.
    """
    # We only care about positive literals in 1..n*n
    assignment = set(l for l in model_lits if l > 0 and 1 <= abs(l) <= n*n)
    pos_to_v = [-1] * n
    for v in range(n):
        for p in range(n):
            vid = var_id(v, p, n)
            if vid in assignment:
                if pos_to_v[p] != -1:
                    # inconsistent (shouldn't happen if solver gave a correct model)
                    return None
                pos_to_v[p] = v
    if any(x == -1 for x in pos_to_v):
        return None
    return pos_to_v

def parse_glucose_model(stdout: str) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Parse Glucose output. Return (lits, status) where:
      - lits: all literals appearing on 'v ' model lines (list of ints)
      - status: 'SATISFIABLE' or 'UNSATISFIABLE' or None
    """
    lits = []
    status = None
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("s "):
            # s SATISFIABLE / s UNSATISFIABLE (depending on build)
            if "SATISFIABLE" in s:
                status = "SATISFIABLE" if "UN" not in s else "UNSATISFIABLE"
        elif s.startswith("v ") or s.startswith("V "):
            parts = s.split()[1:]
            for tok in parts:
                if tok == "0":  # end of model line
                    continue
                try:
                    lits.append(int(tok))
                except ValueError:
                    pass
    return (lits if lits else None, status)

# --------------- CLI ----------------

def load_graph(path: str, fmt: str) -> Tuple[int, List[Tuple[int,int]]]:
    if fmt == "edge":
        return read_graph_edge_list(path)
    elif fmt == "dimacs":
        return read_graph_dimacs_like(path)
    elif fmt == "matrix":
        return read_graph_matrix(path)
    else:
        raise ValueError(f"Unknown --format {fmt}")

def main():
    ap = argparse.ArgumentParser(
        description="Hamiltonian Path -> SAT (Glucose 4.2) encoder/solver"
    )
    ap.add_argument("--graph", type=str, required=True,
                    help="Path to graph instance")
    ap.add_argument("--format", type=str, default="edge",
                    choices=["edge", "dimacs", "matrix"],
                    help="Input graph format (default: edge)")
    ap.add_argument("--glucose-bin", type=str, default="glucose",
                    help="Path to Glucose 4.2 binary (e.g., glucose, glucose-syrup)")
    ap.add_argument("--output-cnf", type=str, default=None,
                    help="If set, write DIMACS CNF to this path")
    ap.add_argument("--no-solve", action="store_true",
                    help="If set, only build CNF (do not call solver)")
    ap.add_argument("--start", type=int, default=None,
                    help="Fix start vertex at position 0")
    ap.add_argument("--end", type=int, default=None,
                    help="Fix end vertex at position n-1")
    ap.add_argument("--fix-first", type=int, default=None,
                    help="Symmetry break: fix this vertex at position 0 (ignored if --start given)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Solver timeout in seconds")
    ap.add_argument("--stats", action="store_true",
                    help="Print raw solver output (stats)")
    ap.add_argument("--print-model", action="store_true",
                    help="Also print raw model literals (debug).")

    args = ap.parse_args()

    n, edges = load_graph(args.graph, args.format)
    cnf, comments = hamiltonian_path_cnf(
        n, edges, start=args.start, end=args.end, fix_first=args.fix_first
    )

    # Prepare DIMACS (with comments describing var mapping)
    dimacs = cnf.to_dimacs(var_comments=comments)

    # Write CNF if requested
    if args.output_cnf:
        with open(args.output_cnf, "w", encoding="utf-8") as f:
            f.write(dimacs)

    if args.no_solve:
        print(f"CNF built: vars={cnf.num_vars}, clauses={len(cnf.clauses)}")
        if not args.output_cnf:
            # if not written, print to stdout so the user can redirect
            sys.stdout.write(dimacs)
        return

    # Solve
    with tempfile.NamedTemporaryFile(prefix="hamilton_", suffix=".cnf", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(dimacs)
        tmp_path = tmp.name

    stdout, stderr, code = run_glucose(args.glucose_bin, tmp_path, args.timeout, want_model=True)
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if code == -2:
        print(stderr, file=sys.stderr)
        sys.exit(2)
    if code == -1:
        print(stderr, file=sys.stderr)
        sys.exit(3)

    if args.stats:
        # print full solver output (stdout + stderr) for reproducibility
        print("----- Glucose stdout -----")
        print(stdout.rstrip())
        if stderr.strip():
            print("----- Glucose stderr -----", file=sys.stderr)
            print(stderr.rstrip(), file=sys.stderr)

    model_lits, status = parse_glucose_model(stdout)

    if status == "UNSATISFIABLE":
        print("Result: UNSAT (no Hamiltonian path under given constraints).")
        return
    elif status == "SATISFIABLE":
        if model_lits is None:
            print("Result: SAT but no model lines found. Try a Glucose build with '-model' support.", file=sys.stderr)
            sys.exit(4)
        path = decode_model_to_path(model_lits, n)
        if path is None:
            print("Result: SAT but model decoding failed.", file=sys.stderr)
            if args.print_model:
                print("Model literals:", model_lits)
            sys.exit(5)
        print("Result: SAT (Hamiltonian path found).")
        print("Path (positions 0..n-1):")
        print(" ".join(map(str, path)))
        # Quick verification (optional, human-friendly)
        edge_set = { (min(u,v), max(u,v)) for (u,v) in edges }
        ok = True
        for i in range(n-1):
            a, b = path[i], path[i+1]
            if (min(a,b), max(a,b)) not in edge_set:
                ok = False
        print(f"Verified adjacency: {'OK' if ok else 'FAILED'}")
        if args.start is not None:
            print(f"Start check: {'OK' if path[0]==args.start else 'FAILED'}")
        if args.end is not None:
            print(f"End check: {'OK' if path[-1]==args.end else 'FAILED'}")
        if args.print_model:
            print("Model literals:", model_lits)
        return
    else:
        # Could not parse 's ' line from solver; still try decoding if model exists
        if model_lits:
            path = decode_model_to_path(model_lits, n)
            if path:
                print("Result: SAT (inferred from model).")
                print(" ".join(map(str, path)))
                return
        print("ERROR: Could not determine SAT/UNSAT from Glucose output.", file=sys.stderr)
        if args.stats:
            print(stdout)
        sys.exit(6)

if __name__ == "__main__":
    main()

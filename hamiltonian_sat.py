#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Tuple, Dict, Set


def read_graph(path: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Read an undirected graph from a file or stdin.

    Format:
        n m
        u v
        u v
        ...

    Vertices are 0-based integers in [0, n-1].
    Lines starting with # are ignored.
    """
    if path == "-" or path is None:
        lines = sys.stdin.read().strip().splitlines()
    else:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()

    tokens: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(line.split())

    if len(tokens) < 2:
        raise ValueError("Input must start with 'n m' on the first non-comment line.")

    it = iter(tokens)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        raise ValueError("Cannot read n and m from input.")
    except ValueError:
        raise ValueError("n and m must be integers.")

    edges: List[Tuple[int, int]] = []
    for i in range(m):
        try:
            u = int(next(it))
            v = int(next(it))
        except StopIteration:
            raise ValueError(f"Not enough edge endpoints, expected {m} edges.")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u}, {v}) has vertex outside [0, {n-1}].")
        if u == v:
            # ignore self-loops, they do not help Hamiltonian path
            continue
        edges.append((u, v))

    return n, edges


class HamiltonianPathEncoding:
    """Encode Hamiltonian Path problem to CNF.

    Variables x(v, p): vertex v is at position p in the path.
    v in [0, n-1], p in [0, n-1].
    Variable indices are in [1, n*n].
    """

    def __init__(self, n: int, edges: List[Tuple[int, int]]):
        self.n = n
        # store edges as undirected set with (min,max)
        self.edge_set: Set[Tuple[int, int]] = set()
        for u, v in edges:
            if u == v:
                continue
            if u > v:
                u, v = v, u
            self.edge_set.add((u, v))
        self.clauses: List[List[int]] = []

    def var(self, v: int, p: int) -> int:
        """Map (vertex, position) to DIMACS variable id (1-based)."""
        return v * self.n + p + 1

    def add_clause(self, lits: List[int]) -> None:
        self.clauses.append(lits)

    def build(self) -> Tuple[int, List[List[int]]]:
        n = self.n

        # 1) Each position p is occupied by at least one vertex.
        for p in range(n):
            clause = [self.var(v, p) for v in range(n)]
            self.add_clause(clause)

        # 2) No position has two different vertices: at most one vertex per position.
        for p in range(n):
            for v1 in range(n):
                for v2 in range(v1 + 1, n):
                    self.add_clause([-self.var(v1, p), -self.var(v2, p)])

        # 3) Each vertex appears in at least one position.
        for v in range(n):
            clause = [self.var(v, p) for p in range(n)]
            self.add_clause(clause)

        # 4) No vertex appears twice: at most one position per vertex.
        for v in range(n):
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    self.add_clause([-self.var(v, p1), -self.var(v, p2)])

        # 5) Adjacency constraints:
        # If (u, v) is not an edge, u and v cannot be consecutive in the path.
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in self.edge_set:
                    continue
                for p in range(n - 1):
                    # u at p, v at p+1 is forbidden
                    self.add_clause([-self.var(u, p), -self.var(v, p + 1)])
                    # v at p, u at p+1 is forbidden
                    self.add_clause([-self.var(v, p), -self.var(u, p + 1)])

        num_vars = n * n
        return num_vars, self.clauses


def write_dimacs(num_vars: int, clauses: List[List[int]], path: str) -> None:
    """Write CNF in DIMACS format to the given file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            line = " ".join(str(lit) for lit in clause) + " 0\n"
            f.write(line)


def find_glucose(explicit_cmd: str = None) -> str:
    """Find glucose or glucose-syrup binary."""
    if explicit_cmd:
        return explicit_cmd
    candidates = ["glucose", "glucose-syrup"]
    for cmd in candidates:
        try:
            # Just check that the binary exists and is executable.
            subprocess.run(
                [cmd, "-h"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return cmd
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "Glucose solver not found. Install 'glucose' or 'glucose-syrup' and ensure it is in PATH "
        "or specify --glucose-cmd."
    )


def run_glucose(glucose_cmd: str, cnf_path: str, show_stats: bool) -> Tuple[bool, Dict[int, bool], str]:
    """Run glucose on cnf_path, return (is_sat, model_dict, raw_output)."""
    proc = subprocess.run(
        [glucose_cmd, "-model", cnf_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout

    if show_stats:
        print("======== Glucose output ========", file=sys.stderr)
        print(output, file=sys.stderr)
        print("======== End of Glucose output ========", file=sys.stderr)

    is_sat = False
    model: Dict[int, bool] = {}

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("s "):
            if "UNSAT" in line:
                is_sat = False
            elif "SAT" in line:
                is_sat = True
        elif line.startswith("v ") or line[0] in "-0123456789":
            if line.startswith("v "):
                parts = line.split()[1:]
            else:
                parts = line.split()
            for tok in parts:
                if tok == "0":
                    break
                lit = int(tok)
                var = abs(lit)
                val = lit > 0
                model[var] = val

    return is_sat, model, output


def decode_hamiltonian_path(n: int, model: Dict[int, bool]) -> List[int]:
    """Given a model for x(v, p) variables, return the Hamiltonian path as a list of vertices."""
    vertices_by_pos: Dict[int, int] = {}

    def var_index(v: int, p: int) -> int:
        return v * n + p + 1

    for v in range(n):
        for p in range(n):
            var = var_index(v, p)
            if model.get(var, False):
                if p in vertices_by_pos:
                    # In case of conflicting assignment, keep the first.
                    continue
                vertices_by_pos[p] = v

    if len(vertices_by_pos) != n:
        raise ValueError("Model does not assign exactly one vertex to every position.")

    path = [vertices_by_pos[p] for p in range(n)]
    return path


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Hamiltonian Path SAT encoder using Glucose."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to input graph file (or '-' for stdin).",
    )
    parser.add_argument(
        "--cnf-out",
        default=None,
        help="Path to save the generated CNF in DIMACS format.",
    )
    parser.add_argument(
        "--only-cnf",
        action="store_true",
        help="Only generate CNF and do not run the SAT solver.",
    )
    parser.add_argument(
        "--glucose-cmd",
        default=None,
        help="Path to the Glucose binary (if not in PATH).",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print raw output (including statistics) from Glucose to stderr.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output.",
    )

    args = parser.parse_args(argv)

    try:
        n, edges = read_graph(args.input)
    except Exception as e:
        print(f"Error reading input graph: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Read graph with {n} vertices and {len(edges)} edges.", file=sys.stderr)

    encoder = HamiltonianPathEncoding(n, edges)
    num_vars, clauses = encoder.build()

    if not args.quiet:
        print(
            f"Built CNF with {num_vars} variables and {len(clauses)} clauses.",
            file=sys.stderr,
        )

    # Decide CNF path
    cnf_path: str
    tmp_file = None
    if args.cnf_out:
        cnf_path = args.cnf_out
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cnf")
        cnf_path = tmp.name
        tmp.close()
        tmp_file = cnf_path

    try:
        write_dimacs(num_vars, clauses, cnf_path)
    except Exception as e:
        print(f"Error writing CNF to '{cnf_path}': {e}", file=sys.stderr)
        return 1

    # IMPORTANT FIX: use args.only_cnf (underscore), not args.only-cnf
    if args.only_cnf:
        if not args.quiet:
            print(f"CNF written to {cnf_path}.", file=sys.stderr)
        return 0

    try:
        glucose_cmd = find_glucose(args.glucose_cmd)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    is_sat, model, _ = run_glucose(glucose_cmd, cnf_path, args.show_stats)

    if tmp_file is not None:
        try:
            os.unlink(tmp_file)
        except OSError:
            pass

    if not is_sat:
        print("NO")
        if not args.quiet:
            print("The graph does not have a Hamiltonian path.", file=sys.stderr)
        return 0

    try:
        path = decode_hamiltonian_path(n, model)
    except Exception as e:
        print(f"Error decoding model: {e}", file=sys.stderr)
        return 1

    print("YES")
    print("Hamiltonian path (0-based vertex indices):")
    print(" ".join(str(v) for v in path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# Hamiltonian Path via SAT (Glucose 4.2)

This project solves the **Hamiltonian Path** problem for undirected graphs by encoding it into SAT (DIMACS CNF) and running **Glucose 4.2** (or 4.2.x).

A *Hamiltonian path* in a graph $G = (V, E)$ is a simple path that visits every vertex exactly once. In this project the input vertices are labeled `0..n-1`, and a solution is printed as an ordering of all vertices.

---

## 1. Problem definition (parameters and constraints)

**Input:** an undirected graph $G = (V, E)$

- $V = \{0,1,\dots,n-1\}$
- $E$ is a set of undirected edges $\{u,v\}$ with $u \neq v$

**Question:** does there exist an ordering $v_0, v_1, \dots, v_{n-1}$ such that:

- every vertex appears exactly once, and
- for every $i \in \{0,\dots,n-2\}$, $\{v_i, v_{i+1}\} \in E$

If the answer is YES, the script prints one such ordering.

---

## 2. CNF encoding (vertex–position variables)

Let the path positions be `0..n-1`. The encoding uses variables of the form:

- **x(v,p)** is true iff vertex `v` is placed at position `p` in the path.

### DIMACS variable mapping

The implementation maps $x(v,p)$ to a DIMACS variable id:

- `var(v,p) = v*n + p + 1`  (so ids are `1..n^2`)

### Constraints (clause families)

The CNF contains the following clause families:

1) **Exactly one vertex per position**

- *At least one* vertex in each position `p`:
  - `x(0,p) OR x(1,p) OR ... OR x(n-1,p)`
- *At most one* vertex in each position `p` (pairwise):
  - for all `v1 < v2`: `NOT x(v1,p) OR NOT x(v2,p)`

2) **Exactly one position per vertex**

- *At least one* position for each vertex `v`:
  - `x(v,0) OR x(v,1) OR ... OR x(v,n-1)`
- *At most one* position for each vertex `v` (pairwise):
  - for all `p1 < p2`: `NOT x(v,p1) OR NOT x(v,p2)`

3) **Adjacency (consecutive vertices must be connected)**

For every **non-edge** `{u,v}` (with `u != v`), forbid placing `u` and `v` next to each other:

- for every position `p = 0..n-2`:
  - `NOT x(u,p) OR NOT x(v,p+1)`
  - `NOT x(v,p) OR NOT x(u,p+1)`

This makes the SAT model correspond to a permutation of vertices where consecutive vertices are always adjacent in the input graph.

---

## 3. Alternatives (and why this one was used)

There are several reasonable ways to encode Hamiltonian Path to SAT:

- **Different “at most one” encodings.**
  This project uses the simple *pairwise* encoding. For larger `n`, one could replace it with sequential counters or sorting networks to reduce the number of clauses.

- **Adjacency as implications instead of forbidding non-edges.**
  Instead of adding clauses for every non-edge, you can enforce:
  `x(u,p) -> (OR of x(v,p+1) for v in N(u))`.
  That typically produces fewer clauses on sparse graphs (but uses large disjunctions).

- **Symmetry breaking.**
  If a path exists, reversing it is also a solution. Fixing the first vertex (e.g., `x(0,0)`) can reduce that symmetry and sometimes speed up solving. I kept the encoding symmetric because it is easier to explain and matches the textbook formulation.

---

## 4. Script usage (input and output)

### Requirements

- Python 3
- Glucose 4.2 installed as `glucose` or `glucose-syrup` (or provide the path with `--glucose-cmd`)

### Input format

Each instance file is:

```text
n m
u1 v1
u2 v2
...
um vm
```

- vertices are `0..n-1`
- edges are undirected
- blank lines and lines starting with `#` are ignored
- self-loops are ignored

### Run solver

```bash
python3 hamiltonian_sat.py instances/small_sat.txt
```

or equivalently:

```bash
python3 hamiltonian_sat.py --input instances/small_sat.txt
```

### CNF only (DIMACS)

```bash
python3 hamiltonian_sat.py --input instances/small_sat.txt --cnf-out out.cnf --only-cnf
```

### Print solver statistics

```bash
python3 hamiltonian_sat.py --input instances/small_sat.txt --show-stats
```

### Output format

- UNSAT:

```text
NO
```

- SAT:

```text
YES
Hamiltonian path (0-based vertex indices):
3 2 1 0
```

(Extra diagnostics go to stderr; use `--quiet` to suppress them.)

---

## 5. Included instances

All instances are in `instances/`:

- `small_sat.txt` – small satisfiable example.
- `small_unsat.txt` – small unsatisfiable example (graph has an isolated vertex).
- `nontrivial_sat.txt` – 20 vertices with a backbone plus a few chords.
- `heavy_sat_50.txt` – 50 vertices with a backbone plus extra edges.

Hard / timing-oriented:

- `hard_backbone_125.txt` – 125-vertex pure backbone (very sparse → many non-edge constraints).
- `hard_backbone_140.txt` – same idea, larger (intended for the “~10 seconds” requirement on typical machines).

CNF sizes (from the DIMACS header when generating CNF):

- `heavy_sat_50.txt`: `p cnf 2500 237064`
- `hard_backbone_125.txt`: `p cnf 15625 3828997`
- `hard_backbone_140.txt`: `p cnf 19600 5390978`

---

## 6. Experiments / runtime report

Runtime depends on hardware and the Glucose build, so the project includes both:

- a couple of larger fixed instances, and
- a small helper to search for a “hard enough” satisfiable instance on *your* machine.

### Quick benchmark commands

Measure total time (CNF + solver):

```bash
time python3 hamiltonian_sat.py --input instances/heavy_sat_50.txt --quiet
```

Generate CNF only (useful to see encoding overhead):

```bash
time python3 hamiltonian_sat.py --input instances/hard_backbone_140.txt --only-cnf --cnf-out /tmp/hp.cnf --quiet
```

On a Linux test machine (Python 3, Glucose not included here), the **CNF generation only** times were roughly:

- `heavy_sat_50.txt`: ~0.4 s
- `hard_backbone_125.txt`: ~6–7 s
- `hard_backbone_140.txt`: ~10–12 s

Glucose solving time depends on your build and CPU, but for the larger backbone instances the *total* runtime is usually dominated by CNF size.

### Finding a satisfiable instance that takes ≥ 10 seconds

The script `find_10s_instance.py` searches backbone-style graphs (satisfiable by construction) and measures wall-clock time:

```bash
python3 find_10s_instance.py --target 10
```

If it succeeds, it writes the found instance (by default to `instances/hard_sat_10s.txt`). If it doesn’t, it still prints what sizes it tried so you can adjust the search range (`--max-n`, `--tries-per-n`, etc.).

Why this works well for this project: with the chosen encoding, **sparse graphs** create many “non-edge adjacency” clauses, which makes the CNF much larger and typically slows down both CNF generation and solving.

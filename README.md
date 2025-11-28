# Hamiltonian Path via SAT (Glucose 4.2)

## 1. Problem description

This project solves the **Hamiltonian Path** problem using a SAT solver.

Given a finite undirected graph \( G = (V, E) \), a *Hamiltonian path* is a simple path that visits every vertex exactly once. The decision problem is:

> Given a graph \( G \), does there exist a Hamiltonian path in \( G \)?

### Parameters and constraints

- The input is a finite undirected graph.
- Vertices are labelled by integers \( 0, 1, \dots, n-1 \).
- Edges are unordered pairs \( \{u, v\} \) with \( u \neq v \).
- A valid Hamiltonian path is an ordering of all vertices where  
  - each vertex appears exactly once, and  
  - consecutive vertices in the ordering are adjacent in the graph.

The script answers **YES** or **NO**, and for YES it prints one Hamiltonian path.

This is a pure decision problem, so it fits the assignment requirement to consider yes/no problems.

---

## 2. CNF encoding

The encoding uses the classical “vertex–position” model.

### Propositional variables

Assume the graph has \( n \) vertices and we look for a path with positions \( 0, \dots, n-1 \).

We introduce propositional variables

\[
x_{v,p} \quad\text{for } v \in \{0, \dots, n-1\},\ p \in \{0, \dots, n-1\},
\]

with the intended meaning:

> \( x_{v,p} \) is true iff vertex \( v \) appears at position \( p \) in the Hamiltonian path.

In DIMACS, variables are mapped to integers by:

\[
\text{var}(v, p) = v \cdot n + p + 1.
\]

So we use exactly \( n^2 \) Boolean variables.

### Clauses

The CNF is the conjunction of the following clause families.

#### (A) Every position is occupied by some vertex

For each position \( p \in \{0, \dots, n-1\} \):

\[
x_{0,p} \lor x_{1,p} \lor \dots \lor x_{n-1,p}.
\]

This ensures that each position in the path contains at least one vertex.

#### (B) No position contains two different vertices

For each position \( p \) and for all distinct vertices \( v_1 \neq v_2 \):

\[
\neg x_{v_1,p} \lor \neg x_{v_2,p}.
\]

Together with (A), this enforces that each position is occupied by **exactly** one vertex.

#### (C) Every vertex appears somewhere in the path

For each vertex \( v \in \{0, \dots, n-1\} \):

\[
x_{v,0} \lor x_{v,1} \lor \dots \lor x_{v,n-1}.
\]

#### (D) No vertex appears more than once

For each vertex \( v \) and for all distinct positions \( p_1 \neq p_2 \):

\[
\neg x_{v,p_1} \lor \neg x_{v,p_2}.
\]

Together with (C), this means that every vertex appears in **exactly** one position.

#### (E) Adjacency constraints

Let \( E \) be the set of edges of the graph, represented as unordered pairs \( \{u, v\} \) with \( u < v \).

For every pair of distinct vertices \( (u, v) \) with \( \{u, v\} \notin E \) (a non-edge), we forbid them from being consecutive in the path.

For each such non-edge and each position \( p = 0, \dots, n-2 \):

\[
\neg x_{u,p} \lor \neg x_{v,p+1},
\]
\[
\neg x_{v,p} \lor \neg x_{u,p+1}.
\]

This ensures that whenever two vertices appear next to each other in the path, they are connected by an edge.

### Discussion and alternatives

The chosen encoding is straightforward and standard:

- Uses \( n^2 \) variables and \( O(n^3) \) clauses.
- It is symmetric (e.g. reversing the path gives another solution), but easy to implement and reason about.

Possible alternatives (not implemented here):

- **Symmetry breaking**, e.g. fixing vertex 0 to position 0 to reduce equivalent solutions.
- **Successor-based encoding**, with variables \( y_{u,v} \) for “\( v \) is the successor of \( u \)” in the path.
- Additional problem-specific pruning constraints based on degrees, connectivity, etc.

For the scope of this project, the vertex–position encoding is sufficient and transparent.

---

## 3. Script usage and formats

The main script is `hamiltonian_sat.py`. It is written in Python 3 and uses only the standard library. SAT solving is delegated to Glucose 4.2, called as an external binary.

### Input format (graph instance)

The input graph is given in a simple text format:

```text
n m
u1 v1
u2 v2
...
um vm
```

- `n` – number of vertices (integer, `n >= 1`).
- `m` – number of edges (integer, `m >= 0`).
- Each `ui vi` is an undirected edge.
- Vertices are 0-based: integers in `[0, n-1]`.
- Lines starting with `#` are treated as comments and ignored.
- Self-loops (`u == v`) are read but ignored (they do not help the Hamiltonian path).

### Output format (solution)

By default (solve mode), the script prints to **stdout**:

- If there is **no** Hamiltonian path:

  ```text
  NO
  ```

- If there **is** a Hamiltonian path:

  ```text
  YES
  Hamiltonian path (0-based vertex indices):
  3 2 1 0
  ```

The second line contains one Hamiltonian path as a sequence of vertex indices.

Diagnostic information (graph size, CNF size, solver statistics) is printed to **stderr**, unless `--quiet` is used.

### Command-line options

Run:

```bash
python3 hamiltonian_sat.py -h
```

to see a short help. Main options:

- `--input / -i PATH`  
  Path to the input graph file.  
  If omitted or `-`, the script reads from stdin.

- `--cnf-out PATH`  
  Write the generated CNF in DIMACS format to `PATH`.

- `--only-cnf`  
  Only generate the CNF and exit (the SAT solver is **not** called).

- `--glucose-cmd PATH`  
  Explicit path to the Glucose executable.  
  If not given, the script tries `glucose` and `glucose-syrup` in `PATH`.

- `--show-stats`  
  Print the full Glucose output (including statistics) to stderr.

- `--quiet`  
  Suppress non-essential messages printed to stderr.

### Example commands

Solve the small satisfiable instance:

```bash
python3 hamiltonian_sat.py --input instances/small_sat.txt
```

Solve the small unsatisfiable instance:

```bash
python3 hamiltonian_sat.py --input instances/small_unsat.txt
```

Generate CNF only:

```bash
python3 hamiltonian_sat.py     --input instances/small_sat.txt     --cnf-out small_sat.cnf     --only-cnf
```

Solve a larger instance and keep the CNF and solver statistics:

```bash
python3 hamiltonian_sat.py     --input instances/nontrivial_sat.txt     --cnf-out nontrivial.cnf     --show-stats
```

Solve the heaviest instance and see Glucose statistics:

```bash
python3 hamiltonian_sat.py     --input instances/heavy_sat_50.txt     --show-stats
```

---

## 4. Description of attached instances

All example instances are in the `instances/` directory.

### 4.1 `instances/small_sat.txt`

```text
# Small satisfiable instance: path 0-1-2-3
4 3
0 1
1 2
2 3
```

- Path graph on 4 vertices.
- There is a Hamiltonian path: `0 1 2 3` (and the reverse `3 2 1 0`).

### 4.2 `instances/small_unsat.txt`

```text
# Small unsatisfiable instance: vertex 2 is isolated
3 1
0 1
```

- Vertices: 0, 1, 2.
- Only one edge: `{0,1}`.
- Vertex 2 is isolated, so there is no Hamiltonian path.

### 4.3 `instances/nontrivial_sat.txt`

```text
# Nontrivial satisfiable instance
# 20 vertices, 24 edges
20 24
0 1
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
10 11
11 12
12 13
13 14
14 15
15 16
16 17
17 18
18 19
0 5
3 7
4 10
8 15
12 18
```

- 20 vertices, 24 edges.
- Contains a long “backbone” path `0–1–2–…–19` plus several chords.
- The instance is satisfiable and gives a larger CNF than the tiny toy graph, demonstrating that the encoding scales.

### 4.4 `instances/heavy_sat_50.txt`

```text
# Heavy satisfiable instance (50 vertices)
# Path backbone 0-1-2-...-49 plus several chords.
50 57
0 1
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10
10 11
11 12
12 13
13 14
14 15
15 16
16 17
17 18
18 19
19 20
20 21
21 22
22 23
23 24
24 25
25 26
26 27
27 28
28 29
29 30
30 31
31 32
32 33
33 34
34 35
35 36
36 37
37 38
38 39
39 40
40 41
41 42
42 43
43 44
44 45
45 46
46 47
47 48
48 49
0 10
5 15
10 20
15 25
20 30
25 35
30 40
35 45
```

- 50 vertices, 57 edges.
- Contains a backbone path `0–1–…–49` plus several chords.
- The CNF for this instance has 2,500 variables and 237,064 clauses.
- It is satisfiable and yields a Hamiltonian path such as `49 48 … 1 0`.

This is used as the heaviest satisfiable instance in the experiments.

---

## 5. Experiments and runtimes

The assignment asks for a report on experiments and, ideally, a satisfiable instance that runs at least 10 seconds (but at most 10 minutes). Actual runtimes naturally depend on hardware and on the Glucose build.

### Experimental setup

I tested the script on several instances:

- `small_sat.txt` (4 vertices, 3 edges),
- `small_unsat.txt` (3 vertices, 1 edge),
- `nontrivial_sat.txt` (20 vertices, 24 edges),
- `heavy_sat_50.txt` (50 vertices, 57 edges).

For each instance, I used:

```bash
time python3 hamiltonian_sat.py --input <instance> --show-stats
```

The script prints the graph size, CNF size, and the full Glucose statistics.

### Observations

- For the small instances (`small_sat.txt`, `small_unsat.txt`), CNF generation and solving are essentially instantaneous.
- For `nontrivial_sat.txt` (20 vertices), the CNF has 400 variables and 13,948 clauses. Glucose solves it well under one second.
- For the heaviest instance `heavy_sat_50.txt`:
  - The CNF has **2,500 variables** and **237,064 clauses**.
  - On my iMac with Glucose 4.2.1, the total runtime reported by `time` was approximately:
    - **0.434 seconds** (user ~0.37 s, system ~0.03 s, 92% CPU).
  - Glucose still solves this instance very quickly and prints a valid Hamiltonian path (`49 48 47 … 1 0`).

### About the “≥ 10 seconds” requirement

The assignment states:

> a nontrivial satisfiable instance that runs at least 10s (and max 10 mins), **if you cannot find it, describe what you tried**.

I experimented with several satisfiable instances of increasing size (up to 50 vertices and 237,064 clauses in the CNF). On my machine, even the largest instance `heavy_sat_50.txt` was solved in **significantly less than 10 seconds** (about 0.434 s of real time).

Despite increasing the graph size and CNF size, Glucose remained fast on these instances. Therefore, I did **not** obtain a satisfiable instance whose solving time is at least 10 seconds. Instead, I document the experiments and use `heavy_sat_50.txt` as a clearly nontrivial satisfiable instance that demonstrates how the encoding scales.

---

## 6. Implementation notes

- The script is written in **Python 3** and uses only the standard library:
  - `argparse` for command-line argument parsing,
  - `subprocess` to call Glucose,
  - `tempfile` and `os` for managing temporary CNF files,
  - basic type hints for clarity.
- CNF construction is implemented manually, without any CNF or SAT libraries, as required.
- The script automatically searches for `glucose` or `glucose-syrup` in `PATH`, or accepts an explicit path via `--glucose-cmd`.
- The option `--only-cnf` allows generating DIMACS CNF files without solving, and `--show-stats` prints the full solver statistics.

Overall, the project satisfies the assignment requirements:

1. It solves a decision version of the Hamiltonian Path problem.
2. It reads problem instances, encodes them into CNF (DIMACS), calls Glucose, and decodes the result.
3. It outputs solutions in a human-readable format.
4. It can output the CNF and solver statistics.
5. It includes small satisfiable and unsatisfiable instances, plus nontrivial satisfiable instances, and a report on experiments and runtimes.

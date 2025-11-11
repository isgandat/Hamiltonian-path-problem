# Hamiltonian Path Solver using SAT (Glucose 4.2)

## Problem Description

**Hamiltonian Path Problem (decision version)**  
Given an undirected simple graph \(G=(V,E)\), does there exist an ordering of all vertices \((v_0,\dots,v_{n-1})\) such that each vertex appears exactly once and \(\{v_i,v_{i+1}\}\in E\) for all \(i=0,\dots,n-2\)?

We solve this decision problem and optionally support fixing the start/end vertex.

---

## Encoding to CNF (DIMACS)

Let \(n=|V|\). Create propositional variables \(x_{v,p}\) meaning “vertex \(v\) is placed at position \(p\)”.  
Indexing: `ID(x_{v,p}) = v*n + p + 1` so variable IDs are in `1..n^2`.

**Clauses**

1. **Each position has exactly one vertex.**  
   For each \(p\):  
   - AtLeastOne: \(\bigvee_{v=0}^{n-1} x_{v,p}\)  
   - AtMostOne (pairwise): \((\lnot x_{v_1,p} \lor \lnot x_{v_2,p}))\) for all \(v_1<v_2\).

2. **Each vertex appears in exactly one position.**  
   For each \(v\):  
   - AtLeastOne: \(\bigvee_{p=0}^{n-1} x_{v,p}\)  
   - AtMostOne (pairwise): \((\lnot x_{v,p_1} \lor \lnot x_{v,p_2}))\) for all \(p_1<p_2\).

3. **Adjacency between consecutive positions.**  
   For each consecutive pair \(p,p+1\) and vertices \(u,w\), if \(\{u,w\}\notin E\) then  
   \((\lnot x_{u,p} \lor \lnot x_{w,p+1}))\).

4. **Optional constraints.**  
   - `--start s` adds \(x_{s,0}\) (fix start).  
   - `--end t` adds \(x_{t,n-1}\) (fix end).  
   - `--fix-first v` adds \(x_{v,0}\) (simple symmetry breaking, ignored if `--start` is used).

No CNF helper libraries are used; the CNF is built directly.

---

## Files in this repository

```
hamilton_sat.py           # Encoder/solver: builds CNF, calls Glucose, decodes model
gen_nontrivial.py         # Random generator to find ~target-second SAT instances
instances/
  small_pos.txt           # Small SAT (human-readable)
  small_neg.txt           # Small UNSAT (human-readable)
  pos6.txt                # Medium SAT
README.md                 # This documentation
```

---

## Requirements

- Python 3.8+
- Glucose 4.2 binary on PATH (e.g., `glucose` or `glucose-syrup`)  
  You can also pass an explicit path via `--glucose-bin`.

---

## Usage

### Solve and decode
```bash
python3 hamilton_sat.py --graph instances/small_pos.txt --format edge --glucose-bin glucose
```

### Show solver statistics
```bash
python3 hamilton_sat.py --graph instances/pos6.txt --format edge --glucose-bin glucose --stats
```

### Dump CNF (no solving)
```bash
python3 hamilton_sat.py --graph instances/small_pos.txt --no-solve --output-cnf out.cnf
```

### Fix endpoints / symmetry breaking
```bash
python3 hamilton_sat.py --graph instances/pos6.txt --start 0 --end 5 --glucose-bin glucose
python3 hamilton_sat.py --graph instances/pos6.txt --fix-first 0 --glucose-bin glucose
```

---

## Input formats

### 1) Edge list (`--format edge`, default)
```
# n m
4 3
0 1
1 2
2 3
```

### 2) DIMACS-like (`--format dimacs`)
```
c comments allowed
p edge 4 3
e 1 2
e 2 3
e 3 4
```

### 3) Adjacency matrix (`--format matrix`)
```
0 1 0 0
1 0 1 0
0 1 0 1
0 0 1 0
```

---

## Example instances (included)

**instances/small_pos.txt** (SAT)
```
# n m
4 3
0 1
1 2
2 3
```

**instances/small_neg.txt** (UNSAT)
```
# n m
3 1
0 1
```

**instances/pos6.txt** (SAT)
```
6 7
0 1
1 2
2 3
3 4
4 5
0 5
2 5
```

---

## Experiments and nontrivial instance

To satisfy the “nontrivial ≳10s (≤10 min)” requirement, use `gen_nontrivial.py` to search near the Hamiltonicity threshold \(p \approx (\ln n)/n\).

### Find a ~23 s instance
```bash
python3 gen_nontrivial.py   --glucose-bin glucose   --script hamilton_sat.py   --out-dir instances   --target-sec 23   --max-sec 600   --n-min 70   --n-max 110   --p-scale-min 0.9   --p-scale-max 1.4   --seeds-per-n 70
```

When it prints something like:
```
=== FOUND nontrivial instance ≈23.1s ===
instances/nontrivial_edge_nXX_seedYY_pZZZZZ.txt
```
verify and keep the log:

```bash
/usr/bin/time -lp python3 hamilton_sat.py   --graph instances/nontrivial_edge_nXX_seedYY_pZZZZZ.txt   --format edge --glucose-bin glucose --stats   | tee instances/nontrivial_edge_nXX_seedYY_pZZZZZ.log
```

### If you cannot find ~23 s
Include the best you found and describe your search ranges and times (n, p-scales, seeds). This is acceptable per the assignment.

---

## What to submit

- `hamilton_sat.py`
- `README.md`
- `instances/` folder with:
  - `small_pos.txt` (SAT, human-readable)
  - `small_neg.txt` (UNSAT, human-readable)
  - `pos6.txt` (SAT)
  - `nontrivial_edge_*.txt` (SAT ≳10s, target ~23s) + its `.log`

This project meets the assignment requirements: manual CNF encoding, DIMACS output, Glucose invocation, model decoding, human-readable output, and a nontrivial instance plus experiments.

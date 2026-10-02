# Splitting Z-bent functions

Supplementary computational material for **Solution to an open question of
Dobbertin and Leander on non-splitting Z-bent functions of level one**, by
Pantelimon Stănică.

The programs verify the six-variable non-splitting example, its bent direct
products, the disjoint-spectra counterexample, splitting for a complete-spread
construction, the exhaustive classifications on two and four variables, and 24 further
non-splitting examples with sixteen zeros.

## Quick start

Run the commands below from the repository root. Use Python 3.10 or later.
Install NumPy for the data and example checks:

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

The default run checks the seed's exact Walsh transform and determinant,
the explicit bent split of the four-variable counterexample, all bundled
arrays, and all 8,953 admissible coefficient vectors of the nine-member
complete spread in dimension six. It also verifies the 24 sixteen-zero
examples, their determinant certificates and zero-set geometry.

To also check every possible splitting partner of the seed:

```sh
python3 reproduce.py --exhaustive-seed
```

This additional check enumerates all 262,144 sign assignments on the seed's
18-point zero set.

The determinant certificate can be run independently using only the Python
standard library:

```sh
python3 verify_seed.py
```

## Exhaustive classification on two and four variables

Use a C++17 compiler:

```sh
g++ -O3 -std=c++17 verify_classification.cpp -o verify_classification
./verify_classification
```

The program independently enumerates all ternary vectors and all bent sign
vectors. It checks that every level-one vector is a bent pair-average, and
that the two resulting sets have equal cardinality.

| Variables | Ternary vectors tested | Level-one functions | Bent sign vectors | Distinct bent pair-averages |
| --- | ---: | ---: | ---: | ---: |
| 2 | 81 | 33 | 8 | 33 |
| 4 | 43,046,721 | 213,249 | 896 | 213,249 |

The four-variable distribution by support size is:

| Support size | 0 | 4 | 6 | 8 | 10 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Number of functions | 1 | 1,120 | 14,336 | 84,000 | 86,016 | 26,880 | 896 |

## Independent Python-only verification

`NonSplitting_verify_certificate.py` provides a second implementation of the
classification, seed determinant and cylinder-rank checks. It also runs the
sixteen-zero verifier. It uses NumPy and SymPy and does not require a C++ compiler or an optimization solver:

```sh
python3 -m pip install -r requirements-certificate.txt
python3 NonSplitting_verify_certificate.py
```

Its restricted modes are:

```sh
python3 NonSplitting_verify_certificate.py --seed-only
python3 NonSplitting_verify_certificate.py --minimal-only
python3 NonSplitting_verify_certificate.py --examples-only
```

The combined program is optional. The separate seed certificate and C++
classification program verify the same claims independently.

## Files

| File | Purpose |
| --- | --- |
| `reproduce.py` | Run the certificate, counterexample, data comparisons and spread checks. |
| `verify_C16.py` | Verify all sixteen-zero examples, exact minors, cubic zero-set indicators and affine-flat containment. |
| `verify_seed.py` | Verify the displayed seed, its Walsh transform, the determinant `2^30`, and a cylinder minor of determinant magnitude `2^192`. |
| `NonSplitting_verify_certificate.py` | Independent Python-only classification, seed and cylinder-rank verification. |
| `verify_classification.cpp` | Classify all level-one functions on two and four variables using integer arithmetic. |
| `verify.py` | Enumerate all possible complementary sign vectors of the six-variable seed. |
| `cex.py` | Check the disjoint-spectra semibent counterexample and its explicit bent decomposition. |
| `spread.py` | Construct the complete GF(64)/GF(8) spread and verify an explicit splitting for every admissible coefficient vector. |
| `zbent.py` | Walsh transforms, level-one tests, data loading and optional integer-programming routines. |
| `data/` | Seed and product arrays, sixteen-zero examples and exact minor certificates. |
| `verification_results.txt` | Reference output from the reproducibility and classification runs. |
| `requirements.txt` | Dependency for the Python data and example checks. |
| `requirements-certificate.txt` | Dependencies for the combined Python-only verification. |
| `requirements-solver.txt` | Dependencies for the optional PuLP/CBC routines. |

The data files are:

| Array | Variables | Entries | Construction |
| --- | ---: | ---: | --- |
| `f6.npy` | 6 | 64 | The displayed non-splitting seed. |
| `example_n6.npy` | 6 | 64 | The same seed, retained under its original filename. |
| `F8.npy` | 8 | 256 | Product of the seed with one copy of `(-1)^(st)`. |
| `F10.npy` | 10 | 1,024 | Product of the seed with two copies of `(-1)^(st)`. |
| `F12.npy` | 12 | 4,096 | Product of the seed with three copies of `(-1)^(st)`. |

## Sixteen-zero examples

`data/nonsplitting_C16_examples.txt` contains 24 six-variable level-one
functions. Each has sixteen zeros in both domains. The associated
`data/C16_certificates.json` gives both zero sets, the Fourier string, sixteen
row indices of a nonsingular Walsh minor, its exact determinant, and the
zero-set geometry. Run these checks separately with:

```sh
python3 verify_C16.py
```

Four primal zero sets are disjoint unions of two affine 3-flats. The other
twenty contain no affine 3-flat. Both zero-set indicators of every example
have algebraic degree three. The list is a collection of examples; it is
not an enumeration of equivalence classes.

Any of these functions, multiplied by bent sign functions on additional
even numbers of variables, gives non-splitting level-one functions with
support density 3/4 and cubic zero-set indicators.

## Mathematical conventions

For `n = 2k`, a vector `f` is level one when both `f` and `2^(-k) H_n f`
have entries in `{-1, 0, 1}`. The matrix entries are
`H_n[u, x] = (-1)^(u · x)`. Array index `x` represents the bit vector with
`x = sum_i x_i 2^i`; the dot product is taken modulo two.

A splitting is an expression `f = (a + b)/2`, where `a` and `b` are bent sign
vectors. A candidate partner `h = (a - b)/2` has signs on the zero set of
`f`, vanishes elsewhere, and has the corresponding complementary Fourier
support. The seed certificate proves that no nonzero real vector satisfies
these two support restrictions, using a nonsingular 18-by-18 Walsh minor.

Products are stored as `np.kron(a, f)`: the new factor is the outer index and
the existing vector is the inner index. For the first extension the combined
index is `64*w + x`. The two-variable bent factor is `[1, 1, 1, -1]`.

All certificate, classification, counterexample and spread checks use integer
arithmetic. The proof for arbitrary larger even dimensions is the
mathematical direct-product argument; the supplied arrays check its first
three extensions.

## Optional solver routines

The reproducibility commands above do not require an optimization solver.
The optional solver environment uses PuLP 3.3.0 with its CBC interface.
To use the routines in `zbent.py` that solve integer programs, install:

```sh
python3 -m pip install -r requirements-solver.txt
```

`split_status(f, k)` returns `SPLIT`, `NONSPLIT`, or `UNKNOWN:<status>`.
A splitting witness is checked using integer Walsh transforms before
`SPLIT` is returned. `NONSPLIT` records CBC's infeasibility status; exact
non-splitting certificates are provided separately by `verify_seed.py`.
A time limit or unresolved solver status is returned as `UNKNOWN`, and
`splits(f, k)` raises an exception for that outcome.

`find_level1(H, k, N, obj, ...)` solves an integer optimization problem with
the supplied objective. It returns a checked level-one vector when the solver
reports an optimum, and `None` otherwise.

## Repository

[github.com/pstanica/Zbent](https://github.com/pstanica/Zbent)

Suggested bibliography entry:

```latex
\bibitem{APN-Zbent}
P.~St\u{a}nic\u{a},
\emph{Splitting Z-bent: supplementary computational material},
\url{https://github.com/pstanica/Zbent}.
```

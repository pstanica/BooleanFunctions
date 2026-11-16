# BentMutations

Verification programs for *Lifting rotation symmetric bent functions outside the completed Maiorana--McFarland class* by William Casey and Pantelimon Stănică. Place this folder at the root of the [BooleanFunctions repository](https://github.com/pstanica/BooleanFunctions).

## Run all checks

Use Python 3.10 or later, NumPy, and a C++20 compiler such as GCC or Clang. From the repository root, run

```sh
cd BentMutations
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

The runner builds the C++ program in a temporary directory and stops if any check fails. To choose a compiler, run `CXX=g++ python run_all.py` or `CXX=clang++ python run_all.py`. Keep assertions enabled. In particular, run Python without `-O` and compile C++ without `-DNDEBUG`.

The expected output is in [expected_output.txt](expected_output.txt). To check a run against it in a POSIX shell, use

```sh
python run_all.py > results.txt && diff -u expected_output.txt results.txt
```

## Programs

| Source | Checks |
| --- | --- |
| [verify_seed_and_lift.py](verify_seed_and_lift.py) | Checks the ten-variable seed decomposition, complementary Walsh supports, tensor radical and centroid, and a totally isotropic plane. Tests the twenty-variable lifts of the matching, complete graph, and four-cycle links. NumPy is used for the twenty-variable checks. |
| [enumerate_isotropic_subspaces.py](enumerate_isotropic_subspaces.py) | Enumerates the totally isotropic subspaces of the ten-variable seed tensor and checks that every four-dimensional one equals its common contraction kernel. |
| [enumerate_circulant_graphs.py](enumerate_circulant_graphs.py) | Counts connected circulant graphs with nonsingular binary adjacency matrices up to isomorphism on 2, 4, 6, 8, 10, 12, and 14 vertices. |
| [verify_su8.py](verify_su8.py) | Checks all 256 Walsh coefficients of Su's eight-variable function and tests every pair of independent directions for a constant second difference. |
| [enumerate_quartic_replacements.cpp](enumerate_quartic_replacements.cpp) | Verifies the eight-variable quartic crossing pair, enumerates the normalized quartic rotation symmetric bent functions, tests completed Maiorana--McFarland membership, and counts the single-orbit replacement edges. |

Each Python program can also be run directly with `python filename.py`. To run only the C++ check, use

```sh
c++ -std=c++20 -O3 -Wall -Wextra -pedantic enumerate_quartic_replacements.cpp -o enumerate_quartic_replacements
./enumerate_quartic_replacements
```

The counts concern functions in fixed cyclic coordinates with their affine terms removed. A crossing edge is an unordered pair of adjacent functions with different membership in the completed Maiorana--McFarland class. To count directed orbit replacements, the program directs every crossing edge from the member to the nonmember, then counts distinct ordered pairs of removed and inserted orbits. No external data files are required.

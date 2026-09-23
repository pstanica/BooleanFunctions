"""
Exact, floating-point-free verification for
  "On non-splitting Z-bent functions of level one:
   a correction, an exact criterion, and an infinite family".

The script verifies the finite certificate used in the proof:
  1. the displayed six-variable seed and its normalized Walsh transform;
  2. the two zero sets C and D;
  3. the exact determinant det W_{R,C} = 2^30;
  4. the resulting full-column-rank support-uncertainty certificate;
  5. as an independent check, full cylinder rank in dimension 8.

Run:
    python3 NonSplitting_verify_certificate_clean.py

Requires: numpy, sympy.
All certification steps use integer arithmetic; no floating-point rank test or
optimization solver is used.
"""
import numpy as np
from sympy import Matrix, Integer


def dot(u, x):
    return (u & x).bit_count() & 1


def main():
    seed = (
        "0-+0---+-00--+-++++++00+--00-0+0++0--0++--0++0++0+-+++-0+0++-+0-"
    )
    f = np.array([{"0": 0, "+": 1, "-": -1}[c] for c in seed], dtype=int)
    assert len(f) == 64

    # Unnormalized Walsh matrix H_6(u,x)=(-1)^{u.x}.
    H6 = np.array(
        [[(-1) ** dot(u, x) for x in range(64)] for u in range(64)], dtype=int
    )
    Hf = H6 @ f
    assert set(np.unique(Hf).tolist()) <= {-8, 0, 8}
    fhat = Hf // 8

    expected_fhat = (
        "+--00+00+0+++0+--0-0-++--+--++++-0-00+0-++-++0-0--+0++-0-0-+---0"
    )
    actual_fhat = "".join({0: "0", 1: "+", -1: "-"}[int(v)] for v in fhat)
    assert actual_fhat == expected_fhat
    print("[1] f and f-hat are {-1,0,1}-valued: OK")

    C = [x for x in range(64) if f[x] == 0]
    D = [u for u in range(64) if fhat[u] == 0]
    assert C == [0, 3, 9, 10, 21, 22, 26, 27, 29, 31, 34, 37, 42, 45, 48, 55, 57, 62]
    assert D == [3, 4, 6, 7, 9, 13, 17, 19, 33, 35, 36, 38, 45, 47, 51, 55, 57, 63]
    print("[2] zero sets C and D, each of size 18: OK")

    R = [0, 1, 2, 5, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 24, 32, 40]
    Dc = [u for u in range(64) if u not in set(D)]
    assert set(R) <= set(Dc)
    assert len(R) == len(set(R)) == len(C) == 18

    WRC = Matrix(
        18,
        18,
        lambda i, j: Integer((-1) ** dot(R[i], C[j])),
    )
    det = WRC.det(method="bareiss")
    assert det == 2**30
    print("[3] det W_{R,C} = 2^30 exactly: OK")

    Wfull = Matrix(
        len(Dc),
        len(C),
        lambda i, j: Integer((-1) ** dot(Dc[i], C[j])),
    )
    assert Wfull.rank() == 18
    print("[4] rank W_{D^c,C} = 18: OK")

    # Independent finite check of the cylinder statement for m=2.
    cols = [(w, x) for w in range(4) for x in C]
    rows = [(u, v) for u in range(4) for v in Dc]
    A = Matrix(
        len(rows),
        len(cols),
        lambda i, j: Integer(
            (-1)
            ** (
                (dot(rows[i][0], cols[j][0]) + dot(rows[i][1], cols[j][1]))
                & 1
            )
        ),
    )
    assert A.rank() == len(cols) == 72
    print("[5] dimension-8 cylinder rank is 72 (full column rank): OK")

    print("All exact certificate checks passed.")


if __name__ == "__main__":
    main()

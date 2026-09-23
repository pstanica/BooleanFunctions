"""
Exact, floating-point-free verification for the paper
  "Solution to an open question of Dobbertin and Leander
   on non-splitting Z-bent functions of level one".

Default run verifies all finite computations used in the paper:
  (1) every level-one Z-bent function in dimensions 2 and 4 is splitting;
  (2) the six-variable non-splitting seed and its Walsh-minor argument;
  (3) the cylinder-rank check in dimension 8.

Run:
    python3 NonSplitting_verify_certificate.py

Options:
    --minimal-only   verify only the dimensions 2 and 4 classification
    --seed-only      verify only the six-variable seed and cylinder rank

Requires: numpy, sympy.  All mathematical tests use exact integer arithmetic;
no floating-point rank test and no integer-programming solver is used.
"""

import sys
import numpy as np
from sympy import Matrix, Integer


def dot(u, x):
    return (u & x).bit_count() & 1


def walsh_matrix(n):
    """Unnormalized 2^n x 2^n Walsh matrix with entries +/-1."""
    N = 1 << n
    return np.array(
        [[-1 if dot(u, x) else 1 for x in range(N)] for u in range(N)],
        dtype=np.int16,
    )


def enumerate_level_one_codes(n, chunk_size=1_000_000):
    """
    Enumerate exactly all f in {-1,0,1}^{2^n} such that H_n f has entries
    in {0,+/-2^{n/2}}.  A ternary vector is encoded by its base-3 integer
    with digits f(x)+1 in the coordinate order x=0,...,2^n-1.
    """
    N = 1 << n
    scale = 1 << (n // 2)
    H = walsh_matrix(n)
    total = 3 ** N
    valid_codes = []
    support_distribution = np.zeros(N + 1, dtype=np.int64)

    for start in range(0, total, chunk_size):
        m = min(chunk_size, total - start)
        codes = np.arange(start, start + m, dtype=np.int64)
        q = codes.copy()
        f = np.empty((m, N), dtype=np.int8)
        for x in range(N):
            f[:, x] = (q % 3) - 1
            q //= 3

        Hf = f.astype(np.int16) @ H.T
        ok = np.all((Hf == 0) | (Hf == scale) | (Hf == -scale), axis=1)
        if np.any(ok):
            valid_codes.append(codes[ok])
            weights = np.count_nonzero(f[ok], axis=1)
            support_distribution += np.bincount(weights, minlength=N + 1)

    return np.concatenate(valid_codes), support_distribution


def bent_sign_vectors(n):
    """Enumerate exactly all sign vectors of bent functions in dimension n."""
    N = 1 << n
    scale = 1 << (n // 2)
    H = walsh_matrix(n)
    masks = np.arange(1 << N, dtype=np.uint32)
    signs = np.empty((1 << N, N), dtype=np.int8)
    for x in range(N):
        signs[:, x] = np.where(((masks >> x) & 1) != 0, 1, -1)
    Hs = signs.astype(np.int16) @ H.T
    return signs[np.all(np.abs(Hs) == scale, axis=1)]


def distinct_split_codes(bent):
    """Base-3 codes of all distinct averages (a+b)/2 of bent sign vectors."""
    N = bent.shape[1]
    powers = 3 ** np.arange(N, dtype=np.int64)
    blocks = []
    for i in range(len(bent)):
        averages = (bent[i] + bent[i:]) // 2
        blocks.append((averages.astype(np.int64) + 1) @ powers)
    return np.unique(np.concatenate(blocks))


def minimal_dimension_check():
    expected = {
        2: {
            "level_one": 33,
            "bent": 8,
            "support": {0: 1, 2: 24, 4: 8},
        },
        4: {
            "level_one": 213249,
            "bent": 896,
            "support": {
                0: 1,
                4: 1120,
                6: 14336,
                8: 84000,
                10: 86016,
                12: 26880,
                16: 896,
            },
        },
    }

    for n in (2, 4):
        level_one, dist = enumerate_level_one_codes(n)
        bent = bent_sign_vectors(n)
        split = distinct_split_codes(bent)
        support = {i: int(v) for i, v in enumerate(dist) if v}

        assert len(level_one) == expected[n]["level_one"]
        assert len(bent) == expected[n]["bent"]
        assert support == expected[n]["support"]
        assert len(split) == len(level_one)
        assert np.array_equal(split, level_one)

        print(
            f"[n={n}] BF_1 has {len(level_one)} functions; "
            f"{len(bent)} bent sign vectors; all level-one functions split : OK"
        )
        print(f"      support-size distribution: {support}")

    print("--> dimension 6 is the minimum possible dimension for non-splitting at level 1.")


def seed_and_rigidity_check():
    s = "0-+0---+-00--+-++++++00+--00-0+0++0--0++--0++0++0+-+++-0+0++-+0-"
    f = np.array([{'0': 0, '+': 1, '-': -1}[c] for c in s], dtype=int)

    H6 = walsh_matrix(6).astype(int)
    Hf = H6.dot(f)
    assert set(np.unique(Hf).tolist()) <= {-8, 0, 8}
    fhat = Hf // 8
    assert "".join({0: '0', 1: '+', -1: '-'}[v] for v in fhat) == \
        "+--00+00+0+++0+--0-0-++--+--++++-0-00+0-++-++0-0--+0++-0-0-+---0"
    print("[seed 1] f and hat(f) are {-1,0,1}-valued -> f in BF_1^3 : OK")

    C = [x for x in range(64) if f[x] == 0]
    D = [u for u in range(64) if fhat[u] == 0]
    assert C == [0, 3, 9, 10, 21, 22, 26, 27, 29, 31, 34, 37, 42, 45, 48, 55, 57, 62]
    assert D == [3, 4, 6, 7, 9, 13, 17, 19, 33, 35, 36, 38, 45, 47, 51, 55, 57, 63]
    print("[seed 2] zero sets C,D match the paper (|C|=|D|=18) : OK")

    R = [0, 1, 2, 5, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 24, 32, 40]
    assert set(R) <= (set(range(64)) - set(D)) and len(set(R)) == 18
    W = Matrix(18, 18, lambda i, j: Integer((-1) ** dot(R[i], C[j])))
    assert W.det() == 2 ** 30
    print("[seed 3] R subset D^c, det W_{R,C}=2^30 (exact integer) : OK")

    Dc = [u for u in range(64) if u not in set(D)]
    rank = Matrix(
        len(Dc), 18, lambda i, j: Integer((-1) ** dot(Dc[i], C[j]))
    ).rank()
    assert rank == 18
    print("[seed 4] support rigidity on F_2^6 (rank 18) : OK")

    cols = [(w, x) for w in range(4) for x in C]
    rows = [(u, v) for u in range(4) for v in Dc]
    A = Matrix(
        len(rows),
        len(cols),
        lambda i, j: Integer(
            (-1) ** ((dot(rows[i][0], cols[j][0]) + dot(rows[i][1], cols[j][1])) & 1)
        ),
    )
    assert A.rank() == len(cols) == 72
    print("[seed 5] cylinder rigidity at n=8 (rank 72 = full) : OK")
    print("--> the seed yields non-splitting level-one Z-bent functions in every even n>=6.")


def main():
    minimal_only = "--minimal-only" in sys.argv
    seed_only = "--seed-only" in sys.argv
    if minimal_only and seed_only:
        raise SystemExit("Choose at most one of --minimal-only and --seed-only.")
    if not seed_only:
        minimal_dimension_check()
    if not minimal_only:
        seed_and_rigidity_check()


if __name__ == "__main__":
    main()

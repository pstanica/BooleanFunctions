"""
Exact, floating-point-free verification for
  "On non-splitting Z-bent functions of level one:
   a correction, an exact criterion, and an infinite family".

Default run checks the finite certificate underlying the infinite family
(Theorem 4.1 / Corollary 4.3 and the cylinder rank at n=8):
    python3 NonSplitting_verify_certificate.py

Optional (a few minutes) also verifies Proposition 4.x, that all 8953
level-one PS_ap functions on F_2^6 split:
    python3 NonSplitting_verify_certificate.py --psap

Requires: numpy, sympy  (exact integer arithmetic only).
"""
import sys, numpy as np
from sympy import Matrix, Integer

def dot(u, x): return bin(u & x).count("1") & 1

def main_certificate():
    s = "0-+0---+-00--+-++++++00+--00-0+0++0--0++--0++0++0+-+++-0+0++-+0-"
    f = np.array([{'0':0,'+':1,'-':-1}[c] for c in s], int)

    # exact Walsh transform (Sylvester--Hadamard, entries +-1) and level-one check
    H6 = np.array([[(-1)**dot(u, x) for x in range(64)] for u in range(64)], int)
    Hf = H6.dot(f)
    assert set(np.unique(Hf).tolist()) <= {-8, 0, 8}
    fhat = Hf // 8
    assert "".join({0:'0',1:'+',-1:'-'}[v] for v in fhat) == \
        "+--00+00+0+++0+--0-0-++--+--++++-0-00+0-++-++0-0--+0++-0-0-+---0"
    print("[1] f, hat f are {-1,0,1}-valued  ->  f in BF_1^3 : OK")

    C = [x for x in range(64) if f[x] == 0]
    D = [u for u in range(64) if fhat[u] == 0]
    assert C == [0,3,9,10,21,22,26,27,29,31,34,37,42,45,48,55,57,62]
    assert D == [3,4,6,7,9,13,17,19,33,35,36,38,45,47,51,55,57,63]
    print("[2] zero sets C, D match the paper (|C|=|D|=18) : OK")

    R = [0,1,2,5,8,10,11,12,14,15,16,18,20,21,22,24,32,40]
    assert set(R) <= (set(range(64)) - set(D)) and len(set(R)) == 18
    W = Matrix(18, 18, lambda i, j: Integer((-1)**dot(R[i], C[j])))
    assert W.det() == 2**30
    print("[3] R subset D^c, det W_{R,C} = 2^30 (exact integer) : OK")

    Dc = [u for u in range(64) if u not in set(D)]
    assert Matrix(len(Dc), 18, lambda i, j: Integer((-1)**dot(Dc[i], C[j]))).rank() == 18
    print("[4] real support-uncertainty on F_2^6 (rank 18) : OK")

    cols = [(w, x) for w in range(4) for x in C]
    rows = [(u, v) for u in range(4) for v in Dc]
    A = Matrix(len(rows), len(cols),
               lambda i, j: Integer((-1)**((dot(rows[i][0], cols[j][0]) +
                                            dot(rows[i][1], cols[j][1])) & 1)))
    assert A.rank() == len(cols) == 72
    print("[5] cylinder rigidity at n=8 (rank 72 = full) : OK")
    print("--> non-splitting level-1 Z-bent functions exist in every even n>=6.")

def psap_check():
    """Proposition: all level-1 PS_ap functions on F_2^6 (Desarguesian spread) split."""
    import itertools, pulp
    N = 64
    def gmul(a, b):
        p = 0
        for _ in range(6):
            if b & 1: p ^= a
            b >>= 1; hi = a & 0x20; a = (a << 1) & 0x3f
            if hi: a ^= 0x03          # x^6 = x + 1
        return p
    pw = [1]
    for _ in range(62): pw.append(gmul(pw[-1], 2))
    GF8 = [0] + [pw[(9*j) % 63] for j in range(7)]
    spread = [sorted(set(gmul(pw[i % 63], t) for t in GF8)) for i in range(9)]
    assert all(len(V) == 8 for V in spread)
    assert all(set(spread[i]) & set(spread[j]) == {0} for i in range(9) for j in range(9) if i < j)
    phis = []
    for V in spread:
        v = np.zeros(N, int); v[V] = 1; phis.append(v)
    H6 = np.array([[(-1)**dot(u, x) for x in range(64)] for u in range(64)], int)

    def splits(f):
        z = (H6.dot(f)) // 8
        Sz = [x for x in range(N) if f[x] == 0]
        pr = pulp.LpProblem("s", pulp.LpMinimize)
        b = {x: pulp.LpVariable(f"b{x}", cat="Binary") for x in Sz}
        fp = [(2*b[x]-1) if f[x] == 0 else 0 for x in range(N)]
        for u in range(N):
            e = pulp.lpSum(int((-1)**dot(u, x))*fp[x] for x in Sz)
            if z[u] != 0: pr += e == 0
            else:
                w = pulp.LpVariable(f"w{u}", cat="Binary"); pr += e == 8*(2*w-1)
        pr += 0
        return pulp.LpStatus[pr.solve(pulp.PULP_CBC_CMD(msg=0))] == "Optimal"

    cnt = ns = 0
    for c in itertools.product([-1,0,1], repeat=9):
        if sum(c) not in (-1,0,1): continue
        f = sum(ci*p for ci, p in zip(c, phis))
        if not np.all(np.isin(f, [-1,0,1])): continue
        if set(np.unique(H6.dot(f)).tolist()) - {-8,0,8}: continue
        cnt += 1
        if (f != 0).sum() and not splits(f): ns += 1
    assert cnt == 8953 and ns == 0
    print(f"[PSap] {cnt} level-1 PS_ap functions on F_2^6, non-splitting = {ns} : OK")

if __name__ == "__main__":
    main_certificate()
    if "--psap" in sys.argv:
        psap_check()

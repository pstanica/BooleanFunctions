# NumPy is used only for the k = 2 lift check.
from itertools import combinations, product

n = 10
def orbit(S, m=n):
    return {tuple(sorted((i + t) % m for i in S)) for t in range(m)}

b = set()
for S in [(0, 5), (0, 1, 4), (0, 1, 8), (0, 2, 4), (0, 2, 7)]:
    b ^= orbit(S)
link = {tuple(sorted((i, j))) for i in range(1, n, 2) for j in range(0, n, 2)}
bt = b ^ link                      # b~ = b + l_ev * l_odd
assert link == orbit((0, 1)) | orbit((0, 3)) | orbit((0, 5))

def ev(monos, x):
    return sum(all((x >> i) & 1 for i in m) for m in monos) & 1
B  = [ev(b, x) for x in range(1 << n)]
Bt = [ev(bt, x) for x in range(1 << n)]
bit = lambda x, i: (x >> i) & 1
l_odd = lambda x: sum(bit(x, i) for i in range(1, n, 2)) & 1
l_ev  = lambda x: sum(bit(x, i) for i in range(0, n, 2)) & 1
rot = lambda x: ((x << 1) | (x >> (n - 1))) & ((1 << n) - 1)   # (Rx)_i = x_{i-1}
def v(a):
    A = a.bit_count() & 1
    return sum(((a >> j & 1) << (2*j)) |
               (((a >> j & 1) ^ A) << (2*j+1)) for j in range(5))
def g(a):
    aa = [(a >> j) & 1 for j in range(5)]
    q = sum(aa[j] * aa[(j+1) % 5] for j in range(5))
    e3 = sum(aa[i] * aa[j] * aa[l]
             for i, j, l in combinations(range(5), 3))
    return (q + e3) & 1
assert all(Bt[v(a)] == g(a) for a in range(32))
# coset identities (eq:cosets) behind the direct proof of (d)
e2 = lambda a: sum(((a>>i)&1)*((a>>j)&1) for i, j in combinations(range(5), 2)) & 1
q5 = lambda a: sum(((a>>j)&1)*((a>>((j+1)%5))&1) for j in range(5)) & 1
Hk = [a for a in range(32) if a.bit_count() % 2 == 0]
assert all(g(h) == q5(h) and g(31 ^ h) == (1 + q5(h) + e2(h)) & 1 for h in Hk)
assert all(Bt[rot(v(c)) ^ v(a)] == (g(c) ^ g(a))
           for a in range(32) for c in range(32))

# b~ and b are rotation symmetric
assert all(Bt[rot(x)] == Bt[x] and B[rot(x)] == B[x] for x in range(1 << n))

# V2 = {x : x0+x1 = x2+x3 = ... = x8+x9, l_odd(x) = 0},  V1 = R(V2)
pair = lambda x, j: bit(x, 2 * j) ^ bit(x, 2 * j + 1)
V2 = [x for x in range(1 << n) if len({pair(x, j) for j in range(5)}) == 1 and l_odd(x) == 0]
V1 = sorted(rot(x) for x in V2)
assert len(V1) == len(V2) == 32 and set(V1) & set(V2) == {0}
assert sorted(rot(x) for x in V1) == sorted(V2)             # R swaps V1 and V2
assert all(l_ev(x) == 0 for x in V1) and all(l_odd(x) == 0 for x in V2)
assert V1 == sorted(x for x in range(1 << n)
                    if len({bit(x, (2*j+1) % n) ^ bit(x, (2*j+2) % n) for j in range(5)}) == 1
                    and l_ev(x) == 0)                        # explicit equations for V1

# Additivity: b~(v1+v2) = b~(v1) + b~(v2), and b~(0) = 0
assert Bt[0] == 0
assert all(Bt[u ^ w] == Bt[u] ^ Bt[w] for u in V1 for w in V2)

# Semi-bent with complementary spectra: for every alpha exactly one of
# W_g(alpha), W_g(alpha + l_odd) is nonzero, and it equals +-8   (g = b~|V1)
dot = lambda a, x: bin(a & x).count("1") & 1
Wg = lambda a: sum((-1) ** (Bt[v] ^ dot(a, v)) for v in V1)
LODD = sum(1 << i for i in range(1, n, 2))
for a in range(1 << n):
    w0, w1 = abs(Wg(a)), abs(Wg(a ^ LODD))
    assert {w0, w1} == {0, 8}

# Cubic tensor restricted to V1: T(a,c,d) = D_a D_c D_d b(0)
def T(a, c, d):
    s = 0
    for e in range(8):
        v = (a if e & 1 else 0) ^ (c if e & 2 else 0) ^ (d if e & 4 else 0)
        s ^= B[v]
    return s
basis = []
for v in V1:
    span = {0}
    for u in basis:
        span |= {s ^ u for s in span}
    if v not in span:
        basis.append(v)
assert len(basis) == 5
def comb(c):  # coordinates in F2^5 -> vector of V1
    x = 0
    for i in range(5):
        if (c >> i) & 1:
            x ^= basis[i]
    return x
t = [[[T(basis[i], basis[j], basis[k]) for k in range(5)] for j in range(5)] for i in range(5)]
# radical is zero
rad = [a for a in range(1, 32) if all(T(comb(a), comb(c), comb(d)) == 0
                                      for c in range(32) for d in range(32))]
assert rad == []
# centroid = F2 : solve the linear system for A in M_5(F2)
rows = []
for i, j, k in product(range(5), repeat=3):
    for pos in (1, 2):              # T(Ae_i,e_j,e_k) = T(e_i,Ae_j,e_k) and = T(e_i,e_j,Ae_k)
        r = 0
        for m in range(5):
            if t[m][j][k]: r ^= 1 << (m * 5 + i)
            if pos == 1 and t[i][m][k]: r ^= 1 << (m * 5 + j)
            if pos == 2 and t[i][j][m]: r ^= 1 << (m * 5 + k)
        if r: rows.append(r)
piv = {}
for r in rows:
    while r:
        h = r.bit_length() - 1
        if h in piv:
            r ^= piv[h]
        else:
            piv[h] = r
            break
assert 25 - len(piv) == 1           # centroid is 1-dimensional, i.e. F2 * identity
# a 2-dimensional totally isotropic subspace of V1
iso2 = [(a, c) for a, c in combinations(V1[1:], 2)
        if all(T(a, c, d) == 0 for d in range(1 << n))]
assert iso2
a0, c0 = iso2[0]
print("V1 isotropic pair:", format(a0, "010b")[::-1], format(c0, "010b")[::-1])

# Check the lift at n = 20 (k = 2)
import numpy as np
k, N = 2, 20
idx = np.arange(1 << N, dtype=np.uint32)
X = [((idx >> i) & 1).astype(np.uint8) for i in range(N)]
def tt(monos, lin=None):
    f = np.zeros(1 << N, dtype=np.uint8)
    for m in monos:
        z = np.ones(1 << N, dtype=np.uint8)
        for i in m: z &= (lin[i] if lin else X[i])
        f ^= z
    return f
s = [np.zeros(1 << N, dtype=np.uint8) for _ in range(2 * k)]
for tt_ in range(N): s[tt_ % (2 * k)] ^= X[tt_]
base = tt({tuple(sorted(k * i + r for i in m)) for r in range(k) for m in bt})
def walsh_abs_ok(f):
    a = 1 - 2 * f.astype(np.int32)
    h = 1
    while h < a.size:
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], 1).reshape(-1)
        h *= 2
    return bool(np.all(np.abs(a) == 1 << (N // 2)))
rotN = ((idx << 1) | (idx >> (N - 1))) & ((1 << N) - 1)
F2 = tt({tuple(sorted(k * i + r for i in m)) for r in range(k) for m in b})
for name, h, expect in [("matching", orbit((0, 2), 4), True),
                        ("K4", orbit((0, 1), 4) | orbit((0, 2), 4), True),
                        ("4-cycle", orbit((0, 1), 4), False)]:
    G = base ^ tt(h, s)
    assert walsh_abs_ok(G) == expect and bool((G[rotN] == G).all())
    if name == "matching": assert (G == F2).all()
    print(f"{name} lift: rotation symmetric; bent = {expect}")
print("All seed and lift checks passed.")

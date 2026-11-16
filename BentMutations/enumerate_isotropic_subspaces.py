from collections import Counter
n = 10
seeds = [(0,1,4), (0,1,8), (0,2,4), (0,2,7)]
triples = {tuple(sorted((i+t)%n for i in S))
           for S in seeds for t in range(n)}
assert len(triples) == 40

# rows[a][j] represents v -> T_b(a,v,e_j).
rows = []
for a in range(1 << n):
    R = [0]*n
    for i,j,l in triples:
        R[l] ^= (((a>>i)&1)<<j) | (((a>>j)&1)<<i)
        R[j] ^= (((a>>i)&1)<<l) | (((a>>l)&1)<<i)
        R[i] ^= (((a>>j)&1)<<l) | (((a>>l)&1)<<j)
    rows.append(R)

def kernel(equations):
    piv = {}
    for row in equations:
        while row:
            i = row.bit_length()-1
            if i in piv:
                row ^= piv[i]
            else:
                piv[i] = row
                break
    basis = []
    for i in range(n):
        if i in piv:
            continue
        v = 1 << i
        for p in sorted(piv):
            if (piv[p] & v).bit_count() & 1:
                v ^= 1 << p
        basis.append(v)
    assert len(basis) + len(piv) == n
    assert all(((row & v).bit_count() & 1) == 0
               for row in equations for v in basis)
    return basis

def span(B):
    out = [0]
    for b in B:
        out += [v ^ b for v in out]
    return frozenset(out)

seen = [set() for _ in range(6)]
Kdim4 = Counter()

def visit(B, equations):
    U = span(B)
    d = len(B)
    if U in seen[d]:
        return
    seen[d].add(U)
    K = span(kernel(equations))
    if d == 4:
        Kdim4[len(K).bit_length()-1] += 1
    if d == 5:
        return
    for v in K - U:
        visit(B + [v], equations + rows[v])

visit([], [])
assert [len(s) for s in seen] == [1,1023,5131,2955,225,0]
assert Kdim4 == {4:225}
print("Isotropic subspaces by dimension 0 through 5:", [len(s) for s in seen])
print("Number of four-dimensional isotropic subspaces with K(U) = U:", Kdim4[4])

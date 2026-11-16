from collections import Counter

def rank(rows):
    pivots = {}
    for row in rows:
        while row:
            lead = row.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = row
                break
            row ^= pivots[lead]
    return len(pivots)

def isomorphic(A, B):
    n = len(A)
    # Degrees and pairwise common-neighbor counts are invariant.
    def pair_data(rows):
        return [[((rows[i] >> j) & 1, (rows[i] & rows[j]).bit_count())
                 for j in range(n)] for i in range(n)]

    P, Q = pair_data(A), pair_data(B)
    a = [tuple(sorted(Counter(row).items())) for row in P]
    b = [tuple(sorted(Counter(row).items())) for row in Q]
    if Counter(a) != Counter(b):
        return False

    def extend(mapping, unused):
        if not unused:
            return True
        best = None
        for i in range(n):
            if i in mapping:
                continue
            candidates = [j for j in unused if a[i] == b[j]
                          and all(P[i][s] == Q[j][t]
                                  for s, t in mapping.items())]
            if not candidates:
                return False
            if best is None or len(candidates) < len(best[1]):
                best = (i, candidates)
        i, candidates = best
        return any(extend({**mapping, i: j}, unused - {j})
                   for j in candidates)

    return extend({}, set(range(n)))

def connected(rows):
    reached, pending = 1, 1
    while pending:
        v = pending & -pending
        pending ^= v
        new = rows[v.bit_length() - 1] & ~reached
        reached |= new
        pending |= new
    return reached == (1 << len(rows)) - 1

def count(k):
    n = 2 * k
    representatives = []
    for mask in range(1 << k):
        distances = [d for d in range(1, k + 1) if mask >> (d - 1) & 1]
        rows = [sum(1 << j for j in {(i + s*d) % n for d in distances
                                     for s in (-1, 1)})
                for i in range(n)]
        if rank(rows) == n and connected(rows):
            if not any(isomorphic(rows, old) for old in representatives):
                representatives.append(rows)
    return len(representatives)

counts = tuple(count(k) for k in range(1, 8))
assert counts == (1, 1, 1, 4, 6, 9, 19)
print("Connected nonsingular circulant graphs up to isomorphism (k = 1 through 7):", counts)

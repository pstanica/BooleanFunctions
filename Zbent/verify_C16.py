"""Exact certificates and affine-flat checks for the 24 sixteen-zero examples."""
from pathlib import Path
from itertools import combinations
import json
from verify_seed import bareiss, character

DATA = Path(__file__).resolve().parent / 'data'

def anf_degree(values):
    """Algebraic degree of a Boolean truth table in binary integer order."""
    a = list(values); n = (len(a)-1).bit_length()
    assert len(a) == 1 << n and set(a) <= {0,1}
    for i in range(n):
        for x in range(len(a)):
            if x >> i & 1: a[x] ^= a[x ^ (1 << i)]
    return max((x.bit_count() for x,v in enumerate(a) if v),default=-1)

def span(vectors):
    result = {0}
    for v in vectors: result |= {x ^ v for x in result}
    return result

def contained_three_flats(C):
    """Enumerate every affine 3-flat contained in C, without duplicates."""
    C = set(C); flats = set()
    for a in sorted(C):
        for basis in combinations(sorted({x ^ a for x in C} - {0}),3):
            V = span(basis)
            if len(V) == 8:
                A = frozenset(x ^ a for x in V)
                if A <= C: flats.add(A)
    return flats

def main():
    lines = [line.split() for line in (DATA/'nonsplitting_C16_examples.txt').read_text().splitlines()
             if line.strip() and not line.startswith('#')]
    certificates = json.loads((DATA/'C16_certificates.json').read_text())
    assert len(lines) == len(certificates) == 24
    union_count = no_flat_count = 0
    for number, ((flag,string),item) in enumerate(zip(lines,certificates),1):
        assert item['id'] == number and item['f'] == string and item['flag'] == int(flag)
        f = [{'0':0,'+':1,'-':-1}[c] for c in string]; assert len(f) == 64
        walsh = [sum(character(u,x)*f[x] for x in range(64)) for u in range(64)]
        assert set(walsh) <= {-8,0,8}
        assert item['dual'] == ''.join({-8:'-',0:'0',8:'+'}[v] for v in walsh)
        C = [x for x in range(64) if f[x] == 0]
        D = [u for u in range(64) if walsh[u] == 0]
        assert C == item['C'] and D == item['D'] and len(C) == len(D) == 16
        R = item['R']; assert len(set(R)) == 16 and not set(R) & set(D)
        determinant = bareiss([[character(u,x) for x in C] for u in R])
        assert determinant == item['det'] and determinant != 0
        cdeg = anf_degree([int(x in C) for x in range(64)])
        ddeg = anf_degree([int(x in D) for x in range(64)])
        assert cdeg == item['C_degree'] == 3 and ddeg == item['D_degree'] == 3
        flats = contained_three_flats(C); assert len(flats) == item['flats']
        pairs = [(A,frozenset(set(C)-A)) for A in flats if frozenset(set(C)-A) in flats]
        assert bool(pairs) == bool(int(flag))
        if int(flag):
            A,B = [frozenset(v) for v in item['decomposition']]
            assert A in flats and B in flats and not A & B and A | B == frozenset(C)
            union_count += 1
        else:
            assert not flats and item['decomposition'] is None
            no_flat_count += 1
    assert union_count == 4 and no_flat_count == 20
    print('C16 examples: all 24 exact Walsh transforms and nonzero 16-by-16 minors verified.')
    print('Zero-set geometry: 4 unions of two affine 3-flats; 20 contain no affine 3-flat.')
    print('All primal and Fourier zero-set indicators have algebraic degree 3.')

if __name__ == '__main__': main()

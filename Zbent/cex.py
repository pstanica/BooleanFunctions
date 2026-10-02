"""Verify the four-variable disjoint-spectra counterexample and explicit split."""
import numpy as np
from zbent import fwht, is_level1, is_bent_sign

def main():
    bit = lambda x, i: (x >> i) & 1
    f1 = np.array([1-2*(bit(x,2)*bit(x,3)) for x in range(16)], dtype=np.int64)
    f2 = np.array([f1[x]*(1-2*bit(x,0)) for x in range(16)], dtype=np.int64)
    w1, w2 = fwht(f1), fwht(f2)
    for w in [w1, w2]:
        assert np.isin(w, [-8,0,8]).all() and np.count_nonzero(w) == 4
    assert np.all(w1*w2 == 0)
    f = (f1+f2)//2
    g = np.array([1-2*((bit(x,0)*bit(x,1)+bit(x,2)*bit(x,3)) % 2) for x in range(16)], dtype=np.int64)
    h = np.array([g[x]*(1-2*bit(x,0)) for x in range(16)], dtype=np.int64)
    assert is_level1(f,2) and is_bent_sign(g,2) and is_bent_sign(h,2)
    assert np.array_equal((g+h)//2,f)
    print('Four-variable counterexample: disjoint semibent spectra and explicit bent split verified.')
if __name__ == '__main__': main()

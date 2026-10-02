"""Check every admissible signed coefficient vector on the GF(64)/GF(8) spread."""
import itertools
import numpy as np
from zbent import hadamard

def gmul(a,b):
    result=0
    for _ in range(6):
        if b & 1: result ^= a
        b >>= 1;high=a & 0x20;a=(a << 1) & 0x3f
        if high: a ^= 0x03
    return result

def main():
    powers=[1]
    for _ in range(62): powers.append(gmul(powers[-1],2))
    assert len(set(powers)) == 63 and gmul(powers[-1],2) == 1
    subfield=[0]+[powers[9*j] for j in range(7)]
    spread=[{gmul(powers[i],x) for x in subfield} for i in range(9)]
    assert set.union(*spread) == set(range(64))
    assert all(len(v)==8 and all(x^y in v for x in v for y in v) for v in spread)
    assert all(spread[i] & spread[j] == {0} for i in range(9) for j in range(i+1,9))
    indicators=np.array([[int(x in v) for x in range(64)] for v in spread],dtype=np.int64)
    H=hadamard(6);count=0
    for coeff in itertools.product([-1,0,1],repeat=9):
        total=sum(coeff)
        if total not in [-1,0,1]: continue
        c=np.array(coeff,dtype=np.int64);d=np.zeros(9,dtype=np.int64)
        zeros=np.flatnonzero(c==0);positive=len(zeros)//2 if total else (len(zeros)+1)//2
        d[zeros[:positive]]=1;d[zeros[positive:]]=-1
        f=c @ indicators;partner=d @ indicators
        assert np.isin(f,[-1,0,1]).all() and np.isin(H @ f,[-8,0,8]).all()
        assert np.all(np.abs(f+partner)==1) and np.all(np.abs(f-partner)==1)
        assert np.all(np.abs(H @ (f+partner))==8) and np.all(np.abs(H @ (f-partner))==8)
        count+=1
    print(f'Complete spread: all {count} admissible coefficient vectors have an explicit bent split.')
if __name__ == '__main__': main()

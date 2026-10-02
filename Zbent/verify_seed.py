#!/usr/bin/env python3
"""Exact seed certificate. Python 3 standard library only."""
F = '0-+0---+-00--+-++++++00+--00-0+0++0--0++--0++0++0+-+++-0+0++-+0-'
DUAL = '+--00+00+0+++0+--0-0-++--+--++++-0-00+0-++-++0-0--+0++-0-0-+---0'
C = [0,3,9,10,21,22,26,27,29,31,34,37,42,45,48,55,57,62]
D = [3,4,6,7,9,13,17,19,33,35,36,38,45,47,51,55,57,63]
R = [0,1,2,5,8,10,11,12,14,15,16,18,20,21,22,24,32,40]
def character(u,x): return -1 if (u & x).bit_count() % 2 else 1
def bareiss(matrix):
    a=[row[:] for row in matrix]; n=len(a); sign=1; previous=1
    for k in range(n-1):
        if a[k][k]==0:
            p=next((i for i in range(k+1,n) if a[i][k]),None)
            if p is None: return 0
            a[k],a[p]=a[p],a[k];sign=-sign
        pivot=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                numerator=pivot*a[i][j]-a[i][k]*a[k][j]
                assert numerator % previous == 0
                a[i][j]=numerator//previous
            a[i][k]=0
        previous=pivot
    return sign*a[-1][-1]
def main():
    decode={'0':0,'+':1,'-':-1}
    f=[decode[c] for c in F];dual=[decode[c] for c in DUAL]
    assert len(f)==len(dual)==64
    w=[sum(character(u,x)*f[x] for x in range(64)) for u in range(64)]
    assert w==[8*v for v in dual]
    assert C==[i for i,v in enumerate(f) if v==0]
    assert D==[i for i,v in enumerate(dual) if v==0]
    assert not set(R)&set(D)
    det=bareiss([[character(u,x) for x in C] for u in R])
    assert det==2**30
    # Exact cylinder minor H_2 tensor W_{R,C}: 72 x 72.
    cylinder=[[character(u,v)*character(r,c) for v in range(4) for c in C]
              for u in range(4) for r in R]
    det_cylinder=bareiss(cylinder)
    assert abs(det_cylinder)==2**192
    print('Seed transform, zero sets, and determinant verified:',det)
    print('Cylinder minor rank 72 verified; |det| = 2^192')
if __name__=='__main__': main()

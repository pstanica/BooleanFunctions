"""Exact, floating-point-free verification of the proof in nonsplitting_revised.tex.
Run: python3 verify_certificate.py   (needs numpy, sympy)"""
import numpy as np
from sympy import Matrix, Integer

s="0-+0---+-00--+-++++++00+--00-0+0++0--0++--0++0++0+-+++-0+0++-+0-"
f=np.array([{'0':0,'+':1,'-':-1}[c] for c in s],int)

# Sylvester-Hadamard H6 and exact (unnormalized) Walsh transform
def dot(u,x): return bin(u&x).count("1")&1
H6=np.array([[(-1)**dot(u,x) for x in range(64)] for u in range(64)],dtype=int)
Hf=H6.dot(f)
assert set(np.unique(Hf).tolist())<= {-8,0,8}                 # level one
fhat=Hf//8
assert "".join({0:'0',1:'+',-1:'-'}[v] for v in fhat)==\
   "+--00+00+0+++0+--0-0-++--+--++++-0-00+0-++-++0-0--+0++-0-0-+---0"
print("[1] f, hat f are {-1,0,1}-valued  ->  f in BF_1^3 : OK")

C=[x for x in range(64) if f[x]==0]
D=[u for u in range(64) if fhat[u]==0]
assert C==[0,3,9,10,21,22,26,27,29,31,34,37,42,45,48,55,57,62]
assert D==[3,4,6,7,9,13,17,19,33,35,36,38,45,47,51,55,57,63]
print("[2] zero sets C, D match the paper (|C|=|D|=18) : OK")

R=[0,1,2,5,8,10,11,12,14,15,16,18,20,21,22,24,32,40]
assert set(R) <= (set(range(64))-set(D)) and len(set(R))==18
W=Matrix(18,18,lambda i,j:Integer((-1)**dot(R[i],C[j])))
assert W.det()==2**30
print("[3] R subset D^c, det W_{R,C} = 2^30 (exact integer)  ->  W_{D^c,C} full column rank : OK")

# Corollary (real rigidity): supp(h) subset C, supp(hat h) subset D  =>  h=0.
# Equivalent to full column rank of the 46x18 block; the 18x18 minor above already proves it.
Dc=[u for u in range(64) if u not in set(D)]
assert Matrix(len(Dc),18,lambda i,j:Integer((-1)**dot(Dc[i],C[j]))).rank()==18
print("[4] real support-uncertainty on F_2^6 (rank 18) : OK")

# Cylinder lifting, direct exact check at m=2 (n=8): rank must be 4*18=72
def cyl_rank(m):
    M=2**m
    cols=[(w,x) for w in range(M) for x in C]
    rows=[(u,v) for u in range(M) for v in Dc]
    A=Matrix(len(rows),len(cols),
        lambda i,j:Integer((-1)**((dot(rows[i][0],cols[j][0])+dot(rows[i][1],cols[j][1]))&1)))
    return A.rank(),len(cols)
r,nc=cyl_rank(2); assert r==nc==72
print("[5] cylinder rigidity at n=8 (rank 72 = full) : OK")
print("\nALL EXACT CERTIFICATES VERIFIED -> non-splitting level-1 Z-bent functions")
print("exist in every even dimension n>=6 (Corollary, via Theorem product + cylinder lemma).")

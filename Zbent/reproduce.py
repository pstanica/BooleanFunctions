"""Run the exact certificates, prior-data comparisons, and finite example checks."""
import argparse
import numpy as np
import cex, spread, verify, verify_seed
from zbent import load_data, is_level1

def check_data():
    seed=np.array([{'0':0,'+':1,'-':-1}[c] for c in verify_seed.F],dtype=np.int64)
    for name in ['f6.npy','example_n6.npy']:
        assert np.array_equal(load_data(name),seed)
    f=seed
    for n in [8,10,12]:
        f=np.kron(np.array([1,1,1,-1],dtype=np.int64),f)
        assert np.array_equal(load_data(f'F{n}.npy'),f)
        assert is_level1(f,n//2)
    print('Bundled arrays: seed and products in dimensions 8, 10, and 12 agree exactly.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhaustive-seed',action='store_true',help='also enumerate all 2^18 possible seed partners')
    args=parser.parse_args()
    verify_seed.main();cex.main();check_data();spread.main()
    if args.exhaustive_seed:verify.main()
    print('All requested checks passed.')
if __name__ == '__main__': main()

"""Verify the bundled seed by checking every possible complementary sign vector."""
import numpy as np
from zbent import fwht, is_level1, load_data

def main():
    f = load_data('f6.npy'); assert is_level1(f,3)
    dual = fwht(f)//8; zero = np.flatnonzero(f == 0)
    columns = np.empty((64,len(zero)),dtype=np.int64)
    for j,x in enumerate(zero):
        unit=np.zeros(64,dtype=np.int64);unit[x]=1;columns[:,j]=fwht(unit)
    tested = 0
    for mask in range(1 << len(zero)):
        signs=np.array([1 if mask>>j & 1 else -1 for j in range(len(zero))],dtype=np.int64)
        w=columns @ signs;tested+=1
        if np.all(w[dual != 0] == 0) and np.all(np.abs(w[dual == 0]) == 8):
            raise AssertionError('A splitting partner was found.')
    print(f'Six-variable seed: all {tested} complementary sign vectors checked; no splitting partner.')
if __name__ == '__main__': main()

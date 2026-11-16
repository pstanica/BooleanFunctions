from itertools import combinations

def su8(x):
    z = [(x >> i) & 1 for i in range(8)]
    quadratic = sum(z[i] * z[i + 4] for i in range(4))
    correction = sum(z[i] * z[(i + 1) % 8] * z[(i + 2) % 8]
                     * (1 - z[(i + 4) % 8]) for i in range(8))
    return (quadratic + correction) % 2

table = [su8(x) for x in range(256)]
probes = [x for x in range(256) if x.bit_count() <= 2]

def second(x, a, b):
    return table[x] ^ table[x ^ a] ^ table[x ^ b] ^ table[x ^ a ^ b]

constant_pairs = 0
for a, b in combinations(range(1, 256), 2):
    value = second(0, a, b)
    if all(second(x, a, b) == value for x in probes):
        constant_pairs += 1
assert constant_pairs == 0

for a in range(256):
    walsh = sum((-1) ** (table[x] + (a & x).bit_count())
                for x in range(256))
    assert abs(walsh) == 16

print("Walsh coefficients of absolute value 16: 256")
print("Unordered pairs of distinct nonzero directions: 32385")
print("Pairs with a constant second difference:", constant_pairs)

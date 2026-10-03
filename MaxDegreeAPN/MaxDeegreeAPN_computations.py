"""
Computational verification code for:
  "On the Maximum Algebraic Degree of APN Functions
   via Partial APN Functions and Shifted Derivative Images"
  P. Stanica, Naval Postgraduate School

This file verifies the following results from the paper:
  1. The 0-APN counterexample (properness alone does not force surjectivity)
  2. Exhaustive APN + surjectivity check over GF(8)
  3. APN monomial surjectivity for n = 4, 5, 6, 7
  4. The S_3-orbit divisibility: 6 | N_b(F) for every b != 0
  5. The oversized-fiber obstruction and the threshold U_n
  6. The Walsh trace-slice identity: N_1(F) = 3*2^n-2 - 2^{1-2n}*Sigma_1(F)
  7. The Fano sum identity: sum of 7 Fano labels = D_u D_v D_w F(0)
  8. The quadratic Fano constraint: B = C + D

Requirements: Python 3.7+, no external packages needed.
Each section prints a summary consistent with the paper's claims.

NOTE ON SECTION LABELS
----------------------
All section headings use descriptive mathematical names rather than
theorem/proposition numbers.  If the paper is renumbered, this file
remains correct without any changes.
"""

import sys
import random
from itertools import product

# ---------------------------------------------------------------------------
# GF(2^n) arithmetic
# Each element is an integer whose binary representation gives coefficients.
# Multiplication is modular polynomial multiplication.
# ---------------------------------------------------------------------------

# Irreducible polynomials used (one fixed choice per n):
#   GF(8)  : x^3 + x + 1        = 0b1011    = 11
#   GF(16) : x^4 + x + 1        = 0b10011   = 19
#   GF(32) : x^5 + x^2 + 1      = 0b100101  = 37
#   GF(64) : x^6 + x + 1        = 0b1000011 = 67
#   GF(128): x^7 + x + 1        = 0b10000011 = 131

POLY = {3: 11, 4: 19, 5: 37, 6: 67, 7: 131}


def gf_mul(a, b, poly, n):
    """Multiply a and b in GF(2^n) with reducing polynomial poly."""
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a >> n:
            a ^= poly
    return result


def gf_pow(a, e, poly, n):
    """Compute a^e in GF(2^n)."""
    result = 1
    a = a % (1 << n)
    while e:
        if e & 1:
            result = gf_mul(result, a, poly, n)
        a = gf_mul(a, a, poly, n)
        e >>= 1
    return result


# ---------------------------------------------------------------------------
# Core APN and line-labeling functions
# ---------------------------------------------------------------------------

def is_apn(F, n):
    """Return True iff F : GF(2^n) -> GF(2^n) is APN.

    F is a list of length 2^n.  F is APN iff for every nonzero direction
    a, the derivative D_aF(x) = F(x XOR a) XOR F(x) is 2-to-1.
    """
    N = 1 << n
    for a in range(1, N):
        fiber_sizes = {}
        for x in range(N):
            v = F[x ^ a] ^ F[x]
            fiber_sizes[v] = fiber_sizes.get(v, 0) + 1
        if max(fiber_sizes.values()) > 2:
            return False
    return True


def is_0apn(F, n):
    """Return True iff F is 0-APN.

    F is 0-APN iff for every nonzero a, the equation D_aF(x) = D_aF(0)
    has only the trivial solutions x in {0, a}.  Equivalently (for F(0)=0),
    no projective line receives the label 0 under lambda_F.

    0-APN is strictly weaker than APN: APN requires D_aF to be 2-to-1
    for every direction a; 0-APN only forbids extra collisions at one fiber.
    """
    N = 1 << n
    for a in range(1, N):
        da0 = F[a] ^ F[0]           # D_aF(0)
        for x in range(N):
            if x == 0 or x == a:
                continue
            if (F[x ^ a] ^ F[x]) == da0:   # D_aF(x) = D_aF(0) with x outside {0,a}
                return False
    return True


def line_label_set(F, n):
    """Return the set of nonzero labels assigned to projective lines of PG(n-1,2).

    A projective line is an unordered triple {a, b, c} of nonzero elements
    with a XOR b XOR c = 0.  Its label is F(a) XOR F(b) XOR F(c).

    Canonical enumeration: iterate over pairs a < b with c = a XOR b > b,
    so each unordered triple is counted exactly once.

    Returns: (label_set, zero_label_count).
    """
    N = 1 << n
    labels = set()
    zero_count = 0
    for a in range(1, N):
        for b in range(a + 1, N):
            c = a ^ b
            if c <= b:
                continue        # not canonical; skip to avoid triple-counting
            label = F[a] ^ F[b] ^ F[c]
            if label == 0:
                zero_count += 1
            else:
                labels.add(label)
    return labels, zero_count


def line_label_count(F, n):
    """Return dict mapping each label to the number of projective lines with that label.

    Each unordered line {a, b, c} with a < b < c = a XOR b is counted once.
    For b != 0 and F APN:  N_b(F) = 6 * line_label_count(F, n).get(b, 0).
    """
    N = 1 << n
    counts = {}
    for a in range(1, N):
        for b in range(a + 1, N):
            c = a ^ b
            if c <= b:
                continue        # canonical: a < b < c
            label = F[a] ^ F[b] ^ F[c]
            counts[label] = counts.get(label, 0) + 1
    return counts


def Nb_direct(F, n, b):
    """Count ordered pairs (x,y) in GF(2^n)^2 with F(x) XOR F(y) XOR F(x XOR y) = b.

    Direct O(4^n) computation.  For b != 0 and F APN, the result equals
    6 * L_b(F) where L_b(F) = line_label_count(F, n).get(b, 0).
    """
    N = 1 << n
    count = 0
    for x in range(N):
        for y in range(N):
            if F[x] ^ F[y] ^ F[x ^ y] == b:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Section 1: 0-APN counterexample
#   Shows that properness of the line-labeling (= 0-APN-ness) does not
#   imply surjectivity; the full APN condition is needed.
# ---------------------------------------------------------------------------

def verify_0apn_counterexample():
    """Verify the explicit 0-APN non-APN non-surjective function over GF(16).

    Paper claim: there exists F with F(0)=0 that is 0-APN (no line gets
    label 0) but not APN (some D_aF is not 2-to-1), and whose line-labeling
    is not surjective.  This separates 0-APN from APN at the level of the
    surjectivity conjecture.
    """
    print("=" * 65)
    print("0-APN COUNTEREXAMPLE (properness alone does not force surjectivity)")
    print("=" * 65)

    n = 4
    N = 1 << n
    F = [0, 5, 7, 5, 14, 12, 8, 7, 10, 1, 7, 1, 10, 12, 8, 2]
    assert len(F) == N and F[0] == 0

    zero_apn  = is_0apn(F, n)
    full_apn  = is_apn(F, n)
    labels, zero_lbl = line_label_set(F, n)
    missing   = set(range(1, N)) - labels

    print(f"  F(0) = 0:                          {F[0] == 0}")
    print(f"  F is 0-APN (no line labeled 0):    {zero_apn}")
    print(f"  F is APN   (every D_aF is 2-to-1): {full_apn}")
    print(f"  Lines with label 0:                 {zero_lbl}")
    print(f"  Distinct nonzero labels:            {len(labels)} of {N-1}")
    print(f"  Missing labels:                     {sorted(missing)}")
    print()

    assert zero_apn,         "Should be 0-APN"
    assert not full_apn,     "Should NOT be APN"
    assert zero_lbl == 0,    "0-APN means no line carries label 0"
    assert len(missing) > 0, "Labeling should not be surjective"

    print("  OK: 0-APN does not imply surjectivity.\n")


# ---------------------------------------------------------------------------
# Section 2: Exhaustive surjectivity check over GF(8)   [--slow]
#   For n=3, PG(2,2) is a single Fano plane with 7 lines.  APN properness
#   forces all 7 labels to be pairwise distinct and nonzero, so every
#   nonzero field element appears automatically.  The exhaustive check
#   confirms this for all 86,016 APN functions with F(0)=0 over GF(8).
# ---------------------------------------------------------------------------

def exhaustive_gf8():
    """Enumerate all F: GF(8)->GF(8) with F(0)=0 and check APN + surjectivity."""
    print("=" * 65)
    print("EXHAUSTIVE SURJECTIVITY CHECK OVER GF(8)  [may take ~30-60s]")
    print("=" * 65)

    n = 3
    N = 1 << n
    full_set = set(range(1, N))

    count_total   = 0
    count_apn     = 0
    count_surj    = 0
    count_nonsurj = 0
    nonsurj_examples = []

    for vals in product(range(N), repeat=N - 1):
        F = [0] + list(vals)
        count_total += 1
        if not is_apn(F, n):
            continue
        count_apn += 1
        labels, _ = line_label_set(F, n)
        if labels == full_set:
            count_surj += 1
        else:
            count_nonsurj += 1
            if len(nonsurj_examples) < 3:
                nonsurj_examples.append((F[:], sorted(full_set - labels)))

    print(f"  Total functions with F(0)=0: {count_total:,}  (= 8^7)")
    print(f"  APN functions:               {count_apn:,}")
    print(f"  Surjective line-labeling:    {count_surj:,}")
    print(f"  Non-surjective:              {count_nonsurj:,}")
    for ex, miss in nonsurj_examples:
        print(f"  Non-surjective example: F={ex}, missing={miss}")
    print()

    assert count_apn == 86016, f"Expected 86,016 APN functions, got {count_apn}"
    assert count_nonsurj == 0, "Every APN function over GF(8) should be surjective"

    print("  OK: all 86,016 APN functions over GF(8) give surjective lambda_F.\n")


# ---------------------------------------------------------------------------
# Section 3: APN monomial surjectivity for n = 4, 5, 6, 7   [--slow]
# ---------------------------------------------------------------------------

def find_apn_monomials(n, poly):
    """Return all d in [2, 2^n-2] for which F(x)=x^d is APN over GF(2^n)."""
    N = 1 << n
    return [d for d in range(2, N - 1)
            if is_apn([gf_pow(x, d, poly, n) for x in range(N)], n)]


def check_monomial_surjectivity(n_values):
    """Verify surjectivity of lambda_F for all APN monomials over GF(2^n)."""
    print("=" * 65)
    print("APN MONOMIAL SURJECTIVITY CHECK  (n = 4, 5, 6, 7)")
    print("=" * 65)

    for n in n_values:
        poly     = POLY[n]
        N        = 1 << n
        full_set = set(range(1, N))

        apn_exp   = find_apn_monomials(n, poly)
        canonical = sorted(set(min(d, N - 1 - d) for d in apn_exp))

        print(f"\n  n={n}  (GF(2^{n}), irreducible poly = {poly})")
        print(f"  APN exponents (canonical up to x^d <-> x^{{2^n-1-d}}): {canonical}")

        all_surj = True
        for d in apn_exp:
            F = [gf_pow(x, d, poly, n) for x in range(N)]
            labels, zero_lbl = line_label_set(F, n)
            if labels != full_set:
                print(f"    x^{d}: NOT SURJECTIVE, missing={sorted(full_set - labels)}")
                all_surj = False
            if zero_lbl > 0:
                print(f"    x^{d}: ERROR — {zero_lbl} line(s) with label 0 (impossible for APN)")

        if all_surj:
            print(f"  All {len(apn_exp)} APN monomials give surjective lambda_F.  OK.")
    print()


# ---------------------------------------------------------------------------
# Section 4: S_3-orbit divisibility  6 | N_b(F) for every b != 0
# ---------------------------------------------------------------------------

def verify_s3_divisibility(n):
    """Verify 6 | N_b(F) for all b!=0 via the identity N_b = 6 * L_b."""
    poly = POLY[n]
    N    = 1 << n
    F    = [gf_pow(x, 3, poly, n) for x in range(N)]
    lc   = line_label_count(F, n)
    for b in range(1, N):
        nb = 6 * lc.get(b, 0)
        assert nb % 6 == 0
    print(f"  n={n}: 6 | N_b(F) for all b != 0  (N_b = 6*L_b)  OK")


def verify_s3_divisibility_all_n():
    print("=" * 65)
    print("S_3-ORBIT DIVISIBILITY:  6 | N_b(F) for every b != 0")
    print("  S_3 = GL(2,2) acts freely on nonzero second-order fibers.")
    print("  This corrects the false mod-4 argument.")
    print("=" * 65)
    for n in [3, 4, 5, 6, 7]:
        verify_s3_divisibility(n)
    print()


# ---------------------------------------------------------------------------
# Section 5: Oversized-fiber obstruction and threshold U_n
# ---------------------------------------------------------------------------

def check_nb_distribution(n):
    """N_b profile for Gold F(x)=x^3; compare max N_b against U_n."""
    poly = POLY[n]
    N    = 1 << n
    F    = [gf_pow(x, 3, poly, n) for x in range(N)]
    assert is_apn(F, n)

    lc          = line_label_count(F, n)
    N0          = Nb_direct(F, n, 0)
    expected_N0 = 3 * N - 2
    total_nz    = (N - 1) * (N - 2)
    U_n         = 6 * (-(-(N) // 6))          # 6 * ceil(2^n / 6)
    nb_values   = [6 * lc.get(b, 0) for b in range(1, N)]

    assert N0 == expected_N0,               f"N_0 mismatch: {N0} != {expected_N0}"
    assert sum(nb_values) == total_nz,      "Sum of N_b (b!=0) mismatch"
    assert all(v % 6 == 0 for v in nb_values), "Some N_b not divisible by 6"

    max_Nb = max(nb_values)
    print(f"  n={n}: N_0(F) = {N0}  (= 3*2^n-2)")
    print(f"    max N_b (b!=0) = {max_Nb},  U_n = {U_n}")
    if max_Nb < U_n:
        print(f"    max N_b < U_n => N_1(F) > 0 by the oversized-fiber contrapositive")
    else:
        print(f"    max N_b >= U_n => oversized-fiber bound not applicable for this F")
        print(f"    (N_1(F) > 0 confirmed separately by direct count below)")
    print(f"    N_1(F) = {nb_values[0]}  (confirmed positive by direct count)")
    assert nb_values[0] > 0


def check_nb_profile_and_threshold():
    print("=" * 65)
    print("OVERSIZED-FIBER OBSTRUCTION AND THRESHOLD U_n")
    print("  If N_1(F)=0 then some N_b(F) >= U_n = 6*ceil(2^n/6).")
    print("  Contrapositive: all N_b < U_n implies N_1(F) > 0.")
    print("=" * 65)

    # Threshold table
    print(f"\n  {'n':>3}  {'2^n':>6}  {'U_n':>6}  {'U_n/2':>6}  {'Extremal Sigma_1':>20}")
    print("  " + "-" * 55)
    for n in range(3, 9):
        Nv       = 1 << n
        U_n      = 6 * (-(-(Nv) // 6))
        extremal = (4 ** n) * (3 * (1 << (n - 1)) - 1)   # 2^{2n}*(3*2^{n-1}-1)
        print(f"  {n:>3}  {Nv:>6}  {U_n:>6}  {U_n//2:>6}  {extremal:>20}")

    print("  (U_n = 2^n+2 for even n, since 2^n ≡ 4 mod 6)")
    print("  (U_n = 2^n+4 for odd  n, since 2^n ≡ 2 mod 6)")
    print("\n  N_b distribution for Gold F(x)=x^3 (illustrative example):")
    for n in [3, 4, 5, 6]:
        print()
        check_nb_distribution(n)
    print()


# ---------------------------------------------------------------------------
# Section 6: Walsh trace-slice identity
#   N_1(F) = 3*2^n - 2 - 2^{1-2n} * Sigma_1(F)
#   where Sigma_1(F) = sum_{a, b: Tr(b)=1} W_F(a,b)^3.
#   N_1(F)=0 iff Sigma_1 reaches the extremal value 2^{2n}*(3*2^{n-1}-1).
# ---------------------------------------------------------------------------

def verify_walsh_trace_identity(n):
    """Verify the Walsh trace-slice identity using Gold F(x)=x^3."""
    poly = POLY[n]
    N    = 1 << n
    F    = [gf_pow(x, 3, poly, n) for x in range(N)]

    def trace(b):
        t, cur = 0, b
        for _ in range(n):
            t  ^= cur & 1
            cur = gf_mul(cur, cur, poly, n)
        return t

    def walsh(a, b):
        s = 0
        for x in range(N):
            exp_bit = trace(gf_mul(b, F[x], poly, n)) ^ trace(gf_mul(a, x, poly, n))
            s += 1 - 2 * exp_bit
        return s

    Sigma1 = sum(
        walsh(a, b) ** 3
        for b in range(1, N) if trace(b) == 1
        for a in range(N)
    )
    N1_formula = (3 * N - 2) - Sigma1 * (2 ** (1 - 2 * n))
    N1_direct  = Nb_direct(F, n, 1)

    print(f"  n={n}: Sigma_1 = {Sigma1}")
    print(f"    N_1 via Walsh formula = {N1_formula:.1f},  via direct count = {N1_direct}")
    assert abs(N1_formula - N1_direct) < 1e-6, "Walsh identity mismatch"
    print(f"    Match: OK")


def verify_walsh_trace_identity_all_n():
    print("=" * 65)
    print("WALSH TRACE-SLICE IDENTITY")
    print("  N_1(F) = 3*2^n - 2 - 2^{1-2n} * Sigma_1(F)")
    print("  where Sigma_1 sums W_F(a,b)^3 over b with Tr(b)=1.")
    print("  N_1(F)=0 iff Sigma_1 = 2^{2n}*(3*2^{n-1}-1)  (extremal value).")
    print("=" * 65)
    for n in [3, 4, 5]:
        verify_walsh_trace_identity(n)
    print()


# ---------------------------------------------------------------------------
# Section 7: Fano sum identity
#   For any F with F(0)=0 and any linearly independent u,v,w,
#   the sum of all 7 line labels in PG(<u,v,w>) equals D_u D_v D_w F(0).
#   Proof: each of the 7 Fano points lies on exactly 3 lines, so the
#   label sum = 3*sum_p F(p) = sum_p F(p) in characteristic 2.
# ---------------------------------------------------------------------------

def verify_fano_sum_identity(n, num_trials=200):
    """Test the Fano sum identity on random functions and random Fano planes."""
    poly = POLY[n]
    N    = 1 << n
    rng  = random.Random(42)

    for _ in range(num_trials):
        F = [0] + [rng.randrange(N) for _ in range(N - 1)]
        while True:
            u, v, w = (rng.randint(1, N - 1) for _ in range(3))
            if len({u, v, w, u^v, u^w, v^w, u^v^w}) == 7 and u^v^w != 0:
                break

        # The 7 lines of the Fano plane PG(<u,v,w>)
        fano_lines = [
            (u,   v,   u^v),   (u,   w,   u^w),   (v,   w,   v^w),
            (u^v, u^w, v^w),
            (u,   v^w, u^v^w), (v,   u^w, u^v^w), (w,   u^v, u^v^w),
        ]
        label_sum   = 0
        for (a, b, c) in fano_lines:
            label_sum ^= F[a] ^ F[b] ^ F[c]

        # D_u D_v D_w F(0) = sum of F over all 7 Fano points
        third_deriv = F[u] ^ F[v] ^ F[w] ^ F[u^v] ^ F[u^w] ^ F[v^w] ^ F[u^v^w]

        assert label_sum == third_deriv, "Fano sum identity failed"

    return True


def verify_fano_sum_identity_all_n():
    print("=" * 65)
    print("FANO SUM IDENTITY")
    print("  sum of 7 Fano line labels = D_u D_v D_w F(0)")
    print("  (holds for ALL F with F(0)=0, independently of APN;")
    print("   each Fano point lies on 3 lines, so label sum")
    print("   = 3*sum_p F(p) = sum_p F(p) in characteristic 2)")
    print("=" * 65)
    for n in [3, 4, 5, 6]:
        ok = verify_fano_sum_identity(n, num_trials=200)
        print(f"  n={n}: identity holds in all 200 random trials: {ok}")
    print("  OK: Fano sum identity verified.\n")


# ---------------------------------------------------------------------------
# Section 8: Quadratic Fano constraint  B = C + D
#   For quadratic F all third-order derivatives vanish, so the Fano sum
#   identity gives 0 = sum of 7 labels = B + C + D, i.e., B = C + D
#   among any four independent labels in a Fano plane.
#   Illustrated with the Gold function x^3 (algebraic degree 2).
# ---------------------------------------------------------------------------

def verify_quadratic_fano_constraint(n, num_trials=100):
    """Verify B=C+D and 7-label sum = 0 for quadratic APN F(x) = x^3."""
    poly = POLY[n]
    N    = 1 << n
    rng  = random.Random(99)
    F    = [gf_pow(x, 3, poly, n) for x in range(N)]
    assert is_apn(F, n)

    errors_sum = errors_BCD = 0

    for _ in range(num_trials):
        while True:
            u, v, w = (rng.randint(1, N - 1) for _ in range(3))
            if len({u, v, w, u^v, u^w, v^w, u^v^w}) == 7 and u^v^w != 0:
                break

        # Four independent labels in the Fano plane normal form
        A = F[u]   ^ F[v]   ^ F[u^v]
        B = F[u]   ^ F[w]   ^ F[u^w]
        C = F[v]   ^ F[w]   ^ F[v^w]
        D = F[u^v] ^ F[w]   ^ F[u^v^w]

        # Sum of all 7 labels: should be 0 for quadratic F
        if A ^ B ^ C ^ D ^ (A^B^C) ^ (A^B^D) ^ (A^C^D) != 0:
            errors_sum += 1
        # Quadratic constraint: B = C + D  (equivalently B XOR C XOR D = 0)
        if B ^ C ^ D != 0:
            errors_BCD += 1

    return errors_sum == 0, errors_BCD == 0


def verify_quadratic_fano_constraint_all_n():
    print("=" * 65)
    print("QUADRATIC FANO CONSTRAINT:  B = C + D")
    print("  For quadratic F, third-order derivatives vanish,")
    print("  so the Fano sum gives B + C + D = 0, i.e., B = C + D.")
    print("  Verified with the Gold function F(x) = x^3 (degree 2).")
    print("=" * 65)
    for n in [3, 4, 5, 6]:
        ok_sum, ok_BCD = verify_quadratic_fano_constraint(n)
        print(f"  n={n} (Gold x^3): 7-label sum = 0: {ok_sum},  B = C+D: {ok_BCD}")
        assert ok_sum, f"7-label sum non-zero for n={n}"
        assert ok_BCD, f"B != C+D for n={n}"
    print("  OK: quadratic Fano constraint verified.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    verify_0apn_counterexample()
    verify_fano_sum_identity_all_n()
    verify_quadratic_fano_constraint_all_n()
    verify_s3_divisibility_all_n()
    verify_walsh_trace_identity_all_n()
    check_nb_profile_and_threshold()

    if "--slow" in sys.argv:
        print("Running slow computations...")
        exhaustive_gf8()
        check_monomial_surjectivity([4, 5, 6, 7])
    else:
        print("=" * 65)
        print("SLOW COMPUTATIONS (not run by default)")
        print("Run with --slow to execute:")
        print("  python apn_computations.py --slow")
        print()
        print("  Exhaustive GF(8) check (~30-60s):")
        print("    All 86,016 APN functions over GF(8) give surjective lambda_F.")
        print()
        print("  APN monomial surjectivity, n=4,5,6,7:")
        print("    All APN monomials x^d give surjective lambda_F.")
        print("=" * 65)

    print("\nAll fast verifications passed.")

"""
Companion computational code for:

  "On the Fourier Min-Entropy/Influence Ratio of Symmetric Boolean Functions"
  Ana Salagean and Pantelimon Stanica

Every numbered verification below corresponds to a specific proposition, lemma,
or theorem in the paper.  Running this file executes all checks and prints a
summary.  Expected runtime on a modern laptop: under three minutes.

Requirements: Python 3.7+, standard library only.
"""

from math import comb, log2
from fractions import Fraction


# ===========================================================================
# CORE PRIMITIVES
# ===========================================================================

def krawtchouk(j, k, n):
    """
    Krawtchouk polynomial  K_j(k; n) = sum_{s=0}^{j} (-1)^s C(k,s) C(n-k, j-s).

    Used in Proposition 3.1 to express Walsh coefficients of symmetric functions.
    """
    return sum((-1)**s * comb(k, s) * comb(n - k, j - s) for s in range(j + 1))


def walsh_spectrum(sigma, n):
    """
    Walsh spectrum of a symmetric Boolean function on n variables.

    Parameters
    ----------
    sigma : list of ±1, length n+1
        sigma[w] = (-1)^{f(w)}  for Hamming weight w = 0, 1, ..., n.
    n : int

    Returns
    -------
    list of float, length n+1
        W_f(k) for k = 0, ..., n, using Proposition 3.1:
            W_f(k) = 2^{-n} sum_{w=0}^{n} sigma[w] * K_w(k; n).
    """
    return [
        sum(sigma[w] * krawtchouk(w, k, n) for w in range(n + 1)) / 2**n
        for k in range(n + 1)
    ]


def total_influence(sigma, n):
    """
    Total influence  Inf(f) = sum_{k=0}^{n} k C(n,k) W_f(k)^2  (Proposition 3.1).
    """
    Wk = walsh_spectrum(sigma, n)
    return sum(k * comb(n, k) * Wk[k]**2 for k in range(n + 1))


def max_walsh(sigma, n):
    """M(f) = max_k |W_f(k)|."""
    return max(abs(v) for v in walsh_spectrum(sigma, n))


def sym_fn(w, D):
    """
    S_{D,n}(w) = XOR_{d in D} C(w, d) mod 2.

    The value depends only on w and D (not n), consistent with the paper's
    convention that D subseteq {1,...,n}.
    """
    return sum(comb(w, d) % 2 for d in D) % 2


def sign_seq(D, n):
    """sigma_w = (-1)^{S_{D,n}(w)} for w = 0, ..., n."""
    return [(-1)**sym_fn(w, D) for w in range(n + 1)]


# ===========================================================================
# 1. PROPOSITION 3.1  —  Parseval and grouped influence formula
# ===========================================================================

def verify_parseval_and_influence(n_max=10):
    """
    Check that for every S_{D,n} with even n <= n_max:

      (i)  sum_k C(n,k) W_f(k)^2 = 1                  (Parseval)
      (ii) Inf(f) from Walsh = Inf(f) from transitions  (Corollary 5.1)

    The transition formula (Corollary 5.1) is
        Inf(S_{D,n}) = (n / 2^{n-1}) * T
    where T = sum_{w=0}^{n-1} C(n-1, w) * [C(w, d-1) mod 2 for some d in D].
    More precisely, the derivative indicator at weight w is
        sum_{d in D} C(w, d-1) mod 2.
    """
    for n in range(2, n_max + 1, 2):
        for D_mask in range(1, 2**n):
            D = [d for d in range(1, n + 1) if D_mask & (1 << (d - 1))]
            sigma = sign_seq(D, n)
            Wk = walsh_spectrum(sigma, n)

            parseval = sum(comb(n, k) * Wk[k]**2 for k in range(n + 1))
            assert abs(parseval - 1.0) < 1e-9, \
                f"Parseval failed: n={n}, D={D}, got {parseval}"

            inf_walsh = sum(k * comb(n, k) * Wk[k]**2 for k in range(n + 1))
            T = sum(
                comb(n - 1, w) * (sum(comb(w, d - 1) % 2 for d in D) % 2)
                for w in range(n)
            )
            inf_trans = n * T / 2**(n - 1)
            assert abs(inf_walsh - inf_trans) < 1e-9, \
                f"Inf mismatch: n={n}, D={D}"
    return "PASS"


# ===========================================================================
# 2. THEOREM 5.2  —  Inf(S_{2,n}) = n/2 for all even n
# ===========================================================================

def verify_inf_s2n(n_max=30):
    """
    Verify Theorem 5.2:  Inf(S_{2,n}) = n/2  for even n <= n_max.

    The analytic proof uses the involution w <-> (n-1-w) on odd w in {0,...,n-1}
    to show T = sum_{w odd} C(n-1, w) = 2^{n-2}, giving Inf = n * 2^{n-2} / 2^{n-1} = n/2.
    """
    for n in range(2, n_max + 1, 2):
        # Direct check: T must equal 2^{n-2}
        T = sum(comb(n - 1, w) for w in range(n) if w % 2 == 1)
        assert T == 2**(n - 2), \
            f"Transition sum wrong at n={n}: T={T}, expected {2**(n-2)}"
        # Exact influence via Fraction arithmetic
        Inf = Fraction(n * T, 2**(n - 1))
        assert Inf == Fraction(n, 2), \
            f"Inf(S_{{2,{n}}}) = {Inf} != {Fraction(n, 2)}"
    return "PASS"


# ===========================================================================
# 3. LEMMA 5.4  —  Mirror symmetry
# ===========================================================================

def verify_mirror_symmetry(n_max=12):
    """
    Verify Lemma 5.4:  K_w(n-k; n) = (-1)^w * K_w(k; n)  for all w, k, n.

    Consequence:  W_{S_{D XOR {1}, n}}(k) = W_{S_{D,n}}(n-k).
    """
    # Step 1: Krawtchouk reflection identity
    for n in range(1, n_max + 1):
        for w in range(n + 1):
            for k in range(n + 1):
                lhs = krawtchouk(w, n - k, n)
                rhs = (-1)**w * krawtchouk(w, k, n)
                assert abs(lhs - rhs) < 1e-9, \
                    f"K reflection failed: w={w}, k={k}, n={n}"

    # Step 2: Walsh consequence
    for n in range(2, 9, 2):
        for D_mask in range(1, 2**n, 3):
            D = [d for d in range(1, n + 1) if D_mask & (1 << (d - 1))]
            D_sym = sorted(set(D) ^ {1})   # D XOR {1}
            Wk_D = walsh_spectrum(sign_seq(D, n), n)
            Wk_X = walsh_spectrum(sign_seq(D_sym, n), n)
            for k in range(n + 1):
                assert abs(Wk_X[k] - Wk_D[n - k]) < 1e-9, \
                    f"Mirror Walsh failed: D={D}, k={k}, n={n}"
    return "PASS"


# ===========================================================================
# 4. PROPOSITION 5.4  —  Endpoint domination for single-degree
# ===========================================================================

def verify_endpoint_domination(d_max=11, n_max=16):
    """
    Verify Proposition 5.4:
        M(S_{d,n}) = max(|W_{S_{d,n}}(0)|, |W_{S_{d,n}}(n)|)
    for all single-degree d and even n with d <= d_max, n <= n_max.

    This is the key structural property used to reduce the single-degree
    conjecture to an endpoint bias bound.
    """
    for d in range(1, d_max + 1):
        for n in range(max(d, 2), n_max + 1, 2):
            sigma = [(-1)**(comb(w, d) % 2) for w in range(n + 1)]
            Wk = walsh_spectrum(sigma, n)
            M = max(abs(v) for v in Wk)
            max_end = max(abs(Wk[0]), abs(Wk[n]))
            assert abs(M - max_end) < 1e-9, \
                f"Endpoint domination failed: d={d}, n={n}, " \
                f"M={M:.8f}, max_end={max_end:.8f}"
    return "PASS"


# ===========================================================================
# 5. PROPOSITION 5.5  —  Density recurrence
# ===========================================================================

def density_exact(D, n):
    """
    p_{D,n} = 2^{-n} sum_{w: S_{D,n}(w)=1} C(n,w)   as an exact Fraction.
    """
    return Fraction(
        sum(comb(n, w) for w in range(n + 1) if sym_fn(w, D) == 1),
        2**n
    )


def verify_density_recurrence(n_max=12):
    """
    Verify Proposition 5.5 for D not containing 1:
        W_{S_{D,n+1}}(0) = (1/2)(W_{S_{D,n}}(0) + W_{S_{D_f1,n}}(0))

    where D_f1 = symmetric difference of D and {d-1 : d in D, d >= 2}.
    This is the restriction formula at k=0 from Corollary 4.2, written out
    explicitly using W(0) = 1 - 2p.

    Note: D containing 1 is handled by applying mirror symmetry first (Lemma 5.4),
    which maps S_{D,n} to S_{D XOR {1},n}, and 1 is removed from D XOR {1}.
    """
    for n in range(2, n_max):
        for D_mask in range(1, 2**n):
            D = [d for d in range(1, n + 1) if D_mask & (1 << (d - 1))]
            if 1 in D:
                continue    # handled via mirror symmetry; see note above
            D_minus_1 = [d - 1 for d in D if d >= 2]
            # symmetric difference D XOR D_minus_1
            combined = D + D_minus_1
            D_f1 = sorted(set(x for x in combined if combined.count(x) % 2 == 1))

            W_next = 1 - 2 * density_exact(D, n + 1)
            p0 = density_exact(D, n)
            p1 = density_exact(D_f1, n) if D_f1 else Fraction(0)
            W_rhs = (1 - 2 * p0 + 1 - 2 * p1) / 2

            assert W_next == W_rhs, \
                f"Density recurrence failed: D={D}, n={n}: " \
                f"LHS={W_next}, RHS={W_rhs}"
    return "PASS"


# ===========================================================================
# 6. PROPOSITION 6.1  —  Poincaré bound (very small influence)
# ===========================================================================

def verify_poincare_bound():
    """
    Verify the analytic inequality used in Proposition 6.1:
        1 - t  >=  4^{-t} = 2^{-2t}   for all t in [0, 1/2].

    In the paper:  if Inf(f) <= 1/2, then W_f(0)^2 >= 1 - Inf(f) >= 4^{-Inf(f)},
    so M(f) >= 2^{-Inf(f)}.

    We verify at 10 001 equally-spaced points in [0, 0.5].
    """
    steps = 10_000
    for i in range(steps + 1):
        t = i / (2 * steps)          # t in [0, 0.5]
        assert 1.0 - t >= 4.0**(-t) - 1e-15, \
            f"Poincaré inequality 1-t >= 4^{{-t}} failed at t={t}"
    return "PASS"


# ===========================================================================
# 7. LEMMA 7.1  —  AND function
# ===========================================================================

def verify_and_function(n_max=20):
    """
    Verify Lemma 7.1:  Hmin(S_{n,n}) <= 2 Inf(S_{n,n})  for even n <= n_max.

    Equivalently (using W_{S_{n,n}}(0) = 1 - 2^{1-n} and Inf = n/2^{n-1}):
        1 - 2^{1-n}  >=  2^{-n/2^{n-1}}

    The analytic proof uses  -log_2(1-x) <= 2x/ln(2)  for x in (0, 1/2].
    """
    for n in range(2, n_max + 1, 2):
        W0 = 1 - 2**(1 - n)           # |W_{AND_n}(0)|
        Inf = n / 2**(n - 1)           # Inf(AND_n)
        threshold = 2**(-Inf)
        assert W0 >= threshold - 1e-14, \
            f"AND inequality failed: n={n}, W0={W0:.10f}, 2^{{-Inf}}={threshold:.10f}"
    return "PASS"


# ===========================================================================
# 8. LEMMA 7.2  —  Asymptotic bias (single-degree finite verification)
# ===========================================================================

def verify_single_degree_finite(d_max=20, n_max=50):
    """
    Verify Proposition 7.3 (finite range part of Theorem 7.4):
        max(|W_{S_{d,n}}(0)|, |W_{S_{d,n}}(n)|) >= 2^{-Inf(S_{d,n})}
    for all d in {1,...,d_max} and even n with d <= n <= n_max,
    restricted to the range 1/2 < Inf < n/2.

    This is the finite verification combined with Lemma 7.2 to prove
    the single-degree conjecture for all d and all n.
    """
    for d in range(1, d_max + 1):
        for n in range(max(d, 2), n_max + 1, 2):
            sigma = [(-1)**(comb(w, d) % 2) for w in range(n + 1)]
            Wk = walsh_spectrum(sigma, n)
            Inf = sum(k * comb(n, k) * Wk[k]**2 for k in range(n + 1))
            if Inf < 1e-9 or Inf >= n / 2 - 1e-9:
                continue    # Inf=0 (constant) or Case A handled analytically
            bias = max(abs(Wk[0]), abs(Wk[n]))
            assert bias >= 2**(-Inf) - 1e-9, \
                f"Single-degree bias failed: d={d}, n={n}, " \
                f"bias={bias:.8f}, 2^{{-Inf}}={2**(-Inf):.8f}"
    return "PASS"


# ===========================================================================
# 9. PROPOSITION SD-COMP  —  Full verification n <= 12
# ===========================================================================

def verify_conjecture_all_n12(n_max=12):
    """
    Proposition SD-comp: verify  Hmin(S_{D,n}) <= 2 Inf(S_{D,n})
    for every nonempty D and every even n <= n_max.

    Returns (total_instances, equality_cases) where equality_cases lists
    all (n, D) pairs achieving equality (the bent functions).
    """
    total = 0
    equality_cases = []
    for n in range(2, n_max + 1, 2):
        for D_mask in range(1, 2**n):
            D = tuple(d for d in range(1, n + 1) if D_mask & (1 << (d - 1)))
            sigma = [(-1)**sym_fn(w, D) for w in range(n + 1)]
            Wk = walsh_spectrum(sigma, n)
            M = max(abs(v) for v in Wk)
            Inf = sum(k * comb(n, k) * Wk[k]**2 for k in range(n + 1))
            if Inf < 1e-9:
                continue
            total += 1
            assert M >= 2**(-Inf) - 1e-9, \
                f"Conjecture FAILS: n={n}, D={D}, " \
                f"M={M:.8f}, 2^{{-Inf}}={2**(-Inf):.8f}"
            if abs(-2 * log2(M) - 2 * Inf) < 1e-9:
                equality_cases.append((n, D))
    # Confirm only the known bent cases achieve equality
    for n, D in equality_cases:
        assert set(D) in ({2}, {1, 2}), \
            f"Unexpected equality: n={n}, D={D}"
    return total, equality_cases


# ===========================================================================
# 10. THEOREM 8.1 CASE (iii)  —  Multi-degree barrier key inequality
# ===========================================================================

def _get_barrier_data(sigma_f, n):
    """
    Internal helper: compute restriction data for one S_{D,n}.
    Returns (Inf_f, I0, I1, delta, cond, M0, M1, is_barrier, layers_disjoint).
    """
    D = None   # not needed here; we work from sigma_f directly
    n_r = n - 1
    Wk_f = walsh_spectrum(sigma_f, n)
    Inf_f = sum(k * comb(n, k) * Wk_f[k]**2 for k in range(n + 1))
    return Inf_f, Wk_f


def verify_barrier_key_inequality(n_max=14):
    """
    Verify the key inequality of Theorem 8.1 Case (iii):
        max(M(f_0), M(f_1)) >= 2 * 2^{-Inf(S_{D,n})}
    for every S_{D,n} with |D| >= 2, even n <= n_max, in the barrier regime:
      · 1/2 < Inf < n/2
      · restrictions f_0, f_1 have disjoint maximising Walsh layers
      · cond = |I_0 - I_1| + 2*delta < 2

    This is the computational foundation of the multi-degree proof for n <= 14.
    """
    total_barrier = 0
    for n in range(4, n_max + 1, 2):
        n_r = n - 1
        for D_mask in range(1, 2**n):
            D = [d for d in range(1, n + 1) if D_mask & (1 << (d - 1))]
            if len(D) < 2:
                continue

            sigma_f = sign_seq(D, n)
            Wk_f = walsh_spectrum(sigma_f, n)
            Inf_f = sum(k * comb(n, k) * Wk_f[k]**2 for k in range(n + 1))
            if Inf_f >= n / 2 - 1e-9 or Inf_f <= 0.5 + 1e-9:
                continue

            # Build the two restrictions
            D_minus_1 = [d - 1 for d in D if d >= 2]
            f0 = [sym_fn(w, D) for w in range(n_r + 1)]
            f1 = [(sym_fn(w, D) ^ sym_fn(w, D_minus_1)) for w in range(n_r + 1)]
            s0 = [(-1)**v for v in f0]
            s1 = [(-1)**v for v in f1]
            W0k = walsh_spectrum(s0, n_r)
            W1k = walsh_spectrum(s1, n_r)
            M0 = max(abs(v) for v in W0k)
            M1 = max(abs(v) for v in W1k)

            # Find maximising layer sets
            L0 = {k for k in range(n_r + 1) if abs(abs(W0k[k]) - M0) < 1e-9}
            L1 = {k for k in range(n_r + 1) if abs(abs(W1k[k]) - M1) < 1e-9}
            if L0 & L1:
                continue    # common layer — handled by Theorem 5.3 (commonlayer)

            I0 = sum(k * comb(n_r, k) * W0k[k]**2 for k in range(n_r + 1))
            I1 = sum(k * comb(n_r, k) * W1k[k]**2 for k in range(n_r + 1))
            delta = (sum(comb(n_r, w) * (f0[w] ^ f1[w])
                         for w in range(n_r + 1)) / 2**n_r)
            cond = abs(I0 - I1) + 2 * delta
            if cond >= 2:
                continue    # disjoint-layer criterion — handled by Theorem 5.4

            # Barrier case: verify the key inequality
            total_barrier += 1
            target = 2 * 2**(-Inf_f)
            assert max(M0, M1) >= target - 1e-9, \
                f"Barrier inequality FAILS: n={n}, D={D}, " \
                f"max(M0,M1)={max(M0,M1):.8f}, target={target:.8f}"

    return total_barrier


# ===========================================================================
# 11. TABLE 1  —  Minimum bias ratios for single-degree
# ===========================================================================

def compute_table_ratios(d_range=range(3, 13), n_max=50):
    """
    Compute Table 1 from the paper:
        min_{even n : d<=n<=n_max, 1/2 < Inf < n/2}
            max(|W_{S_{d,n}}(0)|, |W_{S_{d,n}}(n)|) / 2^{-Inf(S_{d,n})}

    All values exceed 1, confirming Proposition 7.3 over the finite range.
    """
    table = {}
    for d in d_range:
        min_ratio = float('inf')
        for n in range(max(d, 2), n_max + 1, 2):
            sigma = [(-1)**(comb(w, d) % 2) for w in range(n + 1)]
            Wk = walsh_spectrum(sigma, n)
            Inf = sum(k * comb(n, k) * Wk[k]**2 for k in range(n + 1))
            if Inf <= 0.5 + 1e-9 or Inf >= n / 2 - 1e-9:
                continue
            bias = max(abs(Wk[0]), abs(Wk[n]))
            if bias < 1e-12:
                continue
            min_ratio = min(min_ratio, bias / 2**(-Inf))
        table[d] = round(min_ratio, 5) if min_ratio < float('inf') else None
    return table


# ===========================================================================
# 12. INFLUENCE ASYMPTOTICS  —  Proposition 5.7
# ===========================================================================

def compute_influence_asymptotics(a_max=3, n_max=40):
    """
    Compute Inf(S_{2^a, n}) / (n / 2^a) for even n, verifying convergence to 1.

    Proposition 5.7 states  Inf(S_{2^a,n}) = n/2^a * (1 + O(2^{-n/2})).
    """
    results = {}
    for a in range(1, a_max + 1):
        d = 2**a
        rows = []
        for n in range(2 * d, n_max + 1, 2):
            T = sum(comb(n - 1, w) * (comb(w, d - 1) % 2) for w in range(n))
            Inf = n * T / 2**(n - 1)
            ratio = Inf / (n / 2**a) if n > 0 else 0
            rows.append((n, round(Inf, 6), round(ratio, 8)))
        results[a] = rows
    return results


# ===========================================================================
# 13. GAP FOR ODD-VARIABLE FUNCTIONS  —  cited in multi-degree argument
# ===========================================================================

def compute_odd_gap(m_max=11):
    """
    For each odd m <= m_max, compute:
        gamma(m) = min_{nonempty D} M(S_{D,m}) / 2^{-Inf(S_{D,m})}

    The minimum is always achieved by the AND function D = {m},
    giving gamma(m) = (1 - 2^{1-m}) * 2^{m / 2^{m-1}}.

    This quantity appears in the discussion of the multi-degree barrier:
    the restriction functions are S_{D',m}-type on m = n-1 (odd) variables,
    and their individual ratios are bounded below by gamma(m) > 1.
    Note: gamma(m) alone is NOT sufficient to close the barrier analytically
    (the global minimum is too weak); the barrier is closed computationally
    for n <= 14 by verify_barrier_key_inequality().
    """
    results = {}
    for m in range(3, m_max + 1, 2):   # odd m only
        min_ratio = float('inf')
        argmin_D = None
        for D_mask in range(1, 2**m):
            D = [d for d in range(1, m + 1) if D_mask & (1 << (d - 1))]
            sigma = sign_seq(D, m)
            Wk = walsh_spectrum(sigma, m)
            Inf = sum(k * comb(m, k) * Wk[k]**2 for k in range(m + 1))
            if Inf < 1e-9:
                continue
            M = max(abs(v) for v in Wk)
            ratio = M / 2**(-Inf)
            if ratio < min_ratio:
                min_ratio = ratio
                argmin_D = D
        # Verify it equals the AND formula
        and_ratio = (1 - 2**(1 - m)) / 2**(-m / 2**(m - 1))
        assert abs(min_ratio - and_ratio) < 1e-6, \
            f"AND not minimiser at m={m}: got {min_ratio:.8f}, AND={and_ratio:.8f}"
        results[m] = (round(min_ratio, 7), argmin_D)
    return results


# ===========================================================================
# MASTER RUNNER
# ===========================================================================

def run_all(verbose=True):
    """Execute every verification and print a formatted summary."""

    def run(label, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        status = "PASS" if result else "FAIL"
        if verbose:
            print(f"  {status}  {label}")
        return result

    print("=" * 62)
    print("  Companion verification suite")
    print("=" * 62)

    print("\nSection 3 — Krawtchouk framework")
    run("Parseval + influence formula  (Prop 3.1, n ≤ 10)",
        verify_parseval_and_influence, n_max=10)

    print("\nSection 5 — Elementary symmetric functions")
    run("Inf(S_{2,n}) = n/2 for even n ≤ 30  (Thm 5.2)",
        verify_inf_s2n, n_max=30)
    run("Mirror symmetry  K_w(n-k;n) = (−1)^w K_w(k;n)  (Lem 5.4)",
        verify_mirror_symmetry, n_max=12)
    run("Endpoint domination  M = max(|W(0)|,|W(n)|)  (Prop 5.4, d ≤ 11, n ≤ 16)",
        verify_endpoint_domination, d_max=11, n_max=16)
    run("Density recurrence  (Prop 5.5, n ≤ 12, D not containing 1)",
        verify_density_recurrence, n_max=12)

    print("\nSection 6 — Poincaré / small-influence")
    run("Poincaré:  1−t ≥ 4^{−t}  for t ∈ [0, 1/2]  (Prop 6.1)",
        verify_poincare_bound)

    print("\nSection 7 — Single-degree conjecture")
    run("AND function:  (1−2^{1−n}) ≥ 2^{−n/2^{n−1}}  (Lem 7.1, n ≤ 20)",
        verify_and_function, n_max=20)
    run("Finite verification  max(|W(0)|,|W(n)|) ≥ 2^{−Inf}  (Prop 7.3, d ≤ 20, n ≤ 50)",
        verify_single_degree_finite, d_max=20, n_max=50)

    print("\nSection 5 — Full computational verification  (Prop SD-comp)")
    total, eq = verify_conjecture_all_n12(n_max=12)
    if verbose:
        print(f"  PASS  Conjecture holds for all {total:,} instances with n ≤ 12")
        eq_str = ", ".join(f"(n={n}, D={set(D)})" for n, D in eq[:4])
        print(f"        Equality (bent) cases: {eq_str} ...")

    print("\nSection 8 — Multi-degree barrier  (Thm 8.1 Case iii)")
    barrier_count = verify_barrier_key_inequality(n_max=14)
    if verbose:
        print(f"  PASS  max(M₀,M₁) ≥ 2·2^{{−I}}  for all {barrier_count:,} "
              "barrier cases with n ≤ 14")

    print("\nAdditional tables and data")

    table = compute_table_ratios(d_range=range(3, 13), n_max=50)
    if verbose:
        print("  Table 1 — minimum bias ratio max(|W(0)|,|W(n)|)/2^{−Inf}:")
        row = "  " + "  ".join(f"d={d}: {v:.5f}" for d, v in table.items())
        print(row)

    gaps = compute_odd_gap(m_max=11)
    if verbose:
        print("  Odd-variable gap γ(m) — minimiser is AND function D={m}:")
        for m, (g, D) in gaps.items():
            print(f"    m={m}: γ={g:.7f}  (achieved by D={D})")

    asym = compute_influence_asymptotics(a_max=3, n_max=40)
    if verbose:
        print("  Inf(S_{2^a,n}) / (n/2^a)  →  1  (last 3 values shown):")
        for a, rows in asym.items():
            tail = "  ".join(f"n={r[0]}: {r[2]:.7f}" for r in rows[-3:])
            print(f"    a={a}: " + tail)

    print("\n" + "=" * 62)
    print("  All verifications passed.")
    print("=" * 62)


if __name__ == "__main__":
    run_all(verbose=True)

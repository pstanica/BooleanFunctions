#!/usr/bin/env python3
"""
verify_two_cubic_links.py

Exact, dependency-free verifier for the coordinate-link obstruction associated
with one or two homogeneous cubic rotation-symmetric SANF generators.

For a normalized cubic generator [0,a,b] over Z/nZ, the coordinate derivative
at e_0 has quadratic edge set

    {{a,b}, {-a,b-a}, {-b,a-b}}.

For two generators, repeated quadratic monomials cancel over F_2, so the active
graph is the symmetric difference of the two edge sets.

For each admissible generator or unordered pair of distinct generators, this
program:
  1. constructs the active graph exactly modulo n;
  2. computes the adjacency-matrix kernel over F_2;
  3. evaluates q_G(z)=sum_{uv in E(G)} z_u z_v on every radical vector;
  4. reports a counterexample if q_G is nonzero on the radical.

If q_G vanishes on the radical, the corresponding quadratic derivative is
unbalanced, so the original Boolean function cannot be bent.

IMPORTANT:
  This is an exact exhaustive verifier for every modulus n explicitly tested.
  A scan over n <= N is finite evidence, not by itself an all-dimensional
  symbolic proof. Its output is intended to support and debug the symbolic
  classification used in the manuscript.

Examples:
    python3 verify_two_cubic_links.py --n 14
    python3 verify_two_cubic_links.py --scan 6 50
    python3 verify_two_cubic_links.py --n 14 --csv cases_n14.csv
    python3 verify_two_cubic_links.py --n 14 --show-failures

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


Generator = Tuple[int, int, int]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class CaseResult:
    n: int
    generator_1: Generator
    generator_2: Optional[Generator]
    vertices: Tuple[int, ...]
    edges: Tuple[Edge, ...]
    kernel_dimension: int
    radical_size: int
    q_nonzero_on_radical: bool
    witness: Optional[Tuple[int, ...]]

    @property
    def passed(self) -> bool:
        return not self.q_nonzero_on_radical

    @property
    def degree_multiset(self) -> Tuple[int, ...]:
        degree = Counter()
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return tuple(sorted(degree.values()))


def normalize_generator(n: int, a: int, b: int) -> Generator:
    """
    Canonical representative of the cyclic-translation orbit of {0,a,b}.

    The support is treated as an unordered 3-subset of Z/nZ.  Each of its three
    points is translated to 0; the lexicographically least sorted result is
    returned.
    """
    support = {0, a % n, b % n}
    if len(support) != 3:
        raise ValueError("A cubic generator must contain three distinct residues.")

    representatives: List[Generator] = []
    for t in support:
        translated = tuple(sorted((x - t) % n for x in support))
        if translated[0] != 0 or len(set(translated)) != 3:
            raise AssertionError("Internal normalization failure.")
        representatives.append(translated)  # type: ignore[arg-type]

    return min(representatives)


def all_cubic_generators(n: int) -> List[Generator]:
    """All normalized cyclic classes of 3-subsets of Z/nZ containing 0."""
    representatives = {
        normalize_generator(n, a, b)
        for a in range(1, n)
        for b in range(a + 1, n)
    }
    return sorted(representatives)


def normalize_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError(
            "A loop occurred. For an admissible cubic support this should not happen."
        )
    return (u, v) if u < v else (v, u)


def link_edges(n: int, generator: Generator) -> Tuple[Edge, Edge, Edge]:
    """The three quadratic edges in Delta_{e_0} of the cubic orbit generator."""
    zero, a, b = generator
    if zero != 0:
        raise ValueError("Generator must be normalized as [0,a,b].")

    raw_edges = (
        (a % n, b % n),
        ((-a) % n, (b - a) % n),
        ((-b) % n, (a - b) % n),
    )
    return tuple(normalize_edge(u, v) for u, v in raw_edges)


def active_edges(
    n: int,
    generator_1: Generator,
    generator_2: Optional[Generator] = None,
) -> Tuple[Edge, ...]:
    """
    Symmetric difference of the link-edge multisets.

    An edge appearing an even number of times cancels over F_2.  Internal
    coincidences within a single link triple are handled correctly as well.
    """
    counts: Counter[Edge] = Counter(link_edges(n, generator_1))
    if generator_2 is not None:
        counts.update(link_edges(n, generator_2))
    return tuple(sorted(edge for edge, count in counts.items() if count % 2 == 1))


def adjacency_rows(vertices: Sequence[int], edges: Sequence[Edge]) -> List[int]:
    """
    Adjacency matrix over F_2, encoded as integer bit rows.
    """
    index = {v: i for i, v in enumerate(vertices)}
    rows = [0] * len(vertices)
    for u, v in edges:
        i = index[u]
        j = index[v]
        rows[i] ^= 1 << j
        rows[j] ^= 1 << i
    return rows


def gf2_kernel_basis(rows: Sequence[int], ncols: int) -> List[int]:
    """
    Return a basis of the nullspace over F_2 as integer bit vectors.

    The input rows are integer bit masks.  Elimination is exact.
    """
    matrix = list(rows)
    pivot_cols: List[int] = []
    pivot_row = 0

    for col in range(ncols):
        pivot = None
        for r in range(pivot_row, len(matrix)):
            if (matrix[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue

        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]

        for r in range(len(matrix)):
            if r != pivot_row and ((matrix[r] >> col) & 1):
                matrix[r] ^= matrix[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    free_cols = [c for c in range(ncols) if c not in set(pivot_cols)]
    basis: List[int] = []

    for free in free_cols:
        vector = 1 << free
        # In reduced row-echelon form, pivot variable equals the sum of the
        # free variables occurring in that row.
        for r, pivot_col in reversed(list(enumerate(pivot_cols))):
            row = matrix[r]
            parity = (row & vector).bit_count() & 1
            if parity:
                vector ^= 1 << pivot_col
        basis.append(vector)

    # Defensive verification.
    for vector in basis:
        for row in rows:
            if (row & vector).bit_count() & 1:
                raise AssertionError("Kernel computation failed.")

    return basis


def span_vectors(basis: Sequence[int]) -> Iterator[int]:
    """Enumerate every vector in the F_2-span of basis."""
    vector = 0
    yield vector
    # Gray-code enumeration changes one basis coefficient at a time.
    previous_gray = 0
    for k in range(1, 1 << len(basis)):
        gray = k ^ (k >> 1)
        changed = gray ^ previous_gray
        index = changed.bit_length() - 1
        vector ^= basis[index]
        yield vector
        previous_gray = gray


def q_graph(vector: int, edge_indices: Sequence[Tuple[int, int]]) -> int:
    """Evaluate sum_{ij in E} x_i x_j in F_2."""
    value = 0
    for i, j in edge_indices:
        value ^= ((vector >> i) & 1) & ((vector >> j) & 1)
    return value


def vector_support(vector: int, vertices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(vertices[i] for i in range(len(vertices)) if (vector >> i) & 1)


def verify_case(
    n: int,
    generator_1: Generator,
    generator_2: Optional[Generator],
) -> CaseResult:
    edges = active_edges(n, generator_1, generator_2)
    vertices = tuple(sorted({v for edge in edges for v in edge}))

    if not vertices:
        # Zero quadratic form: its radical is the zero-dimensional active
        # space here; the derivative is certainly constant/unbalanced.
        return CaseResult(
            n=n,
            generator_1=generator_1,
            generator_2=generator_2,
            vertices=vertices,
            edges=edges,
            kernel_dimension=0,
            radical_size=1,
            q_nonzero_on_radical=False,
            witness=None,
        )

    index = {v: i for i, v in enumerate(vertices)}
    edge_indices = tuple((index[u], index[v]) for u, v in edges)
    rows = adjacency_rows(vertices, edges)
    basis = gf2_kernel_basis(rows, len(vertices))

    witness: Optional[Tuple[int, ...]] = None
    bad = False
    for vector in span_vectors(basis):
        if q_graph(vector, edge_indices):
            bad = True
            witness = vector_support(vector, vertices)
            break

    return CaseResult(
        n=n,
        generator_1=generator_1,
        generator_2=generator_2,
        vertices=vertices,
        edges=edges,
        kernel_dimension=len(basis),
        radical_size=1 << len(basis),
        q_nonzero_on_radical=bad,
        witness=witness,
    )


def graph_signature(result: CaseResult) -> Tuple[int, int, Tuple[int, ...], int, bool]:
    """
    A compact invariant for summary counts.

    This is not claimed to be a complete graph-isomorphism invariant; it is
    used only for readable diagnostics.
    """
    return (
        len(result.vertices),
        len(result.edges),
        result.degree_multiset,
        result.kernel_dimension,
        result.q_nonzero_on_radical,
    )


def write_csv(path: Path, results: Iterable[CaseResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n",
                "generator_1",
                "generator_2",
                "vertices",
                "edges",
                "degree_multiset",
                "kernel_dimension",
                "radical_size",
                "q_nonzero_on_radical",
                "witness",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.n,
                    result.generator_1,
                    result.generator_2,
                    result.vertices,
                    result.edges,
                    result.degree_multiset,
                    result.kernel_dimension,
                    result.radical_size,
                    result.q_nonzero_on_radical,
                    result.witness,
                ]
            )


def verify_modulus(
    n: int,
    include_single: bool = True,
    include_pairs: bool = True,
    show_failures: bool = False,
) -> Tuple[List[CaseResult], Dict[Tuple[int, int, Tuple[int, ...], int, bool], int]]:
    if n < 3:
        raise ValueError("n must be at least 3.")
    if n % 4 != 2:
        raise ValueError(
            f"This verifier is intended for n == 2 (mod 4); received n={n}."
        )

    generators = all_cubic_generators(n)
    results: List[CaseResult] = []
    signatures: Dict[Tuple[int, int, Tuple[int, ...], int, bool], int] = defaultdict(int)

    if include_single:
        for generator in generators:
            result = verify_case(n, generator, None)
            results.append(result)
            signatures[graph_signature(result)] += 1
            if show_failures and not result.passed:
                print(format_failure(result))

    if include_pairs:
        for g1, g2 in itertools.combinations(generators, 2):
            result = verify_case(n, g1, g2)
            results.append(result)
            signatures[graph_signature(result)] += 1
            if show_failures and not result.passed:
                print(format_failure(result))

    return results, signatures


def format_failure(result: CaseResult) -> str:
    return (
        "FAILURE\n"
        f"  n              = {result.n}\n"
        f"  generator 1    = {result.generator_1}\n"
        f"  generator 2    = {result.generator_2}\n"
        f"  active edges   = {result.edges}\n"
        f"  kernel dim.    = {result.kernel_dimension}\n"
        f"  witness support= {result.witness}\n"
    )


def print_summary(
    n: int,
    results: Sequence[CaseResult],
    signatures: Dict[Tuple[int, int, Tuple[int, ...], int, bool], int],
) -> None:
    generators = all_cubic_generators(n)
    failures = [result for result in results if not result.passed]
    singles = sum(result.generator_2 is None for result in results)
    pairs = len(results) - singles

    print(f"n = {n}")
    print(f"normalized cubic generators : {len(generators)}")
    print(f"single-generator cases      : {singles}")
    print(f"two-generator cases         : {pairs}")
    print(f"total verified cases        : {len(results)}")
    print(f"failures                    : {len(failures)}")
    print(f"diagnostic signatures       : {len(signatures)}")

    print("\nSignature counts")
    print("  (vertices, edges, degree multiset, nullity, q|rad nonzero) : count")
    for signature, count in sorted(signatures.items()):
        print(f"  {signature} : {count}")

    if failures:
        print("\nFirst failure")
        print(format_failure(failures[0]))
    else:
        print("\nPASS: q_G vanishes on the adjacency radical in every tested case.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact finite verifier for coordinate-link graphs of one or two "
            "cubic rotation-symmetric generators."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="verify one modulus n == 2 (mod 4)")
    mode.add_argument(
        "--scan",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="verify every n == 2 (mod 4) in the inclusive interval",
    )

    parser.add_argument(
        "--singles-only",
        action="store_true",
        help="verify only one-generator cases",
    )
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="verify only two-generator cases",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="print every failing case",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="write all case data to CSV (available only with --n)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.singles_only and args.pairs_only:
        print("error: --singles-only and --pairs-only are incompatible", file=sys.stderr)
        return 2

    include_single = not args.pairs_only
    include_pairs = not args.singles_only

    if args.n is not None:
        try:
            results, signatures = verify_modulus(
                args.n,
                include_single=include_single,
                include_pairs=include_pairs,
                show_failures=args.show_failures,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print_summary(args.n, results, signatures)

        if args.csv is not None:
            write_csv(args.csv, results)
            print(f"\nCSV written to {args.csv}")

        return 1 if any(not result.passed for result in results) else 0

    if args.csv is not None:
        print("error: --csv is available only with --n", file=sys.stderr)
        return 2

    start, stop = args.scan
    if start > stop:
        start, stop = stop, start

    tested = 0
    total_cases = 0
    total_failures = 0

    for n in range(start, stop + 1):
        if n % 4 != 2 or n < 3:
            continue

        results, _ = verify_modulus(
            n,
            include_single=include_single,
            include_pairs=include_pairs,
            show_failures=args.show_failures,
        )
        failures = sum(not result.passed for result in results)
        tested += 1
        total_cases += len(results)
        total_failures += failures

        print(
            f"n={n:4d}  generators={len(all_cubic_generators(n)):6d}  "
            f"cases={len(results):10d}  failures={failures}"
        )

        if failures and not args.show_failures:
            first = next(result for result in results if not result.passed)
            print(format_failure(first))

    print(
        f"\nScanned {tested} admissible moduli; "
        f"{total_cases} exact cases; {total_failures} failures."
    )

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

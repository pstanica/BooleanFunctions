#!/usr/bin/env python3
"""
classify_two_cubic_families.py

Exact, dependency-free classifier for the non-bipartite active coordinate-link
graphs arising from one or two homogeneous cubic rotation-symmetric SANF
generators.

For a normalized cubic generator [0,a,b] over Z/nZ, define

    E(a,b) = {
        {a,b},
        {-a,b-a},
        {-b,a-b}
    }.

For two generators [0,a,b] and [0,c,d], the active graph has edge set

    E(a,b) triangle E(c,d),

because repeated quadratic monomials cancel over F_2.

This program:

  1. enumerates all normalized cubic SANF generators;
  2. enumerates every unordered pair of distinct normalized generators;
  3. constructs the active coordinate-link graph exactly modulo n;
  4. retains the non-bipartite cases;
  5. computes an exact graph-isomorphism canonical label;
  6. checks q_G(z)=0 on ker(A_G);
  7. tests whether each non-bipartite case belongs, up to all anchored
     representations and interchange of the two generators, to one of the
     candidate parameter families

        F1(a): [0,a,2a], [0,a,3a]
        F2(a): [0,a,2a], [0,a,h]
        F3(a): [0,a,h],  [0,a,h+a]

     where h=n/2;

  8. separates F1 into the subcases 5a=0 and 5a!=0;
  9. prints every unclassified case in full.

IMPORTANT:
  This is an exact exhaustive verifier for every modulus explicitly tested.
  A finite scan is not an all-dimensional symbolic proof.  Its purpose is to
  identify the correct symbolic classification statement and expose any
  missing family.

Examples:
    python3 classify_two_cubic_families.py --n 14
    python3 classify_two_cubic_families.py --scan 6 50
    python3 classify_two_cubic_families.py --scan 6 102 \
        --output family_report_6_102.txt
    python3 classify_two_cubic_families.py --n 14 --show-covered

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


Generator = Tuple[int, int, int]
OrderedForm = Tuple[int, int]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class FamilyMatch:
    family: str
    parameter: int
    form_1: OrderedForm
    form_2: OrderedForm
    generators_swapped: bool

    def description(self, n: int) -> str:
        if self.family == "F1":
            branch = "5a=0" if (5 * self.parameter) % n == 0 else "5a!=0"
            return (
                f"F1 ({branch}), a={self.parameter}, "
                f"forms={self.form_1}; {self.form_2}, "
                f"swapped={self.generators_swapped}"
            )
        return (
            f"{self.family}, a={self.parameter}, "
            f"forms={self.form_1}; {self.form_2}, "
            f"swapped={self.generators_swapped}"
        )


@dataclass(frozen=True)
class CaseData:
    n: int
    generator_1: Generator
    generator_2: Generator
    vertices: Tuple[int, ...]
    edges: Tuple[Edge, ...]
    canonical_label: str
    degree_multiset: Tuple[int, ...]
    kernel_dimension: int
    kernel_basis_supports: Tuple[Tuple[int, ...], ...]
    q_nonzero_on_radical: bool
    radical_witness: Optional[Tuple[int, ...]]
    family_matches: Tuple[FamilyMatch, ...]

    @property
    def classified(self) -> bool:
        return bool(self.family_matches)


def normalize_generator(n: int, a: int, b: int) -> Generator:
    support = {0, a % n, b % n}
    if len(support) != 3:
        raise ValueError("A cubic generator must have three distinct residues.")

    representatives: List[Generator] = []
    for t in support:
        translated = tuple(sorted((x - t) % n for x in support))
        representatives.append(translated)  # type: ignore[arg-type]
    return min(representatives)


def all_cubic_generators(n: int) -> List[Generator]:
    return sorted(
        {
            normalize_generator(n, a, b)
            for a in range(1, n)
            for b in range(a + 1, n)
        }
    )


def anchored_forms(n: int, generator: Generator) -> Tuple[OrderedForm, ...]:
    """
    Return all ordered anchored representations of the cyclic support.

    For support {0,a,b}, choose any support point as origin and order the two
    remaining residues.  Up to six forms occur.
    """
    support = tuple(generator)
    forms = set()

    for origin in support:
        remaining = [x for x in support if x != origin]
        x = (remaining[0] - origin) % n
        y = (remaining[1] - origin) % n
        if x != 0 and y != 0 and x != y:
            forms.add((x, y))
            forms.add((y, x))

    return tuple(sorted(forms))


def normalize_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("Loop detected in an admissible cubic link graph.")
    return (u, v) if u < v else (v, u)


def link_edges(n: int, generator: Generator) -> Tuple[Edge, Edge, Edge]:
    _, a, b = generator
    return (
        normalize_edge(a % n, b % n),
        normalize_edge((-a) % n, (b - a) % n),
        normalize_edge((-b) % n, (a - b) % n),
    )


def active_edges(n: int, g1: Generator, g2: Generator) -> Tuple[Edge, ...]:
    counts: Counter[Edge] = Counter(link_edges(n, g1))
    counts.update(link_edges(n, g2))
    return tuple(sorted(e for e, multiplicity in counts.items() if multiplicity % 2))


def is_bipartite(vertices: Sequence[int], edges: Sequence[Edge]) -> bool:
    adjacency: Dict[int, List[int]] = {v: [] for v in vertices}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    color: Dict[int, int] = {}
    for start in vertices:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in color:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def canonical_graph_label(vertices: Sequence[int], edges: Sequence[Edge]) -> str:
    """
    Exact graph-isomorphism canonical label.

    Permutations are restricted to equal-degree cells, which remains exact
    because every isomorphism preserves vertex degrees.
    """
    if not vertices:
        return "0:"

    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    cells: Dict[int, List[int]] = defaultdict(list)
    for v in vertices:
        cells[degree[v]].append(v)

    ordered_degrees = sorted(cells)
    cell_permutations = [
        list(itertools.permutations(cells[d]))
        for d in ordered_degrees
    ]

    best: Optional[str] = None
    for blocks in itertools.product(*cell_permutations):
        ordered_vertices = [v for block in blocks for v in block]
        relabel = {v: i for i, v in enumerate(ordered_vertices)}
        encoded_edges = sorted(
            normalize_edge(relabel[u], relabel[v])
            for u, v in edges
        )
        edge_string = ",".join(f"{u}-{v}" for u, v in encoded_edges)
        label = f"{len(vertices)}:{edge_string}"
        if best is None or label < best:
            best = label

    if best is None:
        raise AssertionError("Canonical labeling failed.")
    return best


def adjacency_rows(vertices: Sequence[int], edges: Sequence[Edge]) -> List[int]:
    index = {v: i for i, v in enumerate(vertices)}
    rows = [0] * len(vertices)
    for u, v in edges:
        i = index[u]
        j = index[v]
        rows[i] ^= 1 << j
        rows[j] ^= 1 << i
    return rows


def gf2_kernel_basis(rows: Sequence[int], ncols: int) -> List[int]:
    matrix = list(rows)
    pivot_cols: List[int] = []
    pivot_row = 0

    for col in range(ncols):
        pivot = next(
            (
                r
                for r in range(pivot_row, len(matrix))
                if (matrix[r] >> col) & 1
            ),
            None,
        )
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

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(ncols) if c not in pivot_set]
    basis: List[int] = []

    for free in free_cols:
        vector = 1 << free
        for r, pivot_col in reversed(list(enumerate(pivot_cols))):
            if (matrix[r] & vector).bit_count() & 1:
                vector ^= 1 << pivot_col
        basis.append(vector)

    for vector in basis:
        for row in rows:
            if (row & vector).bit_count() & 1:
                raise AssertionError("Kernel computation failed.")

    return basis


def span_vectors(basis: Sequence[int]) -> Iterator[int]:
    vector = 0
    yield vector
    previous_gray = 0
    for k in range(1, 1 << len(basis)):
        gray = k ^ (k >> 1)
        changed = gray ^ previous_gray
        index = changed.bit_length() - 1
        vector ^= basis[index]
        yield vector
        previous_gray = gray


def q_graph(vector: int, indexed_edges: Sequence[Tuple[int, int]]) -> int:
    value = 0
    for i, j in indexed_edges:
        value ^= ((vector >> i) & 1) & ((vector >> j) & 1)
    return value


def vector_support(vector: int, vertices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(vertices[i] for i in range(len(vertices)) if (vector >> i) & 1)


def degree_multiset(edges: Sequence[Edge]) -> Tuple[int, ...]:
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return tuple(sorted(degree.values()))


def match_families(
    n: int,
    g1: Generator,
    g2: Generator,
) -> Tuple[FamilyMatch, ...]:
    """
    Test all anchored forms and both generator orders against F1, F2, F3.
    """
    h = n // 2
    matches: List[FamilyMatch] = []

    for swapped, first, second in (
        (False, g1, g2),
        (True, g2, g1),
    ):
        for form1 in anchored_forms(n, first):
            a, b = form1
            for form2 in anchored_forms(n, second):
                c, d = form2

                if (
                    b % n == (2 * a) % n
                    and c % n == a % n
                    and d % n == (3 * a) % n
                ):
                    matches.append(
                        FamilyMatch("F1", a % n, form1, form2, swapped)
                    )

                if (
                    b % n == (2 * a) % n
                    and c % n == a % n
                    and d % n == h
                ):
                    matches.append(
                        FamilyMatch("F2", a % n, form1, form2, swapped)
                    )

                if (
                    b % n == h
                    and c % n == a % n
                    and d % n == (h + a) % n
                ):
                    matches.append(
                        FamilyMatch("F3", a % n, form1, form2, swapped)
                    )

    # Remove duplicates while preserving deterministic order.
    unique: Dict[Tuple[str, int, OrderedForm, OrderedForm, bool], FamilyMatch] = {}
    for match in matches:
        key = (
            match.family,
            match.parameter,
            match.form_1,
            match.form_2,
            match.generators_swapped,
        )
        unique[key] = match

    return tuple(
        sorted(
            unique.values(),
            key=lambda m: (
                m.family,
                m.parameter,
                m.generators_swapped,
                m.form_1,
                m.form_2,
            ),
        )
    )


def analyze_nonbipartite_case(
    n: int,
    g1: Generator,
    g2: Generator,
) -> CaseData:
    edges = active_edges(n, g1, g2)
    vertices = tuple(sorted({v for e in edges for v in e}))

    if is_bipartite(vertices, edges):
        raise ValueError("Case is bipartite.")

    canonical = canonical_graph_label(vertices, edges)
    rows = adjacency_rows(vertices, edges)
    basis = gf2_kernel_basis(rows, len(vertices))

    index = {v: i for i, v in enumerate(vertices)}
    indexed_edges = tuple((index[u], index[v]) for u, v in edges)

    witness: Optional[Tuple[int, ...]] = None
    bad = False
    for vector in span_vectors(basis):
        if q_graph(vector, indexed_edges):
            bad = True
            witness = vector_support(vector, vertices)
            break

    basis_supports = tuple(vector_support(v, vertices) for v in basis)

    return CaseData(
        n=n,
        generator_1=g1,
        generator_2=g2,
        vertices=vertices,
        edges=edges,
        canonical_label=canonical,
        degree_multiset=degree_multiset(edges),
        kernel_dimension=len(basis),
        kernel_basis_supports=basis_supports,
        q_nonzero_on_radical=bad,
        radical_witness=witness,
        family_matches=match_families(n, g1, g2),
    )


def classify_modulus(
    n: int,
) -> Tuple[List[CaseData], int, int, int]:
    if n < 6:
        raise ValueError("n must be at least 6.")
    if n % 4 != 2:
        raise ValueError("n must satisfy n == 2 mod 4.")

    generators = all_cubic_generators(n)
    nonbipartite: List[CaseData] = []
    total_pairs = 0
    radical_failures = 0
    unclassified = 0

    for g1, g2 in itertools.combinations(generators, 2):
        total_pairs += 1
        edges = active_edges(n, g1, g2)
        vertices = tuple(sorted({v for e in edges for v in e}))
        if is_bipartite(vertices, edges):
            continue

        data = analyze_nonbipartite_case(n, g1, g2)
        nonbipartite.append(data)

        if data.q_nonzero_on_radical:
            radical_failures += 1
        if not data.classified:
            unclassified += 1

    return nonbipartite, total_pairs, radical_failures, unclassified


def family_bucket(data: CaseData) -> str:
    if not data.family_matches:
        return "UNCLASSIFIED"

    names = set()
    for match in data.family_matches:
        if match.family == "F1":
            branch = "F1:5a=0" if (5 * match.parameter) % data.n == 0 else "F1:5a!=0"
            names.add(branch)
        else:
            names.add(match.family)

    return ",".join(sorted(names))


def format_case(data: CaseData) -> str:
    lines = [
        f"n                     : {data.n}",
        f"generator 1           : {data.generator_1}",
        f"generator 2           : {data.generator_2}",
        f"vertices               : {data.vertices}",
        f"edges                  : {data.edges}",
        f"degree multiset        : {data.degree_multiset}",
        f"canonical label        : {data.canonical_label}",
        f"kernel dimension       : {data.kernel_dimension}",
        f"kernel basis supports  : {data.kernel_basis_supports}",
        f"q nonzero on radical   : {data.q_nonzero_on_radical}",
        f"radical witness        : {data.radical_witness}",
        f"family bucket          : {family_bucket(data)}",
        "family matches         :",
    ]

    if data.family_matches:
        for match in data.family_matches:
            lines.append(f"  - {match.description(data.n)}")
    else:
        lines.append("  - none")

    return "\n".join(lines)


def summarize_cases(cases: Sequence[CaseData]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for data in cases:
        counts[family_bucket(data)] += 1
    return dict(counts)


def graph_class_summary(cases: Sequence[CaseData]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for data in cases:
        summary[data.canonical_label][family_bucket(data)] += 1
    return {label: dict(counts) for label, counts in summary.items()}


def write_report(
    path: Path,
    header: str,
    graph_representatives: Sequence[CaseData],
    unclassified_cases: Sequence[CaseData],
    covered_examples: Sequence[CaseData],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip() + "\n\n")

        handle.write("GRAPH-CLASS REPRESENTATIVES\n")
        handle.write("===========================\n\n")
        for i, data in enumerate(graph_representatives, start=1):
            handle.write(f"CLASS {i}\n")
            handle.write(format_case(data))
            handle.write("\n\n")

        handle.write("UNCLASSIFIED CASES\n")
        handle.write("==================\n\n")
        if unclassified_cases:
            for i, data in enumerate(unclassified_cases, start=1):
                handle.write(f"UNCLASSIFIED {i}\n")
                handle.write(format_case(data))
                handle.write("\n\n")
        else:
            handle.write("None.\n\n")

        if covered_examples:
            handle.write("COVERED EXAMPLES\n")
            handle.write("================\n\n")
            for i, data in enumerate(covered_examples, start=1):
                handle.write(f"EXAMPLE {i}\n")
                handle.write(format_case(data))
                handle.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the candidate symbolic families F1, F2, F3 against every "
            "non-bipartite active link graph for one or two cubic generators."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="analyze one modulus")
    mode.add_argument(
        "--scan",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="scan every n == 2 mod 4 in the inclusive interval",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="write a full report to this text file",
    )
    parser.add_argument(
        "--show-covered",
        action="store_true",
        help="print covered cases as well as unclassified cases",
    )
    parser.add_argument(
        "--max-covered-examples",
        type=int,
        default=20,
        help="maximum number of covered examples included in the report",
    )
    parser.add_argument(
        "--max-unclassified",
        type=int,
        default=100,
        help="maximum unclassified cases printed or written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    global_graph_representatives: Dict[str, CaseData] = {}
    all_unclassified: List[CaseData] = []
    covered_examples: List[CaseData] = []
    aggregate_family_counts: Dict[str, int] = defaultdict(int)
    aggregate_graph_family_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    total_moduli = 0
    total_pairs = 0
    total_nonbipartite = 0
    total_radical_failures = 0
    total_unclassified = 0

    if args.n is not None:
        moduli = [args.n]
    else:
        start, stop = args.scan
        if start > stop:
            start, stop = stop, start
        moduli = [n for n in range(start, stop + 1) if n >= 6 and n % 4 == 2]

    for n in moduli:
        try:
            cases, pairs, radical_failures, unclassified = classify_modulus(n)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        total_moduli += 1
        total_pairs += pairs
        total_nonbipartite += len(cases)
        total_radical_failures += radical_failures
        total_unclassified += unclassified

        local_counts = summarize_cases(cases)

        for data in cases:
            global_graph_representatives.setdefault(data.canonical_label, data)
            bucket = family_bucket(data)
            aggregate_family_counts[bucket] += 1
            aggregate_graph_family_counts[data.canonical_label][bucket] += 1

            if not data.classified and len(all_unclassified) < args.max_unclassified:
                all_unclassified.append(data)
            elif (
                data.classified
                and len(covered_examples) < args.max_covered_examples
            ):
                covered_examples.append(data)

        print(
            f"n={n:4d}  pairs={pairs:10d}  "
            f"nonbipartite={len(cases):8d}  "
            f"radical_failures={radical_failures:3d}  "
            f"unclassified={unclassified:6d}  "
            f"families={dict(sorted(local_counts.items()))}"
        )

    representatives = sorted(
        global_graph_representatives.values(),
        key=lambda data: (
            len(data.vertices),
            len(data.edges),
            data.canonical_label,
        ),
    )

    header_lines = [
        f"scanned moduli={total_moduli}",
        f"total unordered generator pairs={total_pairs}",
        f"total non-bipartite cases={total_nonbipartite}",
        f"radical failures={total_radical_failures}",
        f"unclassified cases={total_unclassified}",
        f"distinct non-bipartite graph classes={len(representatives)}",
        "",
        "aggregate family counts:",
    ]
    for bucket, count in sorted(aggregate_family_counts.items()):
        header_lines.append(f"  {bucket}: {count}")

    header_lines.append("")
    header_lines.append("graph-class/family counts:")
    for label, counts in sorted(aggregate_graph_family_counts.items()):
        header_lines.append(f"  {label}")
        for bucket, count in sorted(counts.items()):
            header_lines.append(f"    {bucket}: {count}")

    header = "\n".join(header_lines)
    print("\n" + header + "\n")

    print("GRAPH-CLASS REPRESENTATIVES")
    print("===========================")
    for i, data in enumerate(representatives, start=1):
        print(f"\nCLASS {i}")
        print(format_case(data))

    if all_unclassified:
        print("\nUNCLASSIFIED CASES")
        print("==================")
        for i, data in enumerate(all_unclassified, start=1):
            print(f"\nUNCLASSIFIED {i}")
            print(format_case(data))
    else:
        print("\nNo unclassified cases were found.")

    if args.show_covered:
        print("\nCOVERED EXAMPLES")
        print("================")
        for i, data in enumerate(covered_examples, start=1):
            print(f"\nEXAMPLE {i}")
            print(format_case(data))

    if args.output is not None:
        write_report(
            args.output,
            header,
            representatives,
            all_unclassified,
            covered_examples if args.show_covered else [],
        )
        print(f"\nReport written to {args.output}")

    if total_radical_failures or total_unclassified:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
classify_two_cubic_links.py

Exact, dependency-free classifier for active coordinate-link graphs arising
from one or two homogeneous cubic rotation-symmetric SANF generators.

For a normalized cubic generator [0,a,b] over Z/nZ, define

    E(a,b) = {
        {a,b},
        {-a,b-a},
        {-b,a-b}
    }.

For two generators [0,a,b] and [0,c,d], the active graph has edge set

    E(a,b) triangle E(c,d),

because repeated quadratic monomials cancel over F_2.

This script does all of the following for each explicitly tested modulus
n == 2 (mod 4):

  1. enumerates all normalized cubic SANF generators;
  2. enumerates all unordered pairs of distinct normalized generators;
  3. constructs the active graph exactly modulo n;
  4. checks whether the graph is bipartite;
  5. computes an exact canonical graph label by exhaustive relabeling;
  6. computes the adjacency kernel over F_2;
  7. verifies q_G(z)=0 for every z in ker(A_G);
  8. records one explicit representative for every non-bipartite
     graph-isomorphism class;
  9. records all endpoint equalities among the twelve symbolic endpoint
     expressions for that representative.

The output is designed to help derive the missing all-dimensional symbolic
classification.  It is an exact finite classifier for every modulus tested,
but a finite scan is not by itself an all-dimensional proof.

Examples:
    python3 classify_two_cubic_links.py --n 14
    python3 classify_two_cubic_links.py --scan 6 50
    python3 classify_two_cubic_links.py --scan 6 102 --output classes.txt
    python3 classify_two_cubic_links.py --n 14 --all-classes

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
Edge = Tuple[int, int]


@dataclass(frozen=True)
class GraphData:
    n: int
    generator_1: Generator
    generator_2: Optional[Generator]
    vertices: Tuple[int, ...]
    edges: Tuple[Edge, ...]
    canonical_label: str
    bipartite: bool
    kernel_basis_supports: Tuple[Tuple[int, ...], ...]
    kernel_dimension: int
    q_nonzero_on_radical: bool
    witness: Optional[Tuple[int, ...]]
    endpoint_equalities: Tuple[str, ...]

    @property
    def degree_multiset(self) -> Tuple[int, ...]:
        degree = Counter()
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        return tuple(sorted(degree.values()))


def normalize_generator(n: int, a: int, b: int) -> Generator:
    support = {0, a % n, b % n}
    if len(support) != 3:
        raise ValueError("A cubic generator must have three distinct residues.")

    reps: List[Generator] = []
    for t in support:
        translated = tuple(sorted((x - t) % n for x in support))
        reps.append(translated)  # type: ignore[arg-type]
    return min(reps)


def all_cubic_generators(n: int) -> List[Generator]:
    return sorted(
        {
            normalize_generator(n, a, b)
            for a in range(1, n)
            for b in range(a + 1, n)
        }
    )


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


def active_edges(
    n: int,
    generator_1: Generator,
    generator_2: Optional[Generator],
) -> Tuple[Edge, ...]:
    counts: Counter[Edge] = Counter(link_edges(n, generator_1))
    if generator_2 is not None:
        counts.update(link_edges(n, generator_2))
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
    Exact canonical label under graph isomorphism.

    Since active graphs have at most twelve vertices, exhaustive relabeling is
    feasible for the small number of non-bipartite cases we retain.  To keep
    the finite scan fast, vertices are first partitioned by degree, and only
    permutations within equal-degree cells are tested.
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
    for chosen in itertools.product(*cell_permutations):
        ordered_vertices = [v for block in chosen for v in block]
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
            (r for r in range(pivot_row, len(matrix))
             if (matrix[r] >> col) & 1),
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


def endpoint_expressions(
    n: int,
    g1: Generator,
    g2: Optional[Generator],
) -> List[Tuple[str, int]]:
    _, a, b = g1
    expressions = [
        ("a", a % n),
        ("b", b % n),
        ("-a", (-a) % n),
        ("b-a", (b-a) % n),
        ("-b", (-b) % n),
        ("a-b", (a-b) % n),
    ]

    if g2 is not None:
        _, c, d = g2
        expressions.extend(
            [
                ("c", c % n),
                ("d", d % n),
                ("-c", (-c) % n),
                ("d-c", (d-c) % n),
                ("-d", (-d) % n),
                ("c-d", (c-d) % n),
            ]
        )
    return expressions


def equality_classes(
    n: int,
    g1: Generator,
    g2: Optional[Generator],
) -> Tuple[str, ...]:
    expressions = endpoint_expressions(n, g1, g2)
    classes: Dict[int, List[str]] = defaultdict(list)
    for name, value in expressions:
        classes[value].append(name)

    relations = []
    for value, names in sorted(classes.items()):
        if len(names) > 1:
            relations.append(f"{' = '.join(names)} = {value} mod {n}")

    # Also record zero and the unique nonzero 2-torsion residue h=n/2.
    h = n // 2
    for name, value in expressions:
        if value == 0:
            relations.append(f"{name} = 0 mod {n}")
        elif value == h:
            relations.append(f"{name} = h={h} mod {n}")

    return tuple(sorted(set(relations)))


def analyze_case(
    n: int,
    g1: Generator,
    g2: Optional[Generator],
    compute_canonical: bool,
) -> GraphData:
    edges = active_edges(n, g1, g2)
    vertices = tuple(sorted({v for e in edges for v in e}))
    bip = is_bipartite(vertices, edges)

    canonical = (
        canonical_graph_label(vertices, edges)
        if compute_canonical
        else ""
    )

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

    return GraphData(
        n=n,
        generator_1=g1,
        generator_2=g2,
        vertices=vertices,
        edges=edges,
        canonical_label=canonical,
        bipartite=bip,
        kernel_basis_supports=basis_supports,
        kernel_dimension=len(basis),
        q_nonzero_on_radical=bad,
        witness=witness,
        endpoint_equalities=equality_classes(n, g1, g2),
    )


def classify_modulus(
    n: int,
    all_classes: bool = False,
) -> Tuple[List[GraphData], int, int]:
    if n % 4 != 2:
        raise ValueError("n must satisfy n == 2 mod 4.")
    if n < 6:
        raise ValueError("n must be at least 6.")

    generators = all_cubic_generators(n)
    representatives: Dict[str, GraphData] = {}
    total_cases = 0
    failures = 0

    cases: Iterable[Tuple[Generator, Optional[Generator]]] = itertools.chain(
        ((g, None) for g in generators),
        ((g1, g2) for g1, g2 in itertools.combinations(generators, 2)),
    )

    for g1, g2 in cases:
        total_cases += 1

        # First do the cheap analysis.  Canonical labeling is only needed for
        # classes we retain.
        preliminary = analyze_case(n, g1, g2, compute_canonical=False)
        if preliminary.q_nonzero_on_radical:
            failures += 1

        if not all_classes and preliminary.bipartite:
            continue

        full = analyze_case(n, g1, g2, compute_canonical=True)
        representatives.setdefault(full.canonical_label, full)

    return sorted(
        representatives.values(),
        key=lambda x: (len(x.vertices), len(x.edges), x.canonical_label),
    ), total_cases, failures


def format_class(index: int, data: GraphData) -> str:
    lines = [
        f"CLASS {index}",
        f"  modulus n              : {data.n}",
        f"  generator 1            : {data.generator_1}",
        f"  generator 2            : {data.generator_2}",
        f"  bipartite               : {data.bipartite}",
        f"  vertices                : {data.vertices}",
        f"  edges                   : {data.edges}",
        f"  degree multiset         : {data.degree_multiset}",
        f"  canonical label         : {data.canonical_label}",
        f"  kernel dimension        : {data.kernel_dimension}",
        f"  kernel basis supports   : {data.kernel_basis_supports}",
        f"  q nonzero on radical    : {data.q_nonzero_on_radical}",
        f"  witness                 : {data.witness}",
        "  endpoint equalities    :",
    ]
    if data.endpoint_equalities:
        lines.extend(f"    - {relation}" for relation in data.endpoint_equalities)
    else:
        lines.append("    - none")
    return "\n".join(lines)


def write_report(
    path: Path,
    header: str,
    classes: Sequence[GraphData],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip() + "\n\n")
        for i, data in enumerate(classes, start=1):
            handle.write(format_class(i, data))
            handle.write("\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify graph-isomorphism representatives of active coordinate-"
            "link graphs for one or two cubic SANF generators."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="classify one modulus")
    mode.add_argument(
        "--scan",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="scan all n == 2 mod 4 in the inclusive interval",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="include bipartite classes; default is non-bipartite only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the full representative report to a text file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n is not None:
        try:
            classes, total_cases, failures = classify_modulus(
                args.n,
                all_classes=args.all_classes,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        kind = "all" if args.all_classes else "non-bipartite"
        header = (
            f"n={args.n}\n"
            f"total exact cases={total_cases}\n"
            f"failures={failures}\n"
            f"{kind} graph-isomorphism classes={len(classes)}"
        )
        print(header)
        print()
        for i, data in enumerate(classes, start=1):
            print(format_class(i, data))
            print()

        if args.output is not None:
            write_report(args.output, header, classes)
            print(f"Report written to {args.output}")

        return 1 if failures else 0

    start, stop = args.scan
    if start > stop:
        start, stop = stop, start

    global_classes: Dict[str, GraphData] = {}
    total_cases = 0
    total_failures = 0
    tested_moduli = 0

    for n in range(start, stop + 1):
        if n < 6 or n % 4 != 2:
            continue

        classes, cases, failures = classify_modulus(
            n,
            all_classes=args.all_classes,
        )
        tested_moduli += 1
        total_cases += cases
        total_failures += failures

        for data in classes:
            global_classes.setdefault(data.canonical_label, data)

        print(
            f"n={n:4d}  cases={cases:10d}  failures={failures:3d}  "
            f"local classes={len(classes):3d}  "
            f"cumulative classes={len(global_classes):3d}"
        )

    representatives = sorted(
        global_classes.values(),
        key=lambda x: (len(x.vertices), len(x.edges), x.canonical_label),
    )

    kind = "all" if args.all_classes else "non-bipartite"
    header = (
        f"scanned moduli={tested_moduli}\n"
        f"total exact cases={total_cases}\n"
        f"failures={total_failures}\n"
        f"distinct {kind} graph-isomorphism classes={len(representatives)}"
    )

    print("\n" + header + "\n")
    for i, data in enumerate(representatives, start=1):
        print(format_class(i, data))
        print()

    if args.output is not None:
        write_report(args.output, header, representatives)
        print(f"Report written to {args.output}")

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

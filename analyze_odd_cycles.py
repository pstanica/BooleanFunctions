#!/usr/bin/env python3
"""
analyze_odd_cycles.py

Exact, dependency-free analyzer for odd cycles in active coordinate-link
graphs arising from pairs of cubic rotation-symmetric SANF generators.

For a normalized cubic generator [0,a,b] over Z/nZ, define

    E(a,b) = {
        {a,b},
        {-a,b-a},
        {-b,a-b}
    }.

For two generators, the active graph is the symmetric difference of their
link-edge sets.

The purpose of this script is to support the remaining symbolic proof by
checking, for every explicitly tested modulus n == 2 (mod 4), that:

  * every non-bipartite active graph contains a triangle;
  * no active graph contains an induced (chordless) 5-cycle;
  * every triangle is classified by its edge-color pattern;
  * every non-bipartite case is recognized by the candidate families

        F1(a): [0,a,2a], [0,a,3a]
        F3(a): [0,a,h],  [0,a,h+a],

    where h=n/2.

For every triangle, the program records:
  * the two generators;
  * the triangle vertices;
  * which generator contributes each triangle edge;
  * all anchored forms witnessing F1 or F3;
  * all endpoint equalities among the twelve link endpoints.

The program exits with nonzero status if it finds:
  * a non-bipartite graph with no triangle;
  * an induced 5-cycle;
  * an unclassified non-bipartite case.

This is an exact finite verification for every modulus tested.  It does not,
by itself, replace the all-dimensional symbolic proof.

Examples:
    python3 analyze_odd_cycles.py --n 14
    python3 analyze_odd_cycles.py --scan 6 102
    python3 analyze_odd_cycles.py --scan 6 102 --output odd_cycles_6_102.txt
    python3 analyze_odd_cycles.py --n 14 --show-triangles

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


Generator = Tuple[int, int, int]
OrderedForm = Tuple[int, int]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]
Cycle5 = Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class FamilyWitness:
    family: str
    parameter: int
    form_1: OrderedForm
    form_2: OrderedForm
    swapped: bool

    def describe(self, n: int) -> str:
        if self.family == "F1":
            branch = "5a=0" if (5 * self.parameter) % n == 0 else "5a!=0"
            return (
                f"F1 ({branch}), a={self.parameter}, "
                f"forms={self.form_1}; {self.form_2}, swapped={self.swapped}"
            )
        return (
            f"{self.family}, a={self.parameter}, "
            f"forms={self.form_1}; {self.form_2}, swapped={self.swapped}"
        )


@dataclass(frozen=True)
class TriangleRecord:
    n: int
    generator_1: Generator
    generator_2: Generator
    triangle: Triangle
    colored_edges: Tuple[Tuple[Edge, int], ...]
    color_pattern: str
    family_witnesses: Tuple[FamilyWitness, ...]
    endpoint_equalities: Tuple[str, ...]


@dataclass(frozen=True)
class BadCase:
    n: int
    generator_1: Generator
    generator_2: Generator
    edges: Tuple[Edge, ...]
    reason: str
    cycle5: Optional[Cycle5] = None


def normalize_generator(n: int, a: int, b: int) -> Generator:
    support = {0, a % n, b % n}
    if len(support) != 3:
        raise ValueError("A cubic generator must contain three distinct residues.")

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


def anchored_forms(n: int, generator: Generator) -> Tuple[OrderedForm, ...]:
    support = tuple(generator)
    forms: Set[OrderedForm] = set()

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


def colored_active_edges(
    n: int,
    g1: Generator,
    g2: Generator,
) -> Tuple[Tuple[Edge, ...], Dict[Edge, int]]:
    """
    Return active edges and their unique source color 0 or 1.

    Internal repetitions within a single link triple are first reduced modulo
    two.  Cross-generator repetitions then cancel.
    """
    count_1 = Counter(link_edges(n, g1))
    count_2 = Counter(link_edges(n, g2))

    active_1 = {e for e, multiplicity in count_1.items() if multiplicity % 2}
    active_2 = {e for e, multiplicity in count_2.items() if multiplicity % 2}

    active = active_1 ^ active_2
    colors = {e: (0 if e in active_1 else 1) for e in active}
    return tuple(sorted(active)), colors


def adjacency(vertices: Iterable[int], edges: Sequence[Edge]) -> Dict[int, Set[int]]:
    graph: Dict[int, Set[int]] = {v: set() for v in vertices}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    return graph


def is_bipartite(vertices: Sequence[int], edges: Sequence[Edge]) -> bool:
    graph = adjacency(vertices, edges)
    color: Dict[int, int] = {}

    for start in vertices:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def find_triangles(vertices: Sequence[int], edges: Sequence[Edge]) -> Tuple[Triangle, ...]:
    graph = adjacency(vertices, edges)
    triangles: Set[Triangle] = set()

    for u in vertices:
        for v in graph[u]:
            if v <= u:
                continue
            common = graph[u] & graph[v]
            for w in common:
                if w > v:
                    triangles.add((u, v, w))

    return tuple(sorted(triangles))


def canonical_cycle5(cycle: Sequence[int]) -> Cycle5:
    cycle = tuple(cycle)
    variants: List[Tuple[int, ...]] = []

    for direction in (cycle, tuple(reversed(cycle))):
        for shift in range(5):
            rotated = direction[shift:] + direction[:shift]
            variants.append(rotated)

    return min(variants)  # type: ignore[return-value]


def find_induced_five_cycles(
    vertices: Sequence[int],
    edges: Sequence[Edge],
) -> Tuple[Cycle5, ...]:
    """
    Find all induced 5-cycles using depth-first paths.

    Active graphs have at most six edges, so this is very small and fast.
    """
    graph = adjacency(vertices, edges)
    edge_set = set(edges)
    cycles: Set[Cycle5] = set()

    for start in vertices:
        stack: List[Tuple[int, Tuple[int, ...]]] = [(start, (start,))]

        while stack:
            current, path = stack.pop()

            if len(path) == 5:
                if start not in graph[current]:
                    continue

                # Check inducedness: among the five cycle vertices, the only
                # edges must be the five cycle edges.
                cycle_edges = {
                    normalize_edge(path[i], path[(i + 1) % 5])
                    for i in range(5)
                }
                induced_edges = {
                    normalize_edge(u, v)
                    for u, v in itertools.combinations(path, 2)
                    if normalize_edge(u, v) in edge_set
                }
                if induced_edges == cycle_edges:
                    cycles.add(canonical_cycle5(path))
                continue

            for neighbor in graph[current]:
                if neighbor == start:
                    continue
                if neighbor in path:
                    continue
                stack.append((neighbor, path + (neighbor,)))

    return tuple(sorted(cycles))


def canonical_color_pattern(colors: Sequence[int]) -> str:
    """
    Canonicalize a binary cyclic word under rotation, reversal, and color swap.
    """
    word = tuple(colors)
    variants: List[Tuple[int, ...]] = []

    for swap in (0, 1):
        changed = tuple(x ^ swap for x in word)
        for direction in (changed, tuple(reversed(changed))):
            for shift in range(len(changed)):
                variants.append(direction[shift:] + direction[:shift])

    best = min(variants)
    return "".join("R" if x == 0 else "B" for x in best)


def triangle_coloring(
    triangle: Triangle,
    edge_colors: Dict[Edge, int],
) -> Tuple[Tuple[Tuple[Edge, int], ...], str]:
    u, v, w = triangle
    ordered_edges = (
        normalize_edge(u, v),
        normalize_edge(v, w),
        normalize_edge(w, u),
    )
    colors = tuple(edge_colors[e] for e in ordered_edges)
    colored = tuple(zip(ordered_edges, colors))
    return colored, canonical_color_pattern(colors)


def match_families(
    n: int,
    g1: Generator,
    g2: Generator,
) -> Tuple[FamilyWitness, ...]:
    h = n // 2
    witnesses: Dict[
        Tuple[str, int, OrderedForm, OrderedForm, bool],
        FamilyWitness,
    ] = {}

    for swapped, first, second in (
        (False, g1, g2),
        (True, g2, g1),
    ):
        for form_1 in anchored_forms(n, first):
            a, b = form_1
            for form_2 in anchored_forms(n, second):
                c, d = form_2

                if (
                    b == (2 * a) % n
                    and c == a
                    and d == (3 * a) % n
                ):
                    witness = FamilyWitness("F1", a, form_1, form_2, swapped)
                    key = ("F1", a, form_1, form_2, swapped)
                    witnesses[key] = witness

                if (
                    b == h
                    and c == a
                    and d == (h + a) % n
                ):
                    witness = FamilyWitness("F3", a, form_1, form_2, swapped)
                    key = ("F3", a, form_1, form_2, swapped)
                    witnesses[key] = witness

    return tuple(
        sorted(
            witnesses.values(),
            key=lambda x: (
                x.family,
                x.parameter,
                x.swapped,
                x.form_1,
                x.form_2,
            ),
        )
    )


def endpoint_equalities(
    n: int,
    g1: Generator,
    g2: Generator,
) -> Tuple[str, ...]:
    _, a, b = g1
    _, c, d = g2

    expressions = (
        ("a", a % n),
        ("b", b % n),
        ("-a", (-a) % n),
        ("b-a", (b - a) % n),
        ("-b", (-b) % n),
        ("a-b", (a - b) % n),
        ("c", c % n),
        ("d", d % n),
        ("-c", (-c) % n),
        ("d-c", (d - c) % n),
        ("-d", (-d) % n),
        ("c-d", (c - d) % n),
    )

    by_value: Dict[int, List[str]] = defaultdict(list)
    for name, value in expressions:
        by_value[value].append(name)

    relations: List[str] = []
    for value, names in sorted(by_value.items()):
        if len(names) > 1:
            relations.append(f"{' = '.join(names)} = {value} mod {n}")

    h = n // 2
    for name, value in expressions:
        if value == h:
            relations.append(f"{name} = h={h} mod {n}")

    return tuple(sorted(set(relations)))


def analyze_modulus(
    n: int,
    collect_triangle_records: bool,
) -> Tuple[
    int,
    int,
    int,
    int,
    Counter[str],
    List[TriangleRecord],
    List[BadCase],
]:
    if n < 6 or n % 4 != 2:
        raise ValueError("n must satisfy n >= 6 and n == 2 mod 4.")

    generators = all_cubic_generators(n)

    total_pairs = 0
    nonbipartite_cases = 0
    total_triangles = 0
    induced_c5_count = 0
    pattern_counts: Counter[str] = Counter()
    triangle_records: List[TriangleRecord] = []
    bad_cases: List[BadCase] = []

    for g1, g2 in itertools.combinations(generators, 2):
        total_pairs += 1
        edges, colors = colored_active_edges(n, g1, g2)
        vertices = tuple(sorted({x for edge in edges for x in edge}))

        bipartite = is_bipartite(vertices, edges)
        triangles = find_triangles(vertices, edges)
        five_cycles = find_induced_five_cycles(vertices, edges)

        if five_cycles:
            induced_c5_count += len(five_cycles)
            for cycle in five_cycles:
                bad_cases.append(
                    BadCase(
                        n=n,
                        generator_1=g1,
                        generator_2=g2,
                        edges=edges,
                        reason="induced 5-cycle",
                        cycle5=cycle,
                    )
                )

        if bipartite:
            continue

        nonbipartite_cases += 1

        if not triangles:
            bad_cases.append(
                BadCase(
                    n=n,
                    generator_1=g1,
                    generator_2=g2,
                    edges=edges,
                    reason="non-bipartite graph with no triangle",
                )
            )

        witnesses = match_families(n, g1, g2)
        if not witnesses:
            bad_cases.append(
                BadCase(
                    n=n,
                    generator_1=g1,
                    generator_2=g2,
                    edges=edges,
                    reason="non-bipartite case not recognized by F1 or F3",
                )
            )

        equalities = endpoint_equalities(n, g1, g2)

        for triangle in triangles:
            total_triangles += 1
            colored_edges, pattern = triangle_coloring(triangle, colors)
            pattern_counts[pattern] += 1

            if collect_triangle_records:
                triangle_records.append(
                    TriangleRecord(
                        n=n,
                        generator_1=g1,
                        generator_2=g2,
                        triangle=triangle,
                        colored_edges=colored_edges,
                        color_pattern=pattern,
                        family_witnesses=witnesses,
                        endpoint_equalities=equalities,
                    )
                )

    return (
        total_pairs,
        nonbipartite_cases,
        total_triangles,
        induced_c5_count,
        pattern_counts,
        triangle_records,
        bad_cases,
    )


def format_triangle(record: TriangleRecord) -> str:
    lines = [
        f"n                    : {record.n}",
        f"generator 1          : {record.generator_1}",
        f"generator 2          : {record.generator_2}",
        f"triangle             : {record.triangle}",
        f"color pattern        : {record.color_pattern}",
        "colored edges        :",
    ]

    for edge, color in record.colored_edges:
        source = "generator 1" if color == 0 else "generator 2"
        lines.append(f"  - {edge}: {source}")

    lines.append("family witnesses     :")
    if record.family_witnesses:
        for witness in record.family_witnesses:
            lines.append(f"  - {witness.describe(record.n)}")
    else:
        lines.append("  - none")

    lines.append("endpoint equalities  :")
    if record.endpoint_equalities:
        for relation in record.endpoint_equalities:
            lines.append(f"  - {relation}")
    else:
        lines.append("  - none")

    return "\n".join(lines)


def format_bad_case(case: BadCase) -> str:
    lines = [
        f"reason       : {case.reason}",
        f"n            : {case.n}",
        f"generator 1  : {case.generator_1}",
        f"generator 2  : {case.generator_2}",
        f"active edges : {case.edges}",
    ]
    if case.cycle5 is not None:
        lines.append(f"5-cycle      : {case.cycle5}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze triangles and induced 5-cycles in active coordinate-link "
            "graphs for pairs of cubic SANF generators."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="analyze one modulus")
    mode.add_argument(
        "--scan",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="scan all n == 2 mod 4 in the inclusive interval",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="write a full report to this text file",
    )
    parser.add_argument(
        "--show-triangles",
        action="store_true",
        help="include representative triangle records",
    )
    parser.add_argument(
        "--max-triangle-records",
        type=int,
        default=40,
        help="maximum triangle records printed or written",
    )
    parser.add_argument(
        "--max-bad-cases",
        type=int,
        default=100,
        help="maximum bad cases printed or written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n is not None:
        moduli = [args.n]
    else:
        start, stop = args.scan
        if start > stop:
            start, stop = stop, start
        moduli = [n for n in range(start, stop + 1) if n >= 6 and n % 4 == 2]

    aggregate_pairs = 0
    aggregate_nonbipartite = 0
    aggregate_triangles = 0
    aggregate_c5 = 0
    aggregate_patterns: Counter[str] = Counter()
    all_triangle_records: List[TriangleRecord] = []
    all_bad_cases: List[BadCase] = []

    for n in moduli:
        try:
            (
                pairs,
                nonbipartite,
                triangles,
                c5_count,
                patterns,
                triangle_records,
                bad_cases,
            ) = analyze_modulus(n, collect_triangle_records=args.show_triangles)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        aggregate_pairs += pairs
        aggregate_nonbipartite += nonbipartite
        aggregate_triangles += triangles
        aggregate_c5 += c5_count
        aggregate_patterns.update(patterns)

        room_triangles = args.max_triangle_records - len(all_triangle_records)
        if room_triangles > 0:
            all_triangle_records.extend(triangle_records[:room_triangles])

        room_bad = args.max_bad_cases - len(all_bad_cases)
        if room_bad > 0:
            all_bad_cases.extend(bad_cases[:room_bad])

        print(
            f"n={n:4d}  pairs={pairs:10d}  "
            f"nonbipartite={nonbipartite:7d}  "
            f"triangles={triangles:7d}  "
            f"induced_C5={c5_count:4d}  "
            f"bad={len(bad_cases):4d}  "
            f"patterns={dict(sorted(patterns.items()))}"
        )

    header_lines = [
        f"scanned moduli={len(moduli)}",
        f"total unordered generator pairs={aggregate_pairs}",
        f"total non-bipartite cases={aggregate_nonbipartite}",
        f"total triangles={aggregate_triangles}",
        f"total induced 5-cycles={aggregate_c5}",
        f"bad cases={len(all_bad_cases)}",
        "triangle color-pattern counts:",
    ]

    for pattern, count in sorted(aggregate_patterns.items()):
        header_lines.append(f"  {pattern}: {count}")

    header = "\n".join(header_lines)
    print("\n" + header)

    if all_bad_cases:
        print("\nBAD CASES")
        print("=========")
        for index, case in enumerate(all_bad_cases, start=1):
            print(f"\nBAD CASE {index}")
            print(format_bad_case(case))
    else:
        print("\nPASS: no induced 5-cycle, no triangle-free non-bipartite case,")
        print("and no non-bipartite case outside F1 or F3 was found.")

    if args.show_triangles and all_triangle_records:
        print("\nTRIANGLE RECORDS")
        print("================")
        for index, record in enumerate(all_triangle_records, start=1):
            print(f"\nTRIANGLE {index}")
            print(format_triangle(record))

    if args.output is not None:
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(header + "\n\n")

            if all_bad_cases:
                handle.write("BAD CASES\n")
                handle.write("=========\n\n")
                for index, case in enumerate(all_bad_cases, start=1):
                    handle.write(f"BAD CASE {index}\n")
                    handle.write(format_bad_case(case))
                    handle.write("\n\n")
            else:
                handle.write(
                    "PASS: no induced 5-cycle, no triangle-free "
                    "non-bipartite case, and no non-bipartite case outside "
                    "F1 or F3 was found.\n\n"
                )

            if args.show_triangles and all_triangle_records:
                handle.write("TRIANGLE RECORDS\n")
                handle.write("================\n\n")
                for index, record in enumerate(all_triangle_records, start=1):
                    handle.write(f"TRIANGLE {index}\n")
                    handle.write(format_triangle(record))
                    handle.write("\n\n")

        print(f"\nReport written to {args.output}")

    return 1 if all_bad_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())

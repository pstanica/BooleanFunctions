#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

constexpr int size = 256;
using Table = std::array<unsigned char, size>;
using Adjacency = std::array<std::array<bool, size>, size>;

std::vector<int> orbit(int mask) {
    std::vector<int> result;
    int shifted = mask;
    do {
        result.push_back(shifted);
        shifted = ((shifted << 1) & 255) | (shifted >> 7);
    } while (shifted != mask);
    return result;
}

bool bent(const Table& f) {
    int weight = 0;
    for (int v : f) weight += v;
    if (weight != 120 && weight != 136) return false;
    std::array<int, size> w{};
    for (int x = 0; x < size; ++x) w[x] = 1 - 2 * f[x];
    for (int step = 1; step < size; step *= 2)
        for (int i = 0; i < size; i += 2 * step)
            for (int j = i; j < i + step; ++j) {
                int u = w[j], v = w[j + step];
                w[j] = u + v;
                w[j + step] = u - v;
            }
    return std::all_of(w.begin(), w.end(),
                       [](int v) { return v == 16 || v == -16; });
}

int second(const Table& f, int a, int b, int x) {
    return f[x] ^ f[x ^ a] ^ f[x ^ b] ^ f[x ^ a ^ b];
}

bool compatible(const Table& f, int a, int b, int value = 0) {
    for (int x = 0; x < size; ++x)
        if (std::popcount(unsigned(x)) <= 2 && second(f, a, b, x) != value)
            return false;
    return true;
}

bool extend(const Adjacency& adj, const std::vector<int>& candidates,
            int dimension) {
    if (dimension == 4) return true;
    if (candidates.size() < unsigned(4 - dimension)) return false;
    for (int a : candidates) {
        int pivot = std::bit_width(unsigned(a)) - 1;
        if (pivot > 4 + dimension) break;
        std::vector<int> next;
        for (int b : candidates)
            if (b >= (2 << pivot) && !(b & (1 << pivot)) && adj[a][b])
                next.push_back(b);
        if (extend(adj, next, dimension + 1)) return true;
    }
    return false;
}

bool half_turn(const Table& f) {
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (!compatible(f, 17 << i, 17 << j)) return false;
    return true;
}

bool membership(const Table& f) {
    Adjacency adj{};
    std::vector<int> candidates;
    for (int a = 1; a < size; ++a) {
        candidates.push_back(a);
        for (int b = a + 1; b < size; ++b)
            adj[a][b] = compatible(f, a, b);
    }
    return extend(adj, candidates, 0);
}

int main() {
    std::vector<int> reps;
    std::vector<Table> tables;
    std::vector<std::vector<int>> orbits;
    for (int degree = 2; degree <= 4; ++degree)
        for (int mask = 1; mask < size; ++mask) {
            if (std::popcount(unsigned(mask)) != degree) continue;
            auto terms = orbit(mask);
            if (mask != *std::min_element(terms.begin(), terms.end()))
                continue;
            Table f{};
            for (int x = 0; x < size; ++x)
                for (int term : terms) f[x] ^= (x & term) == term;
            reps.push_back(mask);
            tables.push_back(f);
            orbits.push_back(terms);
        }
    assert(reps.size() == 21);
    auto seed = [&](int last) {
        Table f{};
        for (int mask : {17, 7, 19, 67, 23, last}) {
            auto terms = orbit(mask);
            int rep = *std::min_element(terms.begin(), terms.end());
            auto k = std::find(reps.begin(), reps.end(), rep) - reps.begin();
            for (int x = 0; x < size; ++x) f[x] ^= tables[k][x];
        }
        return f;
    };
    const Table before = seed(83), after = seed(39);
    assert(bent(before) && bent(after) && half_turn(before));
    for (int a = 1; a < size; ++a)
        for (int b = a + 1; b < size; ++b)
            assert(!compatible(after, a, b, second(after, a, b, 0)));

    constexpr unsigned count = 1u << 21;
    std::vector<unsigned char> kind(count);
    Table f{};
    int total = 0, inside = 0, halves = 0;
    for (unsigned i = 1; i < count; ++i) {
        unsigned changed = std::countr_zero(i);
        unsigned mask = i ^ (i >> 1);
        for (int x = 0; x < size; ++x) f[x] ^= tables[changed][x];
        if (!(mask >> 11) || !bent(f)) continue;
        bool half = half_turn(f);
        bool member = half || membership(f);
        kind[mask] = member ? 2 : 1;
        ++total;
        inside += member;
        halves += half;
    }
    std::vector<std::pair<int,int>> moves;
    for (int a = 11; a < 21; ++a)
        for (int b = a + 1; b < 21; ++b)
            for (int candidate : orbits[b])
                if (std::popcount(unsigned(reps[a] ^ candidate)) == 2) {
                    moves.emplace_back(a, b);
                    break;
                }

    std::array<int, 3> edges{};
    std::set<unsigned> inside_ends, outside_ends;
    std::set<std::pair<int,int>> crossing_types;
    for (unsigned x = 0; x < count; ++x) {
        if (!kind[x]) continue;
        for (auto [a, b] : moves) {
            if ((((x >> a) ^ (x >> b)) & 1u) == 0) continue;
            unsigned y = x ^ (1u << a) ^ (1u << b);
            if (y <= x || !kind[y]) continue;
            ++edges[kind[x] + kind[y] - 2];
            if (kind[x] != kind[y]) {
                unsigned source = kind[x] == 2 ? x : y;
                inside_ends.insert(source);
                outside_ends.insert(source == x ? y : x);
                if (source & (1u << a)) crossing_types.emplace(a, b);
                else crossing_types.emplace(b, a);
            }
        }
    }
    assert(total == 3712 && inside == 768 && halves == 768);
    assert((edges == std::array<int,3>{992, 704, 0}));
    assert(inside_ends.size() == 336 && outside_ends.size() == 448);
    assert(moves.size() == 38 && crossing_types.size() == 20);
    std::cout << "Seed pair: both bent; half-turn basis verified for f; "
                 "no constant second difference for independent directions of g.\n"
              << "Normalized quartic rotation symmetric bent functions: " << total << '\n'
              << "Inside M#: " << inside << '\n'
              << "Outside M#: " << total - inside << '\n'
              << "Half-turn M-subspace verified: " << halves << '\n'
              << "Edges inside/inside: " << edges[2] << '\n'
              << "Edges inside/outside: " << edges[1] << '\n'
              << "Edges outside/outside: " << edges[0] << '\n'
              << "Inside vertices on crossing edges: " << inside_ends.size() << '\n'
              << "Outside vertices on crossing edges: " << outside_ends.size() << '\n'
              << "Admissible unordered pairs of quartic orbits: " << moves.size() << '\n'
              << "Directed orbit replacements realized on crossing edges: "
              << crossing_types.size() << '\n';
}

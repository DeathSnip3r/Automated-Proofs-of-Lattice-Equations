#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "term.hpp"

namespace lat {

// Tri-valued cell for DP table
enum class Tval : int8_t { Unknown = -1, False = 0, True = 1 };

struct FreeseStats {
    uint64_t recursive_calls = 0;
    uint64_t memo_hits = 0;
    uint64_t branch_attempts = 0;
    uint64_t branch_successes = 0;
    uint32_t max_stack = 0;
};

struct Freese {
    Interner* I;
    FreeseStats stats;
    bool use_memo = true;

    explicit Freese(Interner* interner) : I(interner) {}

    bool leq(int u, int v);

private:
    // Memoization table: key is (u << 32) | v
    std::unordered_map<uint64_t, Tval> memo_;

    Tval rec(int u, int v, uint32_t depth);
    bool branch(int u, int v, uint32_t depth);
};

} // namespace lat

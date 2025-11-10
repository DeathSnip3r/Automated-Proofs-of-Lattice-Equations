#pragma once
#include <cstdint>
#include "term.hpp"

namespace lat {

struct Whitman {
    Interner* I;                       // not owning
    uint64_t pairs_visited = 0;
    uint32_t max_stack     = 0;
    uint64_t branch_attempts = 0;
    uint64_t branch_right_attempts = 0;
    uint64_t branch_left_attempts  = 0;
    uint64_t branch_successes      = 0;
    uint64_t join_decompositions   = 0;
    uint64_t meet_decompositions   = 0;

    explicit Whitman(Interner* interner): I(interner) {}
    bool leq(int u, int v);

private:
    bool rec(int u, int v, uint32_t depth);

    // Whitman branching helper (free function) is allowed to call rec(...)
    friend bool whitman_branch(Whitman& W, int u, int v, uint32_t depth);
};

} // namespace lat

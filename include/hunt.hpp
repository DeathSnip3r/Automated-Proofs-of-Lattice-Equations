#pragma once
#include <cstdint>
#include "term.hpp"

namespace lat {

struct HuntStats {
    uint64_t cells_evaluated = 0;
    uint64_t and_checks = 0;
    uint64_t or_checks  = 0;
    int RU = 0; // |sub(u)|
    int RV = 0; // |sub(v)|
    int max_hU = 0, max_hV = 0;
};

// Hunt–Rosenkrantz–Bloniarz DP over subterm pairs ordered by heights (variadic).
bool hunt_leq(int u, int v, const Interner& I, HuntStats* stats=nullptr);

} // namespace lat

#pragma once
#include <cstdint>
#include "term.hpp"

namespace lat {

struct CosmaStats {
    uint64_t arcs_enqueued = 0;   // distinct edges added to R
    uint64_t arcs_processed = 0;  // queue pops
    uint64_t max_queue = 0;       // peak queue size
    int N = 0;                    // number of subterms in scope (|V|)
};

// Cosmadakis-style closure on variadic DAG subterms (no new terms created).
bool leq_cosma(int u, int v, const Interner& I, CosmaStats* stats=nullptr);

} // namespace lat

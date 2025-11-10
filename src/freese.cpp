#include "freese.hpp"
#include <vector>
#include <algorithm>

namespace lat {

// Helper to flatten meet/join factors for branching
static void collect_meet(int t, const Interner& I, std::vector<int>& out) {
    const Node& nd = I.nodes[t];
    if (nd.kind == Kind::Meet) {
        for (int c : nd.kids) collect_meet(c, I, out);
    } else {
        out.push_back(t);
    }
}

static void collect_join(int t, const Interner& I, std::vector<int>& out) {
    const Node& nd = I.nodes[t];
    if (nd.kind == Kind::Join) {
        for (int c : nd.kids) collect_join(c, I, out);
    } else {
        out.push_back(t);
    }
}

static inline void dedup(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

bool Freese::leq(int u, int v) {
    memo_.clear();
    stats = {};
    use_memo = I && I->canonicalize;
    Tval result = rec(u, v, 1);
    return result == Tval::True;
}

Tval Freese::rec(int u, int v, uint32_t depth) {
    stats.recursive_calls++;
    if (depth > stats.max_stack) stats.max_stack = depth;

    if (u == v) return Tval::True;

    uint64_t key = 0;
    if (use_memo) {
        key = ((uint64_t)u << 32) | (uint64_t)v;
        auto it = memo_.find(key);
        if (it != memo_.end()) {
            stats.memo_hits++;
            return it->second;
        }
    }

    const Node& A = I->nodes[u];
    const Node& B = I->nodes[v];

    Tval result = Tval::Unknown;

    // Base case: u is a variable
    if (A.kind == Kind::Var) {
        if (B.kind == Kind::Join) { // v = y1 + y2... => u <= v iff u <= yi for some i
            stats.branch_attempts++;
            for (int b : B.kids) {
                if (rec(u, b, depth + 1) == Tval::True) {
                    stats.branch_successes++;
                    result = Tval::True;
                    break;
                }
            }
            if (result == Tval::Unknown) result = Tval::False;
        } else {
            result = Tval::False; // var <= var (handled by u==v) or var <= meet
        }
    }
    // Base case: v is a variable
    else if (B.kind == Kind::Var) {
        if (A.kind == Kind::Meet) { // u = x1 * x2... => u <= v iff xi <= v for some i
            stats.branch_attempts++;
            for (int a : A.kids) {
                if (rec(a, v, depth + 1) == Tval::True) {
                    stats.branch_successes++;
                    result = Tval::True;
                    break;
                }
            }
            if (result == Tval::Unknown) result = Tval::False;
        } else if (A.kind == Kind::Join) { // u = x1 + x2... => u <= v iff all xi <= v
            result = Tval::True;
            for (int a : A.kids) {
                if (rec(a, v, depth + 1) != Tval::True) {
                    result = Tval::False;
                    break;
                }
            }
        } else {
            result = Tval::False;
        }
    }
    // u is Join or Meet, v is Join or Meet
    else if (A.kind == Kind::Join) { // (x1 + x2) <= v  <=>  x1 <= v AND x2 <= v
        result = Tval::True;
        for (int a : A.kids) {
            if (rec(a, v, depth + 1) == Tval::False) {
                result = Tval::False;
                break;
            }
        }
    }
    else if (B.kind == Kind::Meet) { // u <= (y1 * y2)  <=>  u <= y1 AND u <= y2
        result = Tval::True;
        for (int b : B.kids) {
            if (rec(u, b, depth + 1) == Tval::False) {
                result = Tval::False;
                break;
            }
        }
    }
    // Remaining case: u is a Meet, v is a Join. Apply Whitman's condition (W).
    else if (A.kind == Kind::Meet && B.kind == Kind::Join) {
        result = branch(u, v, depth) ? Tval::True : Tval::False;
    }
    else {
        // Should be unreachable if logic is exhaustive
        result = Tval::False;
    }

    if (use_memo) memo_[key] = result;
    return result;
}

bool Freese::branch(int u, int v, uint32_t depth) {
    stats.branch_attempts++;
    std::vector<int> U, V;
    collect_meet(u, *I, U);
    collect_join(v, *I, V);
    if (I->canonicalize) {
        dedup(U);
        dedup(V);
    }

    // Branch on right: u <= vj for some j
    for (int vj : V) {
        if (rec(u, vj, depth + 1) == Tval::True) {
            stats.branch_successes++;
            return true;
        }
    }
    // Branch on left: ui <= v for some i
    for (int ui : U) {
        if (rec(ui, v, depth + 1) == Tval::True) {
            stats.branch_successes++;
            return true;
        }
    }
    return false;
}

// Keep old free functions for compatibility if needed, but they are now obsolete.
// The main runner should be updated to use the Freese class.
bool leq_freese(int u, int v, const Interner& I, FreeseStats& S) {
    Freese freese_instance(const_cast<Interner*>(&I));
    bool result = freese_instance.leq(u, v);
    S = freese_instance.stats;
    return result;
}

bool leq_freese(int u, int v, const Interner& I) {
    Freese freese_instance(const_cast<Interner*>(&I));
    return freese_instance.leq(u, v);
}

} // namespace lat

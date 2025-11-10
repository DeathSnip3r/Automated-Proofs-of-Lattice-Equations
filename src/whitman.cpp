#include "whitman.hpp"
#include <vector>
#include <algorithm>

namespace lat {

// Flatten top-level meet factors of t into out
static void collect_meet(int t, const Interner& I, std::vector<int>& out){
    const Node& nd = I.nodes[t];
    if (nd.kind == Kind::Meet){
        for (int c : nd.kids) collect_meet(c, I, out);
    } else {
        out.push_back(t);
    }
}

// Flatten top-level join summands of t into out
static void collect_join(int t, const Interner& I, std::vector<int>& out){
    const Node& nd = I.nodes[t];
    if (nd.kind == Kind::Join){
        for (int c : nd.kids) collect_join(c, I, out);
    } else {
        out.push_back(t);
    }
}

static inline void dedup(std::vector<int>& v){
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// --- Whitman branching (exponential):
// (∧U) ≤ (∨V)  ⇔  (∨_{vj∈V} [u ≤ vj])  ∨  (∨_{ui∈U} [ui ≤ v])
bool whitman_branch(Whitman& W, int u, int v, uint32_t depth){
    std::vector<int> U, V;
    collect_meet(u, *W.I, U);
    collect_join(v, *W.I, V);
    dedup(U); dedup(V);

    // No branching if both are singletons (prevents (u,v) -> (u,v) “branches”)
    if (U.size() <= 1 && V.size() <= 1) return false;

    ++W.branch_attempts;

    // 1) Prefer branching on the right: u ≤ vj (strictly reduces v)
    for (int vj : V){
        if (vj == v) continue;              // avoid self-recursion
        ++W.branch_right_attempts;
        if (W.rec(u, vj, depth+1)){
            ++W.branch_successes;
            return true;
        }
    }
    // 2) Then branch on the left: ui ≤ v (strictly reduces u)
    for (int ui : U){
        if (ui == u) continue;              // avoid self-recursion
        ++W.branch_left_attempts;
        if (W.rec(ui, v, depth+1)){
            ++W.branch_successes;
            return true;
        }
    }
    return false;
}


bool Whitman::leq(int u, int v){
    pairs_visited = 0;
    max_stack     = 0;
    branch_attempts = 0;
    branch_right_attempts = 0;
    branch_left_attempts = 0;
    branch_successes = 0;
    join_decompositions = 0;
    meet_decompositions = 0;
    return rec(u, v, 1);
}

bool Whitman::rec(int u, int v, uint32_t depth){
    ++pairs_visited;
    if (depth > max_stack) max_stack = depth;

    if (u == v) return true;

    const Node& A = I->nodes[u];
    const Node& B = I->nodes[v];

    // Base case: u is a variable
    if (A.kind == Kind::Var) {
        if (B.kind == Kind::Join) { // v = y1 + y2... => u <= v iff u <= yi for some i
            ++branch_attempts;
            for (int b : B.kids) {
                ++branch_right_attempts;
                if (rec(u, b, depth + 1)) {
                    ++branch_successes;
                    return true;
                }
            }
        }
        return false; // var <= var (handled by u==v) or var <= meet
    }

    // Base case: v is a variable
    if (B.kind == Kind::Var) { // u is join or meet
        if (A.kind == Kind::Meet) { // u = x1 * x2... => u <= v iff xi <= v for some i
            ++branch_attempts;
            for (int a : A.kids) {
                ++branch_left_attempts;
                if (rec(a, v, depth + 1)) {
                    ++branch_successes;
                    return true;
                }
            }
        }
        return false; // join <= var
    }

    // u is Join or Meet, v is Join or Meet
    if (A.kind == Kind::Join) { // (x1 + x2) <= v  <=>  x1 <= v AND x2 <= v
        join_decompositions += A.kids.size();
        for (int a : A.kids) {
            if (!rec(a, v, depth + 1)) return false;
        }
        return true;
    }

    if (B.kind == Kind::Meet) { // u <= (y1 * y2)  <=>  u <= y1 AND u <= y2
        meet_decompositions += B.kids.size();
        for (int b : B.kids) {
            if (!rec(u, b, depth + 1)) return false;
        }
        return true;
    }

    // Remaining case: u is a Meet, v is a Join. Apply Whitman's condition (W).
    // (x1 * x2) <= (y1 + y2) <=> (x1 <= v or x2 <= v) or (u <= y1 or u <= y2)
    if (A.kind == Kind::Meet && B.kind == Kind::Join) {
        return whitman_branch(*this, u, v, depth);
    }

    // Should be unreachable
    return false;
}

} // namespace lat

#include "cosmadakis.hpp"
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace lat {

static void enumerate_reachable(const Interner& I, int root, std::vector<int>& out){
    std::vector<char> vis(I.nodes.size(),0);
    std::function<void(int)> dfs=[&](int x){
        if (vis[x]) return; vis[x]=1;
        for (int k : I.nodes[x].kids) dfs(k);
        out.push_back(x);
    };
    dfs(root);
}

// tiny hash for vector<int> keys (child-sets)
struct VecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (int x : v){
            uint64_t k = static_cast<uint64_t>(static_cast<uint32_t>(x));
            h ^= k + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        }
        // avalanche
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }
};
struct VecEq { bool operator()(const std::vector<int>& a, const std::vector<int>& b) const noexcept { return a==b; } };

bool leq_cosma(int u, int v, const Interner& I, CosmaStats* stats){
    // 1) Build initial local scope V0 = sub(u) ∪ sub(v)
    std::vector<int> U, W;
    enumerate_reachable(I, u, U);
    enumerate_reachable(I, v, W);

    std::unordered_map<int,int> idx_of;
    idx_of.reserve(U.size() + W.size());
    std::vector<int> nodes; nodes.reserve(U.size() + W.size());
    auto add_if_absent = [&](int id)->bool{
        if (idx_of.find(id)!=idx_of.end()) return false;
        int k = (int)nodes.size();
        idx_of.emplace(id,k);
        nodes.push_back(id);
        return true;
    };
    for (int x: U) add_if_absent(x);
    for (int x: W) add_if_absent(x);

    // 1b) Context lifting: if a Join/Meet is in scope, include ALL its children.
    // Repeat until saturated (DAG so this terminates quickly).
    bool grew = true;
    while (grew){
        grew = false;
        for (int i = 0; i < (int)nodes.size(); ++i){
            const Node& nd = I.nodes[nodes[i]];
            if (nd.kind == Kind::Join || nd.kind == Kind::Meet){
                for (int g : nd.kids) grew |= add_if_absent(g);
            }
        }
    }

    const int N = (int)nodes.size();
    if (stats){
        stats->N = N;
        stats->arcs_enqueued = 0;
        stats->arcs_processed = 0;
        stats->max_queue = 0;
    }

    // 2) Classify joins/meets; build child incidence
    std::vector<int> join_nodes, meet_nodes;
    std::vector<std::vector<int>> J_children, M_children;
    std::vector<std::vector<int>> child_to_J(N), child_to_M(N);

    for (int li=0; li<N; ++li){
        const Node& nd = I.nodes[nodes[li]];
        if (nd.kind==Kind::Join){
            int pos = (int)join_nodes.size();
            join_nodes.push_back(li);
            std::vector<int> kids; kids.reserve(nd.kids.size());
            for (int g : nd.kids){
                auto it = idx_of.find(g);
                if (it!=idx_of.end()){
                    kids.push_back(it->second);
                    child_to_J[it->second].push_back(pos);
                }
            }
            J_children.push_back(std::move(kids));
        } else if (nd.kind==Kind::Meet){
            int pos = (int)meet_nodes.size();
            meet_nodes.push_back(li);
            std::vector<int> kids; kids.reserve(nd.kids.size());
            for (int g : nd.kids){
                auto it = idx_of.find(g);
                if (it!=idx_of.end()){
                    kids.push_back(it->second);
                    child_to_M[it->second].push_back(pos);
                }
            }
            M_children.push_back(std::move(kids));
        }
    }

    const int Jn = (int)join_nodes.size();
    const int Mn = (int)meet_nodes.size();

    // 3) Need counters for the two equivalences
    std::vector<int> join_need((size_t)Jn * N, 0);
    std::vector<int> meet_need((size_t)N  * Mn, 0);
    for (int j=0;j<Jn;++j){
        int need = (int)J_children[j].size();
        for (int b=0;b<N;++b) join_need[(size_t)j*N + b] = need;
    }
    for (int m=0;m<Mn;++m){
        int need = (int)M_children[m].size();
        for (int a=0;a<N;++a) meet_need[(size_t)a*Mn + m] = need;
    }

    // 4) Relation R as adjacency + succ/preds
    std::vector<uint8_t> has((size_t)N * N, 0);
    std::vector<std::vector<int>> succ(N), preds(N);

    auto add_edge = [&](int a, int b, std::vector<std::pair<int,int>>& Q){
        size_t id = (size_t)a*N + b;
        if (has[id]) return false;
        has[id] = 1;
        succ[a].push_back(b);
        preds[b].push_back(a);
        Q.emplace_back(a,b);
        if (stats){
            stats->arcs_enqueued++;
            stats->max_queue = std::max<uint64_t>(stats->max_queue, (uint64_t)Q.size());
        }
        return true;
    };

    // 5) Initialize worklist
    std::vector<std::pair<int,int>> Q;
    Q.reserve((size_t)N*N/4 + 16);

    // Reflexivity
    for (int i=0;i<N;++i) add_edge(i,i,Q);

    // Structural seeds
    for (int j=0;j<Jn;++j){
        int J = join_nodes[j];
        for (int c : J_children[j]) add_edge(c, J, Q); // child ≤ join
    }
    for (int m=0;m<Mn;++m){
        int M = meet_nodes[m];
        for (int c : M_children[m]) add_edge(M, c, Q); // meet ≤ child
    }

    // 6) Closure
    size_t head = 0;
    while (head < Q.size()){
        auto [a,b] = Q[head++];
    if (stats) stats->arcs_processed++;

        // Transitivity
        for (int z : preds[a]) add_edge(z, b, Q);
        for (int z : succ[b]) add_edge(a, z, Q);

        // Join-left equivalence
        if (!child_to_J[a].empty()){
            for (int jpos : child_to_J[a]){
                int &need = join_need[(size_t)jpos*N + b];
                if (need>0 && --need==0){
                    int J = join_nodes[jpos];
                    add_edge(J, b, Q);
                }
            }
        }
        // Meet-right equivalence
        if (!child_to_M[b].empty()){
            for (int mpos : child_to_M[b]){
                int &need = meet_need[(size_t)a*Mn + mpos];
                if (need>0 && --need==0){
                    int M = meet_nodes[mpos];
                    add_edge(a, M, Q);
                }
            }
        }
    }

    int ul = idx_of.at(u);
    int vl = idx_of.at(v);
    return has[(size_t)ul*N + vl] != 0;
}


} // namespace lat

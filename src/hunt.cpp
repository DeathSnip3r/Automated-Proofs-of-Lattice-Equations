#include "hunt.hpp"
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cstdint>

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

static void compute_heights(const Interner& I, int u, int v, std::vector<int>& H){
    H.assign(I.nodes.size(), -1);
    std::function<int(int)> go = [&](int id)->int{
        int &h = H[id];
        if (h>=0) return h;
        const Node& n = I.nodes[id];
        if (n.kids.empty()) return h=0;
        int mx = 0;
        for (int k : n.kids) mx = std::max(mx, 1 + go(k));
        return h = mx;
    };
    go(u); go(v);
}

static inline size_t bit_index(size_t row_stride_bits, int a, int b){
    return (size_t)a * row_stride_bits + (size_t)b;
}
static inline bool get_bit(const std::vector<uint64_t>& bits, size_t idx){
    return (bits[idx >> 6] >> (idx & 63)) & 1ull;
}
static inline void set_bit(std::vector<uint64_t>& bits, size_t idx){
    bits[idx >> 6] |= (1ull << (idx & 63));
}

bool hunt_leq(int u, int v, const Interner& I, HuntStats* stats){
    std::vector<int> U, V;
    enumerate_reachable(I, u, U);
    enumerate_reachable(I, v, V);
    const int RU = (int)U.size(), RV = (int)V.size();
    if (stats){
        stats->RU = RU;
        stats->RV = RV;
        stats->cells_evaluated = 0;
        stats->and_checks = 0;
        stats->or_checks = 0;
    }

    std::unordered_map<int,int> u_idx, v_idx;
    u_idx.reserve(RU*2); v_idx.reserve(RV*2);
    for (int i=0;i<RU;++i) u_idx.emplace(U[i], i);
    for (int j=0;j<RV;++j) v_idx.emplace(V[j], j);

    std::vector<int> H; compute_heights(I,u,v,H);
    int max_hU=0, max_hV=0; std::vector<int> hU(RU), hV(RV);
    for (int i=0;i<RU;++i){ hU[i]=H[U[i]]; max_hU = std::max(max_hU,hU[i]); }
    for (int j=0;j<RV;++j){ hV[j]=H[V[j]]; max_hV = std::max(max_hV,hV[j]); }
    if (stats){ stats->max_hU=max_hU; stats->max_hV=max_hV; }

    std::vector<std::vector<int>> Ukids(RU), Vkids(RV);
    std::vector<Kind> Ukind(RU), Vkind(RV);
    for (int i=0;i<RU;++i){
        const Node& n = I.nodes[U[i]];
        Ukind[i]=n.kind;
        for (int g : n.kids){ auto it=u_idx.find(g); if (it!=u_idx.end()) Ukids[i].push_back(it->second); }
    }
    for (int j=0;j<RV;++j){
        const Node& n = I.nodes[V[j]];
        Vkind[j]=n.kind;
        for (int g : n.kids){ auto it=v_idx.find(g); if (it!=v_idx.end()) Vkids[j].push_back(it->second); }
    }

    std::vector<std::vector<int>> BU(max_hU+1), BV(max_hV+1);
    for (int i=0;i<RU;++i) BU[hU[i]].push_back(i);
    for (int j=0;j<RV;++j) BV[hV[j]].push_back(j);

    const size_t total_bits = (size_t)RU * (size_t)RV;
    std::vector<uint64_t> R((total_bits + 63) >> 6, 0ull);

    auto test = [&](int a, int b){ return get_bit(R, bit_index((size_t)RV, a, b)); };
    auto mark = [&](int a, int b){ set_bit(R, bit_index((size_t)RV, a, b)); };

    for (int hu=0; hu<=max_hU; ++hu){
        for (int hv=0; hv<=max_hV; ++hv){
            for (int a : BU[hu]){
                for (int b : BV[hv]){
                    const size_t idx = bit_index((size_t)RV, a, b);
                    if (get_bit(R, idx)){ if (stats) stats->cells_evaluated++; continue; }

                    if (U[a]==V[b]){ set_bit(R, idx); if (stats) stats->cells_evaluated++; continue; }

                    bool ans=false;
                    const Kind ak = Ukind[a], bk = Vkind[b];

                    if (!ans && ak==Kind::Var && bk==Kind::Var){
                        const Node& A = I.nodes[U[a]];
                        const Node& B = I.nodes[V[b]];
                        ans = (A.var == B.var);
                    }

                    if (!ans && ak==Kind::Join){
                        bool ok=true;
                        if (stats) stats->and_checks += (uint64_t)Ukids[a].size();
                        for (int ai : Ukids[a]) if (!test(ai,b)){ ok=false; break; }
                        ans = ok;
                    }

                    if (!ans && ak==Kind::Meet){
                        if (stats) stats->or_checks += (uint64_t)Ukids[a].size();
                        for (int ai : Ukids[a]) if (test(ai,b)){ ans=true; break; }
                    }

                    if (!ans && bk==Kind::Join){
                        if (stats) stats->or_checks += (uint64_t)Vkids[b].size();
                        for (int bj : Vkids[b]) if (test(a,bj)){ ans=true; break; }
                    }

                    if (!ans && bk==Kind::Meet){
                        bool ok=true;
                        if (stats) stats->and_checks += (uint64_t)Vkids[b].size();
                        for (int bj : Vkids[b]) if (!test(a,bj)){ ok=false; break; }
                        ans = ok;
                    }

                    if (ans) set_bit(R, idx);
                    if (stats) stats->cells_evaluated++;
                }
            }
        }
    }

    return test(u_idx.at(u), v_idx.at(v));
}

} // namespace lat

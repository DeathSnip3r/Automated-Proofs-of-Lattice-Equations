#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <memory>

namespace lat {

enum class Kind { Var, Join, Meet };

struct Node {
    Kind kind;
    std::vector<int> kids; // empty for Var
    int var;               // only used if Var (>=0)
};

struct Interner {
    bool canonicalize = true;
    std::vector<Node> nodes;                     // id -> node
    std::unordered_map<std::string,int> id_of;   // canonical key -> id

    struct LegacyStub {
        Kind kind;
        int var;
        LegacyStub* left = nullptr;
        LegacyStub* right = nullptr;
    };
    std::vector<std::unique_ptr<LegacyStub>> legacy_heap;  // owns heap nodes for legacy mode
    std::vector<LegacyStub*> legacy_ptrs;                   // id -> heap node (legacy only)

    Interner() = default;
    explicit Interner(bool canonical): canonicalize(canonical) {}

    static std::string var_label(int var){
        if (var < 0) var = 0;
        if (var < 26) return std::string(1, char('a' + var));
        return "x" + std::to_string(var);
    }

    static std::string key_for(Kind k, const std::vector<int>& kids, int var=-1){
        if (k==Kind::Var) return std::string("v:")+std::to_string(var);
        std::string s = (k==Kind::Join? "J:" : "M:");
        for (int id : kids) { s += std::to_string(id); s.push_back(','); }
        return s;
    }

    int make_var(int v){
        if (canonicalize){
            std::string key = key_for(Kind::Var, {}, v);
            auto it = id_of.find(key);
            if (it!=id_of.end()) return it->second;
            int id = (int)nodes.size();
            nodes.push_back(Node{Kind::Var, {}, v});
            id_of.emplace(key,id);
            legacy_ptrs.resize(nodes.size(), nullptr);
            return id;
        }
        // Legacy mode: preserve tree structure by allocating a fresh node even for
        // repeated variable symbols. This avoids unintended DAG sharing that would
        // otherwise act as an optimisation.
        int id = (int)nodes.size();
        nodes.push_back(Node{Kind::Var, {}, v});
        legacy_ptrs.resize(nodes.size(), nullptr);
        auto stub = std::make_unique<LegacyStub>();
        stub->kind = Kind::Var;
        stub->var = v;
        stub->left = nullptr;
        stub->right = nullptr;
        legacy_ptrs[id] = stub.get();
        legacy_heap.push_back(std::move(stub));
        return id;
    }

    int make_op(Kind k, std::vector<int> kids){
        if (!canonicalize){
            // Legacy mode: keep children exactly as they were parsed (order and multiplicity
            // matter for this representation), and expand to a binary tree chain to mimic
            // the prior pointer-based implementation.
            auto register_stub = [&](int id, const std::vector<int>& child_ids){
                legacy_ptrs.resize(nodes.size(), nullptr);
                auto stub = std::make_unique<LegacyStub>();
                stub->kind = k;
                stub->var = -1;
                stub->left = child_ids.size() > 0 ? legacy_ptrs[child_ids[0]] : nullptr;
                stub->right = child_ids.size() > 1 ? legacy_ptrs[child_ids[1]] : nullptr;
                legacy_ptrs[id] = stub.get();
                legacy_heap.push_back(std::move(stub));
            };

            auto append_node = [&](std::vector<int> child_ids){
                std::vector<int> for_stub = child_ids;
                int id = (int)nodes.size();
                nodes.push_back(Node{k, std::move(child_ids), -1});
                register_stub(id, for_stub);
                return id;
            };

            if (kids.empty()){
                int id = (int)nodes.size();
                nodes.push_back(Node{k, {}, -1});
                register_stub(id, {});
                return id;
            }

            if (kids.size() == 1){
                return append_node({kids[0]});
            }

            int acc = kids[0];
            for (size_t idx = 1; idx < kids.size(); ++idx){
                acc = append_node({acc, kids[idx]});
            }
            return acc;
        }

        // Canonicalize: flatten same-op, sort, unique (AC + idempotence)
        std::vector<int> flat; flat.reserve(kids.size());
        for (int id : kids){
            const Node& n = nodes[id];
            if (n.kind==k) flat.insert(flat.end(), n.kids.begin(), n.kids.end());
            else flat.push_back(id);
        }
        std::sort(flat.begin(), flat.end());
        flat.erase(std::unique(flat.begin(), flat.end()), flat.end());
        if (flat.size()==1) return flat[0];
        std::string key = key_for(k, flat);
        auto it = id_of.find(key);
        if (it!=id_of.end()) return it->second;
        int id = (int)nodes.size();
        nodes.push_back(Node{k, std::move(flat), -1});
        id_of.emplace(key,id);
        legacy_ptrs.resize(nodes.size(), nullptr);
        return id;
    }

    // Pretty printer (variadic S-expression)
    std::string sexpr(int id) const {
        const Node& n = nodes[id];
        if (n.kind==Kind::Var) return var_label(n.var);
        const char* op = (n.kind==Kind::Join? "+" : "*");
        std::string s = "("; s += op;
        for (int c : n.kids) { s += " "; s += sexpr(c); }
        s += ")"; return s;
    }
};

} // namespace lat

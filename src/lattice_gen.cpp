// Build: g++ -O2 -std=c++17 -Iinclude src/lattice_gen.cpp -o bin/lattice_gen
// Emits JSONL lines with just {"u": "...", "v": "..."} (plus minimal config).

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

#include "term.hpp"   // only for Kind enum (Join/Meet/Var)

using std::string;
using std::unique_ptr;
using lat::Kind;

// ------------ raw TREE (used only to shape output S-exprs) ------------
struct TNode {
    Kind kind;
    int var;                                   // only for Var
    std::vector<unique_ptr<TNode>> kids;       // empty for Var
};

static unique_ptr<TNode> make_var(int v){
    auto n = std::make_unique<TNode>(); n->kind=Kind::Var; n->var=v; return n;
}
static unique_ptr<TNode> make_op(Kind k, std::vector<unique_ptr<TNode>> ch){
    auto n = std::make_unique<TNode>(); n->kind=k; n->var=-1; n->kids=std::move(ch); return n;
}

static string label_var(int idx){
    if (idx < 0) idx = 0;
    return "x" + std::to_string(idx);
}

static string show(const TNode* n){
    if (n->kind==Kind::Var) {
        return label_var(n->var);
    }
    const char* op = (n->kind==Kind::Join? "+" : "*");
    string s = "("; s += op;
    for (auto const& c : n->kids){ s += " "; s += show(c.get()); }
    s += ")"; return s;
}

static std::unique_ptr<TNode> clone_tree(const TNode* node){
    if (!node) return nullptr;
    if (node->kind == Kind::Var){
        return make_var(node->var);
    }
    std::vector<std::unique_ptr<TNode>> kids;
    kids.reserve(node->kids.size());
    for (const auto& child : node->kids){
        kids.push_back(clone_tree(child.get()));
    }
    return make_op(node->kind, std::move(kids));
}

static string kind_to_string(Kind k){
    switch(k){
        case Kind::Join: return "join";
        case Kind::Meet: return "meet";
        default:         return "var";
    }
}

struct TreeMetrics {
    int nodes = 0;
    int edges = 0;
    int leaves = 0;
    int height = 0;
    int top_arity = 0;
    int unique_leaves = 0;
    double alt_index = 0.0;
    double share_ratio = 1.0;
    string root_kind = "var";
};

struct MetricAcc {
    int nodes = 0;
    int edges = 0;
    int leaves = 0;
    int max_depth = 0;
    int alternating_edges = 0;
    int operator_edges = 0;
    std::unordered_set<int> leaf_vars;
    std::unordered_map<std::string,int> subtree_freq;
};

static std::string canonical_collect(const TNode* node, MetricAcc& acc, int depth){
    if (!node) return "";
    if (depth > acc.max_depth) acc.max_depth = depth;
    acc.nodes++;

    if (node->kind == Kind::Var){
        acc.leaves++;
        acc.leaf_vars.insert(node->var);
        std::string key = "v:" + std::to_string(node->var);
        acc.subtree_freq[key]++;
        return key;
    }

    std::vector<std::string> child_keys;
    child_keys.reserve(node->kids.size());
    for (const auto& child : node->kids){
        if (!child) continue;
        acc.edges++;
        if (child->kind != Kind::Var){
            acc.operator_edges++;
            if (child->kind != node->kind) acc.alternating_edges++;
        }
        child_keys.push_back(canonical_collect(child.get(), acc, depth+1));
    }
    std::sort(child_keys.begin(), child_keys.end());
    std::string key = (node->kind==Kind::Join?"J:":"M:");
    for (const auto& ck : child_keys){
        key += ck;
        key.push_back('|');
    }
    acc.subtree_freq[key]++;
    return key;
}

static TreeMetrics analyse_tree(const TNode* root){
    TreeMetrics M;
    if (!root) return M;
    MetricAcc acc;
    canonical_collect(root, acc, 0);

    double total_subtrees = 0.0;
    for (const auto& kv : acc.subtree_freq) total_subtrees += kv.second;
    if (total_subtrees <= 0.0) total_subtrees = 1.0;

    M.nodes = acc.nodes;
    M.edges = acc.edges;
    M.leaves = acc.leaves;
    M.height = acc.max_depth;
    M.top_arity = (root->kind == Kind::Var) ? 0 : static_cast<int>(root->kids.size());
    M.unique_leaves = static_cast<int>(acc.leaf_vars.size());
    if (acc.operator_edges > 0) {
        M.alt_index = static_cast<double>(acc.alternating_edges) / static_cast<double>(acc.operator_edges);
    }
    M.share_ratio = static_cast<double>(acc.subtree_freq.size()) / total_subtrees;
    M.root_kind = kind_to_string(root->kind);
    return M;
}

static string metrics_json(const TreeMetrics& m){
    std::ostringstream oss;
    oss << "{\"nodes\":" << m.nodes
        << ",\"edges\":" << m.edges
        << ",\"leaves\":" << m.leaves
        << ",\"height\":" << m.height
        << ",\"top_arity\":" << m.top_arity
        << ",\"unique_leaves\":" << m.unique_leaves
        << ",\"alt_index\":";
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6) << m.alt_index;
    oss.unsetf(std::ios::floatfield);
    oss << ",\"share_ratio\":";
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6) << m.share_ratio;
    oss.unsetf(std::ios::floatfield);
    oss << ",\"root\":\"" << m.root_kind << "\"}";
    return oss.str();
}

// ------------------------------- RNG ----------------------------------
struct RNG {
    std::mt19937_64 g;
    explicit RNG(uint64_t seed): g(seed) {}
    double U(){ return std::uniform_real_distribution<>(0,1)(g); }
    int Ui(int a,int b){ return std::uniform_int_distribution<>(a,b)(g); }
};

// ------------------------------- Gen ----------------------------------
struct GenCfg {
    int num_vars=6;
    int min_arity=2, max_arity=4;
    double p_join=0.5;
    double p_alternate=0.8;
};

struct TreeGen {
    RNG rng; GenCfg cfg;
    explicit TreeGen(uint64_t seed, GenCfg c): rng(seed), cfg(c) {}

    unique_ptr<TNode> leaf(){ return make_var(rng.Ui(0, cfg.num_vars-1)); }
    Kind choose(Kind parent){
        if (rng.U() < cfg.p_alternate){
            if (parent==Kind::Join) return Kind::Meet;
            if (parent==Kind::Meet) return Kind::Join;
        }
        return (rng.U() < cfg.p_join ? Kind::Join : Kind::Meet);
    }
    int arity(int budget){
        int lo = std::max(2, cfg.min_arity);
        int hi = std::min(cfg.max_arity, budget+1);
        if (lo > hi) lo = hi; // safety when budget is tiny
        return rng.Ui(lo, hi);
    }

    // Balanced-ish: exact “budget” internal nodes
    unique_ptr<TNode> budgeted(int B, Kind parent){
        if (B<=0) return leaf();
        Kind k = choose(parent);
        int ar = arity(B);
        std::vector<int> child_budget(ar,0); int rem=B-1;
        for (int i=0;i<ar-1;i++){ int t = rem? rng.Ui(0,rem):0; child_budget[i]=t; rem-=t; }
        child_budget.back() += rem;
        std::vector<unique_ptr<TNode>> kids; kids.reserve(ar);
        for (int i=0;i<ar;i++) kids.push_back(budgeted(child_budget[i], k));
        return make_op(k, std::move(kids));
    }

    // Left/right spines (degenerate)
    unique_ptr<TNode> spine(int B, bool left){
        if (B<=0) return leaf();
        Kind k = choose(Kind::Var);
        std::vector<unique_ptr<TNode>> kids;
        if (left){ kids.push_back(spine(B-1,left)); kids.push_back(leaf()); }
        else     { kids.push_back(leaf()); kids.push_back(spine(B-1,left)); }
        return make_op(k, std::move(kids));
    }

    // Fully alternating (probabilistic root via parent==Var seed)
    unique_ptr<TNode> alternating(int B){
        std::function<unique_ptr<TNode>(int,Kind)> rec = [&](int b, Kind parent)->unique_ptr<TNode>{
            if (b<=0) return leaf();
            Kind k = (parent==Kind::Join? Kind::Meet : Kind::Join);
            int ar = arity(b);
            std::vector<int> child_budget(ar,0); int rem=b-1;
            for (int i=0;i<ar-1;i++){ int t = rem? rng.Ui(0,rem):0; child_budget[i]=t; rem-=t; }
            child_budget.back() += rem;
            std::vector<unique_ptr<TNode>> kids; kids.reserve(ar);
            for (int i=0;i<ar;i++) kids.push_back(rec(child_budget[i], k));
            return make_op(k, std::move(kids));
        };
        return rec(B, Kind::Var);
    }

    // Fully alternating with a *fixed* root operator (deterministic alternation)
    unique_ptr<TNode> alternating_rooted(int B, Kind root){
        std::function<unique_ptr<TNode>(int,Kind)> rec = [&](int b, Kind parent)->unique_ptr<TNode>{
            if (b<=0) return leaf();
            // Alternate deterministically from parent
            Kind k = (parent==Kind::Join? Kind::Meet : Kind::Join);
            int ar = arity(b);
            std::vector<int> child_budget(ar,0); int rem=b-1;
            for (int i=0;i<ar-1;i++){ int t = rem? rng.Ui(0,rem):0; child_budget[i]=t; rem-=t; }
            child_budget.back() += rem;
            std::vector<unique_ptr<TNode>> kids; kids.reserve(ar);
            for (int i=0;i<ar;i++) kids.push_back(rec(child_budget[i], k));
            return make_op(k, std::move(kids));
        };
        // Seed with the opposite so the first chosen is exactly `root`
        Kind fake_parent = (root==Kind::Join? Kind::Meet : Kind::Join);
        return rec(B, fake_parent);
    }

    // Full binary, perfectly alternating by *height* (height H: leaves at depth H)
    unique_ptr<TNode> alt_full_height(int H, Kind root){
        if (H<=0) return leaf();
        Kind next = (root==Kind::Join? Kind::Meet : Kind::Join);
        std::vector<unique_ptr<TNode>> kids;
        kids.push_back(alt_full_height(H-1, next));
        kids.push_back(alt_full_height(H-1, next));
        return make_op(root, std::move(kids));
    }

    // Full binary with random operator labels, mirroring the legacy pointer-based generator.
    unique_ptr<TNode> random_full_binary(int H){
        if (H<=0) return leaf();
        Kind root = (rng.U() < cfg.p_join ? Kind::Join : Kind::Meet);
        std::vector<std::unique_ptr<TNode>> kids;
        kids.push_back(random_full_binary(H-1));
        kids.push_back(random_full_binary(H-1));
        return make_op(root, std::move(kids));
    }

    unique_ptr<TNode> dnf(int clauses, int k_each){
        std::vector<unique_ptr<TNode>> cls; cls.reserve(clauses);
        for (int i=0;i<clauses;i++){
            std::vector<unique_ptr<TNode>> lits; lits.reserve(k_each);
            for (int j=0;j<k_each;j++) lits.push_back(leaf());
            cls.push_back(make_op(Kind::Meet, std::move(lits)));
        }
        return make_op(Kind::Join, std::move(cls));
    }

    unique_ptr<TNode> cnf(int clauses, int k_each){
        std::vector<unique_ptr<TNode>> disj; disj.reserve(clauses);
        for (int i=0;i<clauses;i++){
            std::vector<unique_ptr<TNode>> lits; lits.reserve(k_each);
            for (int j=0;j<k_each;j++) lits.push_back(leaf());
            disj.push_back(make_op(Kind::Join, std::move(lits)));
        }
        return make_op(Kind::Meet, std::move(disj));
    }
};

// ------------------------------- CLI ----------------------------------
struct CLI {
    uint64_t seed=12345; int vars=6; int budget=100; int samples=10;
    std::string shape="balanced"; std::string out="";
    double p_join=0.5; double p_alt=0.8; int min_arity=2; int max_arity=4;
    // NEW: control root kinds for alternating shape
    std::string u_root = "auto";  // "auto" | "meet" | "join"
    std::string v_root = "auto";  // "auto" | "meet" | "join"
};

static void usage(){
    std::cerr
      << "Usage: rand --seed S --vars V --budget B --samples N "
    << "--shape {balanced|leftspine|rightspine|alternating|altfull|dnf|cnf|bestcase|worstcase} "
      << "[--p_join X] [--p_alt X] [--min_arity A] [--max_arity A] "
      << "[--u_root {auto|meet|join}] [--v_root {auto|meet|join}] "
      << "--out file.jsonl\n";
}

static CLI parse_cli(int argc, char** argv){
    if (argc<2){ usage(); std::exit(1); }
    if (std::string(argv[1])!="rand"){ usage(); std::exit(1); }
    CLI c;
    for (int i=2;i<argc;i++){
        std::string a = argv[i];
        auto need=[&](int k){ if(i+k>=argc){ std::cerr<<"Missing value for "<<a<<"\n"; std::exit(1);} };
        if(a=="--seed"){ need(1); c.seed=std::stoull(argv[++i]); }
        else if(a=="--vars"){ need(1); c.vars=std::stoi(argv[++i]); }
        else if(a=="--budget"){ need(1); c.budget=std::stoi(argv[++i]); }
        else if(a=="--samples"){ need(1); c.samples=std::stoi(argv[++i]); }
        else if(a=="--shape"){ need(1); c.shape=argv[++i]; }
        else if(a=="--out"){ need(1); c.out=argv[++i]; }
        else if(a=="--p_join"){ need(1); c.p_join=std::stod(argv[++i]); }
        else if(a=="--p_alt"){ need(1); c.p_alt=std::stod(argv[++i]); }
        else if(a=="--min_arity"){ need(1); c.min_arity=std::stoi(argv[++i]); }
        else if(a=="--max_arity"){ need(1); c.max_arity=std::stoi(argv[++i]); }
        else if(a=="--u_root"){ need(1); c.u_root=argv[++i]; }
        else if(a=="--v_root"){ need(1); c.v_root=argv[++i]; }
        else { usage(); std::exit(1); }
    }
    return c;
}

static std::string jescape(const std::string& s){
    std::string out; out.reserve(s.size()+8);
    for(char c: s){
        switch(c){ case '\\': out+="\\\\"; break; case '"': out+="\\\""; break;
        case '\n': out+="\\n"; break; case '\r': out+="\\r"; break; case '\t': out+="\\t"; break;
        default: out+=c; break; }
    }
    return out;
}

static inline bool is_meet(const std::string& s){ return s=="meet"; }
static inline bool is_join(const std::string& s){ return s=="join"; }
static Kind str_to_kind_or(const std::string& s, Kind fallback){
    if (is_meet(s)) return Kind::Meet;
    if (is_join(s)) return Kind::Join;
    return fallback; // "auto" or anything else → fallback
}

int main(int argc, char** argv){
    std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
    CLI cli = parse_cli(argc, argv);

    // --- in main(), after parse_cli(...) and before constructing TreeGen:
    GenCfg cfg;
    cfg.num_vars = std::max(1, cli.vars);
    cfg.min_arity  = cli.min_arity;
    cfg.max_arity  = cli.max_arity;
    cfg.p_join     = cli.p_join;
    cfg.p_alternate= cli.p_alt;
    TreeGen G(cli.seed, cfg);


    std::ofstream out(cli.out, std::ios::out);
    if (!cli.out.empty() && !out){ std::cerr<<"Cannot open output file\n"; return 1; }

    for (int s=0; s<cli.samples; ++s){
        unique_ptr<TNode> U, V;

        if      (cli.shape=="balanced")   { U=G.budgeted(cli.budget, Kind::Var); V=G.budgeted(cli.budget, Kind::Var); }
        else if (cli.shape=="leftspine")  { U=G.spine(std::max(1,cli.budget), true);  V=G.spine(std::max(1,cli.budget), true); }
        else if (cli.shape=="rightspine") { U=G.spine(std::max(1,cli.budget), false); V=G.spine(std::max(1,cli.budget), false); }
        else if (cli.shape=="alternating"){
            bool rooted = (cli.u_root!="auto" || cli.v_root!="auto");
            if (!rooted){
                U=G.alternating(cli.budget);
                V=G.alternating(cli.budget);
            } else {
                // Defaults: Meet for U and Join for V if "auto" is left on either side.
                Kind ku = str_to_kind_or(cli.u_root, Kind::Meet);
                Kind kv = str_to_kind_or(cli.v_root, Kind::Join);
                U=G.alternating_rooted(cli.budget, ku);
                V=G.alternating_rooted(cli.budget, kv);
            }
        }
        else if (cli.shape=="altfull"){ // budget interpreted as *height*
            U = G.alt_full_height(cli.budget, Kind::Meet);  // u root = Meet
            V = G.alt_full_height(cli.budget, Kind::Join);  // v root = Join
        }
        else if (cli.shape=="legacy_full"){ // emulate legacy random full binary trees
            int height = std::max(0, cli.budget);
            U = G.random_full_binary(height);
            V = G.random_full_binary(height);
        }
        else if (cli.shape=="dnf")        { U=G.dnf(std::max(2, cli.budget/4), 3); V=G.dnf(std::max(2, cli.budget/4), 3); }
        else if (cli.shape=="cnf")        { U=G.cnf(std::max(2, cli.budget/4), 3); V=G.cnf(std::max(2, cli.budget/4), 3); }
        else if (cli.shape=="bestcase")   {
            int depth = std::max(0, cli.budget);
            auto shallow = G.alt_full_height(0, Kind::Meet);   // single generator
            auto deep_meet = G.alt_full_height(depth, Kind::Meet);
            auto deep_join = G.alt_full_height(depth, Kind::Join);
            // Alternate orientations to capture both u≤v and v≤u short-circuits.
            if (s % 2 == 0) {
                U = clone_tree(shallow.get());
                V = std::move(deep_join);
            } else {
                U = std::move(deep_meet);
                V = clone_tree(shallow.get());
            }
        }
        else if (cli.shape=="worstcase")  {
            int depth = std::max(0, cli.budget);
            auto u_core = G.alt_full_height(depth, Kind::Meet);
            auto v_core = G.alt_full_height(depth, Kind::Join);
            if (depth <= 0) {
                U = std::move(u_core);
                V = std::move(v_core);
            } else {
                // Attach additional alternating branches to force expansive recursion.
                std::vector<std::unique_ptr<TNode>> uKids;
                std::vector<std::unique_ptr<TNode>> vKids;
                uKids.push_back(std::move(u_core));
                uKids.push_back(G.alt_full_height(depth-1, Kind::Join));
                uKids.push_back(G.alt_full_height(depth-1, Kind::Meet));
                vKids.push_back(std::move(v_core));
                vKids.push_back(G.alt_full_height(depth-1, Kind::Meet));
                vKids.push_back(G.alt_full_height(depth-1, Kind::Join));
                U = make_op(Kind::Meet, std::move(uKids));
                V = make_op(Kind::Join, std::move(vKids));
            }
        }
        else                              { U=G.budgeted(cli.budget, Kind::Var); V=G.spine(std::max(1,cli.budget), true); }

        const std::string su = show(U.get());
        const std::string sv = show(V.get());

        if (out.is_open()){
            TreeMetrics mu = analyse_tree(U.get());
            TreeMetrics mv = analyse_tree(V.get());
            double avg_alt = 0.5 * (mu.alt_index + mv.alt_index);
            double avg_share = 0.5 * (mu.share_ratio + mv.share_ratio);
            out << "{\"u\":\"" << jescape(su) << "\",\"v\":\"" << jescape(sv) << "\",";
            out << "\"meta\":{\"u\":" << metrics_json(mu)
                << ",\"v\":" << metrics_json(mv)
                << ",\"pair\":{\"total_nodes\":" << (mu.nodes + mv.nodes)
                << ",\"max_height\":" << std::max(mu.height, mv.height)
                << ",\"avg_alt_index\":";
            out.setf(std::ios::fixed);
            out << std::setprecision(6) << avg_alt;
            out.unsetf(std::ios::floatfield);
            out << ",\"avg_share_ratio\":";
            out.setf(std::ios::fixed);
            out << std::setprecision(6) << avg_share;
            out.unsetf(std::ios::floatfield);
            out << "}}";
            out << ",\"config\":{\"seed\":" << cli.seed
                << ",\"vars\":" << cli.vars
                << ",\"budget\":" << cli.budget
                << ",\"shape\":\"" << cli.shape << "\""
                << ",\"sample\":" << s
                << ",\"p_join\":" << cli.p_join
                << ",\"p_alt\":" << cli.p_alt
                << ",\"min_arity\":" << cli.min_arity
                << ",\"max_arity\":" << cli.max_arity;
            if (cli.shape=="alternating" && (cli.u_root!="auto" || cli.v_root!="auto")){
                out << ",\"u_root\":\"" << cli.u_root << "\",\"v_root\":\"" << cli.v_root << "\"";
            }
            out << "}}\n";
        } else {
            std::cout << su << "\n" << sv << "\n";
        }
    }
    return 0;
}

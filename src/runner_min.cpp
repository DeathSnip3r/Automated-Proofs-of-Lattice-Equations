// Build: g++ -O2 -std=c++17 -Iinclude src/whitman.cpp src/freese.cpp src/cosmadakis.cpp src/hunt.cpp src/runner_min.cpp -o bin/check
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#elif defined(__linux__)
#  include <sys/resource.h>
#  include <sys/time.h>
#endif

#include "term.hpp"
#include "sexpr.hpp"
#include "whitman.hpp"
#include "freese.hpp"
#include "cosmadakis.hpp"
#include "hunt.hpp"

using namespace std;
using namespace lat;

static optional<string> grab_json_string(const string& s, const string& key){
    const string needle = "\"" + key + "\"";
    size_t p = s.find(needle); if (p == string::npos) return nullopt;
    p = s.find(':', p);        if (p == string::npos) return nullopt;
    p = s.find('"', p);        if (p == string::npos) return nullopt;
    string out; bool esc = false;
    for (size_t i = p + 1; i < s.size(); ++i){
        char c = s[i];
        if (esc){ out.push_back(c); esc = false; }
        else if (c == '\\') esc = true;
        else if (c == '"')   break;
        else out.push_back(c);
    }
    return out;
}

enum class Engine { Whitman, Freese, Cosma, Hunt };

enum class Representation { Canonical, Legacy };

static string engine_name(Engine E){
    switch(E){
        case Engine::Whitman: return "whitman";
        case Engine::Freese:  return "freese";
        case Engine::Cosma:   return "cosmadakis";
        case Engine::Hunt:    return "hunt";
    }
    return "unknown";
}

static Engine parse_engine(const string& e){
    if (e=="whitman") return Engine::Whitman;
    if (e=="freese")  return Engine::Freese;
    if (e=="cosma" || e=="cosmadakis") return Engine::Cosma;
    if (e=="hunt")    return Engine::Hunt;
    cerr << "unknown --engine: " << e << "\n"; std::exit(1);
}

static Representation parse_representation(const string& r){
    if (r.empty() || r=="canonical") return Representation::Canonical;
    if (r=="legacy") return Representation::Legacy;
    cerr << "unknown --repr: " << r << "\n"; std::exit(1);
}

static string csv_escape(const string& s){
    string out = "\"";
    for (char c : s){
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

struct DirectionResult {
    bool ok = false;
    bool value = false;
    uint64_t micros = 0;
    string stats;
    string error;
    uint64_t working_set = 0;
    uint64_t peak_working_set = 0;
    uint64_t private_bytes = 0;
};

struct MemorySnapshot {
    uint64_t working_set = 0;
    uint64_t peak_working_set = 0;
    uint64_t private_bytes = 0;
};

static MemorySnapshot sample_memory(){
    MemorySnapshot snap;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))){
        snap.working_set = static_cast<uint64_t>(pmc.WorkingSetSize);
        snap.peak_working_set = static_cast<uint64_t>(pmc.PeakWorkingSetSize);
        snap.private_bytes = static_cast<uint64_t>(pmc.PrivateUsage);
    }
#elif defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0){
        snap.working_set = static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
        snap.peak_working_set = snap.working_set;
    }
#endif
    return snap;
}

static string stats_json(const Whitman& W){
    ostringstream oss;
    oss << "{\"pairs\":" << W.pairs_visited
        << ",\"max_stack\":" << W.max_stack
        << ",\"branch_attempts\":" << W.branch_attempts
        << ",\"branch_right\":" << W.branch_right_attempts
        << ",\"branch_left\":" << W.branch_left_attempts
        << ",\"branch_successes\":" << W.branch_successes
        << ",\"join_decomp\":" << W.join_decompositions
        << ",\"meet_decomp\":" << W.meet_decompositions
        << "}";
    return oss.str();
}

static string stats_json(const FreeseStats& S){
    ostringstream oss;
    oss << "{\"recursive_calls\":" << S.recursive_calls
        << ",\"memo_hits\":" << S.memo_hits
        << ",\"branch_attempts\":" << S.branch_attempts
        << ",\"branch_successes\":" << S.branch_successes
        << ",\"max_stack\":" << S.max_stack
        << "}";
    return oss.str();
}

static string stats_json(const CosmaStats& S){
    ostringstream oss;
    oss << "{\"N\":" << S.N
        << ",\"arcs_enqueued\":" << S.arcs_enqueued
        << ",\"arcs_processed\":" << S.arcs_processed
        << ",\"max_queue\":" << S.max_queue
        << "}";
    return oss.str();
}

static string stats_json(const HuntStats& S){
    ostringstream oss;
    oss << "{\"RU\":" << S.RU
        << ",\"RV\":" << S.RV
        << ",\"cells\":" << S.cells_evaluated
        << ",\"and_checks\":" << S.and_checks
        << ",\"or_checks\":" << S.or_checks
        << ",\"max_hU\":" << S.max_hU
        << ",\"max_hV\":" << S.max_hV
        << "}";
    return oss.str();
}

static DirectionResult evaluate_direction(Engine E, Interner& I, int lhs, int rhs, bool want_stats){
    DirectionResult R;
    try{
        switch(E){
            case Engine::Whitman: {
                Whitman W(&I);
                auto t0 = chrono::high_resolution_clock::now();
                bool ans = W.leq(lhs, rhs);
                auto t1 = chrono::high_resolution_clock::now();
                R.ok = true;
                R.value = ans;
                R.micros = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
                if (want_stats) R.stats = stats_json(W);
                MemorySnapshot snap = sample_memory();
                R.working_set = snap.working_set;
                R.peak_working_set = snap.peak_working_set;
                R.private_bytes = snap.private_bytes;
                break;
            }
            case Engine::Freese: {
                Freese F(&I);
                auto t0 = chrono::high_resolution_clock::now();
                bool ans = F.leq(lhs, rhs);
                auto t1 = chrono::high_resolution_clock::now();
                R.ok = true;
                R.value = ans;
                R.micros = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
                if (want_stats) R.stats = stats_json(F.stats);
                MemorySnapshot snap = sample_memory();
                R.working_set = snap.working_set;
                R.peak_working_set = snap.peak_working_set;
                R.private_bytes = snap.private_bytes;
                break;
            }
            case Engine::Cosma: {
                CosmaStats S;
                auto t0 = chrono::high_resolution_clock::now();
                bool ans = leq_cosma(lhs, rhs, I, want_stats ? &S : nullptr);
                auto t1 = chrono::high_resolution_clock::now();
                R.ok = true;
                R.value = ans;
                R.micros = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
                if (want_stats) R.stats = stats_json(S);
                MemorySnapshot snap = sample_memory();
                R.working_set = snap.working_set;
                R.peak_working_set = snap.peak_working_set;
                R.private_bytes = snap.private_bytes;
                break;
            }
            case Engine::Hunt: {
                HuntStats S;
                auto t0 = chrono::high_resolution_clock::now();
                bool ans = hunt_leq(lhs, rhs, I, want_stats ? &S : nullptr);
                auto t1 = chrono::high_resolution_clock::now();
                R.ok = true;
                R.value = ans;
                R.micros = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
                if (want_stats) R.stats = stats_json(S);
                MemorySnapshot snap = sample_memory();
                R.working_set = snap.working_set;
                R.peak_working_set = snap.peak_working_set;
                R.private_bytes = snap.private_bytes;
                break;
            }
        }
    } catch (const std::exception& ex){
        R.error = ex.what();
        R.ok = false;
        R.value = false;
        R.micros = 0;
        R.stats.clear();
    }
    return R;
}

static void run_one(Engine E, Representation Rmode, const string& su, const string& sv, int idx, bool want_stats){
    try{
        bool canonical = (Rmode == Representation::Canonical);
        Interner I(canonical);
        SExprParser P(&I);
        int u = P.parse(su), v = P.parse(sv);

        DirectionResult uv = evaluate_direction(E, I, u, v, want_stats);
        DirectionResult vu = evaluate_direction(E, I, v, u, want_stats);

        uint64_t total_us = uv.micros + vu.micros;
        bool u_leq_v = uv.ok && uv.value;
        bool v_leq_u = vu.ok && vu.value;
        bool equal = u_leq_v && v_leq_u;

    const char* repr_str = canonical ? "canonical" : "legacy";
    cout << engine_name(E) << ","
             << idx << ","
             << (u_leq_v ? 1 : 0) << ","
             << (v_leq_u ? 1 : 0) << ","
             << (equal ? 1 : 0) << ","
             << uv.micros << ","
             << vu.micros << ","
             << total_us << ","
             << (want_stats ? csv_escape(uv.stats) : "\"\"") << ","
           << (want_stats ? csv_escape(vu.stats) : "\"\"") << ","
           << uv.working_set << ","
           << uv.peak_working_set << ","
           << uv.private_bytes << ","
           << vu.working_set << ","
           << vu.peak_working_set << ","
           << vu.private_bytes << ","
                     << csv_escape(uv.error) << ","
                         << csv_escape(vu.error) << ","
                         << repr_str << "\n";
    } catch (const std::exception& e){
                   const char* repr_str = (Rmode == Representation::Canonical) ? "canonical" : "legacy";
                   cout << engine_name(E) << ","
                       << idx << ",0,0,0,0,0,0,\"\",\"\",0,0,0,0,0,0," << csv_escape(e.what()) << ",\"\"," << repr_str << "\n";
    }
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);

    if (argc < 3){
       cerr << "Usage:\n"
           << "  check --engine {whitman,freese,cosma,hunt} [--stats] [--repr canonical|legacy] --pair \"u\" \"v\"\n"
           << "  check --engine {whitman,freese,cosma,hunt} [--stats] [--repr canonical|legacy] --json file.jsonl [--limit N]\n";
        return 1;
    }
    string engs, json_path, u_cli, v_cli, repr_cli; size_t limit = numeric_limits<size_t>::max();
    bool want_stats = false;
    for (int i=1;i<argc;i++){
        string a = argv[i];
        if (a=="--engine" && i+1<argc) engs = argv[++i];
        else if (a=="--pair"  && i+2<argc){ u_cli = argv[++i]; v_cli = argv[++i]; }
        else if (a=="--json"  && i+1<argc) json_path = argv[++i];
        else if (a=="--limit" && i+1<argc) limit = stoull(argv[++i]);
        else if (a=="--repr" && i+1<argc) repr_cli = argv[++i];
        else if (a=="--stats") want_stats = true;
    }
    Engine E = parse_engine(engs);
    Representation Rmode = parse_representation(repr_cli);

    cout << "engine,pair_id,u_leq_v,v_leq_u,eq,uv_us,vu_us,total_us,uv_stats,vu_stats,uv_working_set,uv_peak_working_set,uv_private_bytes,vu_working_set,vu_peak_working_set,vu_private_bytes,uv_error,vu_error,representation\n";

    if (!u_cli.empty()){ run_one(E,Rmode,u_cli,v_cli,0,want_stats); return 0; }

    if (!json_path.empty()){
        ifstream in(json_path);
        if (!in){ cerr<<"cannot open "<<json_path<<"\n"; return 1; }
        string line; size_t idx=0; bool any=false;
        while (idx<limit && getline(in,line)){
            auto u = grab_json_string(line,"u");
            auto v = grab_json_string(line,"v");
            if (u && v){ run_one(E,Rmode,*u,*v,static_cast<int>(idx++),want_stats); any=true; }
        }
        if (!any){
            in.clear(); in.seekg(0);
            string blob((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
            auto u = grab_json_string(blob,"u");
            auto v = grab_json_string(blob,"v");
            if (u && v) run_one(E,Rmode,*u,*v,0,want_stats);
            else { cerr<<"No {u,v} found in "<<json_path<<"\n"; return 1; }
        }
        return 0;
    }

    cerr << "Provide --pair or --json.\n";
    return 1;
}

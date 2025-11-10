#pragma once
#include <string>
#include <stdexcept>
#include <cctype>
#include <vector>
#include <unordered_map>
#include "term.hpp"

namespace lat {

// Grammar: term := VAR | '(' op term+ ')',   op := '+' | '*'
// Variables may be single letters (legacy) or multi-character symbols (e.g. x37).
struct SExprParser {
    const std::string* s = nullptr;
    size_t i = 0;
    Interner* I = nullptr;

    std::unordered_map<std::string,int> var_ids;   // symbol -> variable index
    std::unordered_map<int,std::string> id_names;  // variable index -> canonical symbol
    int next_tmp_id = 0;

    explicit SExprParser(Interner* interner): I(interner) {}

    static bool sp(char c){ return c==' '||c=='\n'||c=='\t'||c=='\r'; }
    void skip(){ while (i<s->size() && sp((*s)[i])) ++i; }
    [[noreturn]] void fail(const char* msg){ throw std::runtime_error(std::string("parse error: ")+msg+" at pos "+std::to_string(i)); }

    int parse(const std::string& str){ s=&str; i=0; return term(); }

    static bool sym_start(char c){ return std::isalpha(static_cast<unsigned char>(c)) || c=='_'; }
    static bool sym_char(char c){
        return std::isalnum(static_cast<unsigned char>(c)) || c=='_';
    }

    std::string symbol(){
        size_t start = i;
        if (start>=s->size() || !sym_start((*s)[start])) fail("expected symbol");
        ++i;
        while (i<s->size() && sym_char((*s)[i])) ++i;
        return s->substr(start, i-start);
    }

    static bool all_digits(const std::string& str, size_t start){
        if (start >= str.size()) return false;
        for (size_t k=start; k<str.size(); ++k){
            if (!std::isdigit(static_cast<unsigned char>(str[k]))) return false;
        }
        return true;
    }

    int deduce_index(const std::string& name){
        if (name.size()==1 && std::isalpha(static_cast<unsigned char>(name[0]))){
            int v = std::tolower(static_cast<unsigned char>(name[0])) - 'a';
            return std::max(0, v);
        }
        if ((name[0]=='x' || name[0]=='X') && all_digits(name,1)){
            return std::stoi(name.substr(1));
        }
        if (std::isdigit(static_cast<unsigned char>(name[0])) && all_digits(name,0)){
            return std::stoi(name);
        }
        return -1;
    }

    int intern_variable(const std::string& name){
        auto it = var_ids.find(name);
        if (it!=var_ids.end()) return I->make_var(it->second);

        int id = deduce_index(name);
        auto existing = id_names.find(id);
        if (id < 0 || (existing!=id_names.end() && existing->second != name)){
            id = next_tmp_id;
            while (id_names.find(id)!=id_names.end()) ++id;
        }

        var_ids.emplace(name, id);
        id_names.emplace(id, name);
        if (id >= next_tmp_id) next_tmp_id = id + 1;
        return I->make_var(id);
    }

    int term(){
        skip(); if (i>=s->size()) fail("unexpected end");
        char c = (*s)[i];
        if (sym_start(c)) {
            std::string name = symbol();
            return intern_variable(name);
        }
        if (c=='('){
            ++i; skip(); if (i>=s->size()) fail("expected operator");
            char op = (*s)[i++]; Kind k;
            if (op=='+') k=Kind::Join; else if (op=='*') k=Kind::Meet; else fail("op must be + or *");
            std::vector<int> kids;
            while (true){
                skip(); if (i>=s->size()) fail("unclosed '('");
                if ((*s)[i]==')'){ ++i; break; }
                kids.push_back(term());
            }
            if (kids.empty()) fail("operator needs at least one child");
            return I->make_op(k, kids);
        }
        fail("unexpected token");
        return -1;
    }
};

} // namespace lat

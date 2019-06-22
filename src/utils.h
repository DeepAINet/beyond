//
// Created by mengqy on 2019/6/13.
//

#ifndef BEYOND_UTILS_H
#define BEYOND_UTILS_H

#include <fstream>
#include <vector>
#include "shape.h"
using namespace std;

#define EOS "</s>"

pair<int, int> only_one_inverted_order(const vector<int>& seq){
    if (seq.empty()) return {-1, -1};
    vector<pair<int, int>> inverted_orders(2);
    int idx = 0;
    for (int i = 0; i < seq.size(); ++i){
        if (seq[i] >= seq.size() || seq[i] < 0)
            return {-1, -1};

        if (i != seq[i]){
            if (idx == 2) return {-1, -1};
            inverted_orders[idx++] = {i, seq[i]};
        }
    }
    if (idx == 2 && (inverted_orders[0].first == inverted_orders[1].second) && (inverted_orders[1].first == inverted_orders[0].second))
        return {inverted_orders[0].second, inverted_orders[1].second};
    return {-1, -1};
}

bool same_shape_backward(const shape& a, const shape& b){
    assert(!a.empty() && !b.empty());
    auto aiter = a.dims.end()-1, biter = b.dims.end()-1;
    while((aiter != a.dims.begin()-1) && (biter != b.dims.begin()-1)){
        if (*aiter != *biter) return false;
        --aiter;
        --biter;
    }
    return true;
}

bool read_word(ifstream& in, string& word){
    int c;
    std::streambuf& sb = *in.rdbuf();
    word.clear();

    while((c = sb.sbumpc()) != EOF){
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0'){
            if (word.empty()){
                if (c == '\n'){
                    word += EOS;
                    return true;
                }
            }else{
                if (c == '\n') sb.sungetc();
                return true;
            }
        }
        word.push_back(c);
    }
    in.get();

    return !word.empty();
}


#endif //BEYOND_UTILS_H

//
// Created by mengqy on 2019/6/13.
//

#ifndef BEYOND_UTILS_H
#define BEYOND_UTILS_H

#include <vector>
using namespace std;

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

#endif //BEYOND_UTILS_H

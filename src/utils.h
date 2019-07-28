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

pair<int, int> only_one_inverted_order(const vector<int> &seq) {
    assert(!seq.empty());
    vector<pair<int, int>> inverted_orders;
    try {
        for (int i = 0; i < seq.size(); ++i) {
            if (i != seq[i]) {
                inverted_orders.push_back({i, seq[i]});
            }
            if (seq[i] < 0) throw std::invalid_argument("seq[" + std::to_string(i) + "] <= 0");
            if (seq[i] >= seq.size()) throw std::invalid_argument("seq[" + std::to_string(i) + "] >= seq.size()");
        }
        string error;
        if (inverted_orders.size() != 2){
            error = "There has not been only one inverted orders in the parameter seq!";
            throw std::invalid_argument(error);
        }
        if (inverted_orders[0].second != inverted_orders[1].first){
            error = "seq[" + std::to_string(inverted_orders[0].first) + "] should be equal to " + std::to_string(inverted_orders[1].first) + "!";
            throw std::invalid_argument(error);
        }
        if (inverted_orders[1].second != inverted_orders[0].first){
            if (!error.empty()) error += ",";
            error += "seq[" + std::to_string(inverted_orders[1].first) + "] should be equal to " + std::to_string(inverted_orders[0].first) + "!";
            throw std::invalid_argument(error);
        }
    } catch (std::invalid_argument& e){
        std::cerr << e.what() << std::endl;
        return {-1, -1};
    }

    return {inverted_orders[0].second, inverted_orders[1].second};
}

bool same_shape_backward(const shape &a, const shape &b) {
    assert(!a.empty() && !b.empty());
    auto aiter = a.dims.end() - 1, biter = b.dims.end() - 1;
    while ((aiter != a.dims.begin() - 1) && (biter != b.dims.begin() - 1)) {
        if (*aiter != *biter) return false;
        --aiter;
        --biter;
    }
    return true;
}

bool read_word(ifstream &in, string &word) {
    int c;
    std::streambuf &sb = *in.rdbuf();
    word.clear();

    while ((c = sb.sbumpc()) != EOF) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
            if (word.empty()) {
                if (c == '\n') {
                    word += EOS;
                    return true;
                }
            } else {
                if (c == '\n') sb.sungetc();
                return true;
            }
        }
        word.push_back(c);
    }
    in.get();

    return !word.empty();
}


/**
 * Get the new transposed shape {result: transposed_shape}.
 * @param src
 * @param transposed_shape
 * @param trans
 * @example 1:
 *   src: {4, 5, 6, 7}
 *   trans: {0, 1, 3, 2}
 *   transposed_shape: {4, 5, 7, 6}
 *
 * @example 2:
 *   src: {6, 5, 4}
 *   trans: {1, 0, 2}
 *   transposed_shape: {5, 6, 4}
 *
 * @example 3:
 *   src: {6, 5, 4}
 */
void get_transposed_shape(vector<int> &src,
                          vector<int> trans,
                          vector<int> &transposed_shape,
                          bool check_src=false){
    try{
        if (check_src){
            string nums = "vector<int>&src == [";
            for (int num: src){
                nums += std::to_string(num) + ',';
            }
            nums = nums.substr(0, nums.size()-1) + "]";

            for (int num: src)
                if (num <= 0) throw std::invalid_argument(nums + ":" + std::to_string(num) + " <= 0\n");
        }
        pair<int, int> p = only_one_inverted_order(trans);
        transposed_shape = src;
        if (p.first == -1 && p.second == -1) {
            std::cerr << "There is no valid transpose operation!\n";
            return;
        }
        swap(*(transposed_shape.begin() + p.first), *(transposed_shape.begin() + p.second));
    } catch (std::invalid_argument& arg){
        std::cerr << arg.what();
    }

}


#endif //BEYOND_UTILS_H

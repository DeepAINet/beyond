//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_SHAPE_H
#define BEYOND_SHAPE_H

#include <iostream>
#include <vector>
#include <fstream>
#include "common.h"

using namespace std;

class shape {
public:
    PLONG size;
    vector<int> dims;
    vector<PLONG> bulks;
public:
    shape() {
        size = 0;
    }

    shape(const vector<int> &sp);

    shape(const shape &sp);

    void get_bulks(const vector<int> &dims);

    shape &operator=(const shape &sp);

    shape &operator=(vector<int> sp);

    shape &operator+(int dim);

    bool operator==(const shape &sp);

    bool operator!=(const shape &sp);

    int operator[](int axis) const;

    const string to_str() const;

    int ndims() const;

    bool empty() const;
};

inline int shape::operator[](int axis) const {
    assert(!dims.empty());
    assert(axis < dims.size() || axis == -1);
    if (axis != -1)
        return dims[axis];
    return dims.back();
}

inline bool shape::empty() const {
    return dims.empty();
}

inline int shape::ndims() const {
    return dims.size();
}


#endif //BEYOND_SHAPE_H

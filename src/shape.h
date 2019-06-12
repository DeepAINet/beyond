//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_SHAPE_H
#define BEYOND_SHAPE_H

#include <iostream>
#include <vector>
#include <fstream>
#include "common.h"

class shape {
public:
    PLONG size;
    vector<int> dims;
private:
    vector<PLONG> bulks;
public:
    shape(){
        size=0;
    }
    shape(const vector<int>& sp);
    shape(const shape& sp);
    void get_bulks(const vector<int>& dims);
    shape& operator=(const shape& sp);
    shape& operator=(vector<int> sp);
    bool operator==(const shape& sp);
    bool operator!=(const shape& sp);
    int operator[](int axis) const;
    const string to_str() const;
    int ndims() const;
    bool empty() const;
};

inline int shape::operator[](int axis) const{
    assert(axis < dims.size());
    return dims[axis];
}

inline bool shape::empty() const{
    return dims.empty();
}

inline int shape::ndims() const{
    return dims.size();
}


#endif //BEYOND_SHAPE_H

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
    PLINT size;
private:
    vector<PLINT> bulks;
    vector<int> dims;
public:
    shape(){
        size=0;
    }
    shape(const vector<int>&);
    shape(const shape& c);
    void get_bulks(const vector<int>& dims);
    shape& operator=(const shape& s);
    shape& operator=(vector<int> s);
    bool operator==(const shape& s);
    bool operator!=(const shape& s);
    int operator[](int axis) const;
    string to_str();
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


#endif //BEYOND_SHAPE_H

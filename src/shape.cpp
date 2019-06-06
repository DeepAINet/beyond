//
// Created by mengqy on 2019/6/4.
//
#include "shape.h"

shape::shape(const vector<int>& dims){
    this->dims = dims;
    get_bulks(dims);
}

shape::shape(const shape& copy){
    this->dims = copy.dims;
    this->bulks = copy.bulks;
    this->size = copy.size;
}

void shape::get_bulks(const vector<int>& dims){
    if (dims.empty()) return;
    bulks.clear();
    bulks.resize(dims.size());
    bulks[dims.size()-1] = 1;
    if (dims.size() == 1){
        size = dims[0] * bulks[0];
        return;
    }

    for (int i = dims.size() - 2; i >= 0; --i){
        bulks[i] = bulks[i+1] * dims[i+1];
    }
    size = dims[0] * bulks[0];
}

shape& shape::operator=(const shape& s){
    this->dims = s.dims;
    this->bulks = s.bulks;
    size = s.size;
    return *this;
}

shape& shape::operator=(vector<int> s){
    this->dims = s;
    get_bulks(s);
    return *this;
}

bool shape::operator==(const shape& s){
    return this->dims == s.dims;
}

bool shape::operator!=(const shape& s){
    return !(*this == s);
}

string shape::to_str(){
    string res = "[shape=(";
    for (int i = 0; i < dims.size(); ++i){
        res += std::to_string(dims[i]) + ",";
    }
    if (DEBUG){
        std::cout << "bulks=[";
        for(int i = 0; i < bulks.size(); ++i){
            std::cout << bulks[i] << ',';
        }
        std::cout << "]\n";
    }

    if (dims.size() > 0) res.erase(res.size()-1, 1);
    res += "), ";
    res += "ndims=" + std::to_string(dims.size()) + ", size=" + std::to_string(size) +"]";
    return res;
}
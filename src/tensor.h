//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_TENSOR_H
#define BEYOND_TENSOR_H
#include <string>
#include "shape.h"
#include "common.h"

template <typename T>
class tensor {
public:
    string name;
private:
    shape sp;
    T *elements;

public:
    tensor(){
        elements=0;
    };

    tensor(const shape& sp):sp(sp){
        alloc();
    }

    void alloc(){
        int a = posix_memalign((void **)&elements, ALIGN_SIZE, sp.size * sizeof(T));
        if (a || 0 == elements){
            std::cerr << "allocate memory failure\n";
            exit(EXIT_FAILURE);
        }
    }

    T *data() const{
        return elements;
    }

    PLINT size() const {
        return sp.size;
    }

    ~tensor(){
        if (elements != 0) {
            free(elements);
            elements = 0;
        }
    }

    string to_string(){
        return sp.to_str();
    }

    string show(){
        int dims=sp.ndims();
        string s = "[";
        for (PLINT i = 0; i < sp.size; ++i){
            s += std::to_string(elements[i]) + ", ";
        }
        s +=  "]";
        return s;
    }

};

#endif //BEYOND_TENSOR_H

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
        elements = 0;
    };

    tensor(const shape& sp):sp(sp){
        elements = 0;
        alloc();
    }

    tensor(vector<int> sp):sp(sp){
        elements = 0;
        alloc();
    }


    void alloc(){
        int a = posix_memalign((void **)&elements, ALIGN_SIZE, sp.size * sizeof(T));
        if (a || 0 == elements){
            std::cerr << "allocate memory failure\n";
            exit(EXIT_FAILURE);
        }
    }

    T *data() const {
        return elements;
    }

    PLONG size() const {
        return sp.size;
    }

    ~tensor(){
        if (0 != elements) {
            free(elements);
            elements = 0;
        }
    }

    string to_string(){
        return sp.to_str();
    }

    string show(){
        int dims = sp.ndims();
        string s = "[";
        for (PLONG i = 0; i < sp.size; ++i){
            s += std::to_string(elements[i]) + ",";
        }
        if (sp.size > 0) s = s.substr(0, s.size()-1);
        s +=  "]";
        return s;
    }

    const shape& get_shape() const{
        return sp;
    }

    void reshape(const shape& sp){
        if (this->sp == sp) return;

        if (this->sp.size == sp.size){
            this->sp = sp;
            return;
        }

        this->sp = sp;
        LONG size = sp.size;
        if (0 != elements) {
            free(elements);
            elements = 0;
        }
        alloc();
        for (LONG i = 0; i < size; ++i)
            elements[i] = 0;

    }

    void reshape(vector<int> sp){
        if (this->sp.dims == sp) return;

        LONG size = 1;
        for (int n: sp) size *= n;
        if (this->sp.size == size){
            this->sp = sp;
            return;
        }

        if (0 != elements) {
            free(elements);
            elements = 0;
        }
        this->sp = sp;
        alloc();
        for (LONG i = 0; i < this->sp.size; ++i)
            elements[i] = 0;
    }
};

#endif //BEYOND_TENSOR_H

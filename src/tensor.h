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
    static int      idx;
    string          name;
private:
    shape           sp;
    T*              elements;

    tensor(vector<int>& sp, T *src):sp(sp){
        elements = 0;
        alloc();
        memcpy(elements, src, sizeof(T) * this->sp.size);
    }

public:
    tensor():name("T" + std::to_string(idx++)){
        elements = 0;
    };

    tensor(const shape& sp, string name="T"):sp(sp), name("T" + std::to_string(idx++)){
        elements = 0;
        alloc();
    }

    tensor(vector<int> sp, string name="T"):sp(sp), name("T" + std::to_string(idx++)){
        elements = 0;
        alloc();
    }

    tensor(const tensor<T>& t){
        sp = t.sp;
        name = t.name + "-copy";
        elements = 0;
        alloc();
        memcpy(elements, t.elements, sp.size * sizeof(T));
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

    tensor<T> operator[](int x){
        assert(x >= 0 && x < sp[0]);
        T *p = elements + x * sp.bulks[0];
        vector<int> sub_sp(sp.dims.begin()+1, sp.dims.end());
        tensor<T> sub_tensor(sub_sp, p);
        return sub_tensor;
    }

    string to_string(){
        string res = sp.to_str();
        res = "[name=\'" + name + "\'," + res.substr(1, res.size()-2) + "]";
        return res;
    }

    string show(){
        string s = "[";
        int ndims = sp.ndims();
        if (ndims == 1){
            T *p = elements;
            for (int i = 0; i < sp[0]; ++i){
                s += std::to_string(*p++) + ",";
            }
            s = s.substr(0, s.size()-1);
        }else if (ndims > 1){
            for (int i = 0; i < sp[0]; ++i){
                s += (*this)[i].show() + ",\n";
            }
            s = s.substr(0, s.size()-2);
        }
        s +=  "]";
        return s;
    }

    const shape& get_shape() const{
        return sp;
    }

    void reshape(const shape& sp){
//        if (this->sp == sp) return;

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

template <typename T>
int tensor<T>::idx = 0;

#endif //BEYOND_TENSOR_H

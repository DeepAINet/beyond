//
// Created by mengqy on 2019/6/22.
//

#ifndef BEYOND_CORPUS_H
#define BEYOND_CORPUS_H

#include <string>
#include <iostream>
#include <fstream>
#include "common.h"
#include "tensor.h"
#include "global.h"

template <typename T>
class corpus{
public:
    string          filename;
    ifstream        in;
    PLONG           size;
    int             x_dim;
    int             num_classes;
public:
    corpus(string filename):filename(filename), in(filename){
        in >> size >> x_dim >> num_classes;
    }

    void reset(){
        in.clear();
        in.seekg(0, ios::beg);
        in >> size >> x_dim >> num_classes;
    }

    void dense_input_fn(tensor<T>& x, tensor<T>& y){
        T *px=x.data(), *py=y.data();
        int bsize = x.get_shape()[0];
        int idx = 0;
        while(!in.eof()){
            for(int i = 0; i < x_dim; ++i, ++px)
                in >> *px;
            in >> *py;
            ++py;
            ++idx;
            if (idx == bsize) return;
            if (in.eof()) {
                reset();
                current_epoch++;
            }
        }
    }

    void sparse_input_fn(tensor<T>& x, tensor<T>& y){
        return;
    }
};

#endif //BEYOND_CORPUS_H

//
// Created by mengqy on 2019/6/6.
//

#ifndef BEYOND_INITIALIZERS_H
#define BEYOND_INITIALIZERS_H

#include <iostream>
#include <random>
#include "tensor.h"
#include "common.h"

template <typename T>
void uniform_initializer(T low, T high, tensor<T>& ts){
    std::default_random_engine engine(time(0));
    std::uniform_real_distribution<T> dis(low, high);
    auto dice = std::bind(dis, engine);
    T *data = ts.data();
    for (PLONG i = 0; i < ts.size(); ++i){
        *data++ = dice();
    }
}

template <typename T>
void zeros(tensor<T>& ts){
    T *data = ts.data();
    for (PLONG i = 0; i < ts.size(); ++i){
        *data++ = 0;
    }
}

#endif //BEYOND_INITIALIZERS_H

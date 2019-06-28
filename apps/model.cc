//
// Created by mengqy on 2019/6/23.
//
#include "../src/common.h"
#include "../src/tensor.h"

class model{
    virtual void train_input_fn(tensor<real>& x, tensor<real>& y)=0;
    virtual void eval_input_fn(tensor<real>& x, tensor<real>& y)=0;
    virtual void model_fn()=0;
};
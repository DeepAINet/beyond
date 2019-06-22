//
// Created by mengqy on 2019/6/12.
//

#ifndef BEYOND_NN_H
#define BEYOND_NN_H

#include "common.h"
#include "variable.h"
#include "global_ops.h"
using namespace glops;

variable* dense(variable& input,
                int output_dim,
                bool add_bias=true,
                string act_name="ReLU",
                real dropout_rate=0.5f){
    variable& weights = get_variable("w", {input.get().get_shape()[-1], output_dim});
    variable& bias = get_variable("b", {output_dim});
    variable& wx = glops::dot_mul(input, weights, false, false);
    variable& wxb = wx + bias;
    variable& res = glops::relu(wxb);
    return &res;
}

variable& softmax_entropy_with_logits(variable& input, variable& label){
    return 0;
}

#endif //BEYOND_NN_H

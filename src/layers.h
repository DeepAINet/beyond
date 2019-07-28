//
// Created by mengqy on 2019/6/12.
//

#ifndef BEYOND_NN_H
#define BEYOND_NN_H

#include "common.h"
#include "variable.h"
#include "global.h"
#include "global_ops.h"

typedef variable &(*activation)(variable &);

#define None ""

/**
 * The full-connected layer, when weight_name or bias_name is an empty string,
 * a new weight or bias will be generated.
 * @param input: variable, the input variable.
 * @param output_dim: int, the dim of output.
 * @param add_bias: bool, indicates whether to add bias.
 * @param act: function pointer, point to the activation function.
 * @return a output variable of full-connected layer.
 */
variable &dense(variable &input,
                int output_dim,
                bool add_bias = true,
                string weight_name = None,
                string bias_name = None,
                activation act = 0) {
    if (weight_name == None)
        weight_name = "w" + std::to_string(weight_idx++);
    if (bias_name == None)
        bias_name = "b" + std::to_string(bias_idx++);

    //初始化以后需要修改.
    variable &weight = get_variable(weight_name,
                                    {input.get().get_shape()[-1], output_dim},
                                    true, true, 0.0f, 0.0f);
    variable *wxb = &(glops::dot_mul(input, weight, false, false));
    if (add_bias) {
        variable &bias_ = get_variable(bias_name,
                                       {output_dim},
                                       true, true, 0.0f, 0.0f);
        variable &wxb_ = *wxb + bias_;
        wxb = &wxb_;
    }

    if (act == 0)  return *wxb;
    return (*act)(*wxb);
}

/**
 * The dropout layer.
 * @param input
 * @param rate
 * @param training
 * @return
 */
variable &dropout(variable& input, real rate, bool training){

}


#endif //BEYOND_NN_H

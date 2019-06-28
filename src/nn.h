//
// Created by mengqy on 2019/6/12.
//

#ifndef BEYOND_NN_H
#define BEYOND_NN_H

#include "common.h"
#include "variable.h"
#include "global.h"
typedef variable& (*activation)(variable&);

#define None ""
/**
 *
 * @param input
 * @param output_dim
 * @param add_bias
 * @param act_name
 * @param dropout_rate
 * @return
 */
variable& dense(variable& input,
                int output_dim,
                bool add_bias=true,
                string weight_name=None,
                string bias_name=None,
                activation act=0,
                real dropout_rate=0.5f){
    if (weight_name == None){
        weight_name = "w" + std::to_string(weight_idx++);
    }
    if (bias_name == None){
        bias_name = "b" + std::to_string(bias_idx++);
    }
    variable& weight = get_variable(weight_name,
                                    {input.get().get_shape()[-1], output_dim},
                                    true, false, 0.0f, 0.0f);
    variable& bias = get_variable(bias_name,
                                  {output_dim},
                                  true, false, 0.0f, 0.0f);
    variable& wx = variable::dot_mul(input, weight, false, false);
    variable& wxb = wx + bias;
    if (act != 0) {
        variable& res = (*act)(wxb);
        return res;
    }
    return wxb;

}


#endif //BEYOND_NN_H

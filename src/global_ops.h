//
// Created by mengqy on 2019/6/29.
//
/**
 * Copyright 2019 by Contributors
 * \file  global_op.h
 * \brief Global operations commonly used to construct neural network models，
 *        with operating object is variable.
 * \author Qingyang Meng.
 */

#ifndef BEYOND_GLOBAL_OPS_H
#define BEYOND_GLOBAL_OPS_H

#include "variable.h"

namespace glops{
    variable& dot_mul(variable& a, variable& b, bool a_transpose, bool b_transpose){
        string dm_name = "(" + a.name + "*" + b.name + ")";
        const shape& asp=a.get().get_shape();
        const shape& bsp=b.get().get_shape();
        vector<int> ssp(2, 0);
        if (!a_transpose && !b_transpose){
            ssp[0] = a.get().get_shape()[0];
            ssp[1] = b.get().get_shape()[1];
        } else if (a_transpose && !b_transpose){
            ssp[0] = a.get().get_shape()[1];
            ssp[1] = b.get().get_shape()[1];
        } else if (a_transpose && b_transpose){
            ssp[0] = a.get().get_shape()[1];
            ssp[1] = b.get().get_shape()[0];
        } else {
            ssp[0] = a.get().get_shape()[0];
            ssp[1] = b.get().get_shape()[0];
        }
        variable& res = variable::get_variable(dm_name, ssp, false, true);
        res.operation = new ops::dot_mul<real>(&a.get(), &b.get(), &res.get(), a_transpose, b_transpose);
        res.add_input(&a);
        res.add_input(&b);
        return res;
    }

    variable& sigmoid(variable& a){
        variable& res = variable::get_variable("sigmoid(" + a.name + ")",
                                               a.get().get_shape().dims, false, true);
        res.add_input(&a);
        res.operation = new ops::sigmoid<real>(&a.get(), &res.get());
        return res;
    }

    variable& tanh(variable& a){
        variable& res = variable::get_variable("tanh(" + a.name + ")",
                                               a.get().get_shape().dims, false, true);
        res.add_input(&a);
        res.operation = new ops::tanh<real>(&a.get(), &res.get());
        return res;
    }

    variable& softmax_cross_entropy_with_logits(variable& predict, variable& label, variable& loss){
        variable& res = variable::get_variable("entropy(" + predict.name + ")",
                                               predict.get().get_shape().dims, false, true);
        loss.add_input(&predict);
        loss.add_input(&label);
        res.operation = new ops::softmax_cross_entropy_with_logits(&predict.get(), &res.get(), &loss.get());
        return res;
    }
};

#endif //BEYOND_GLOBAL_OPS_H

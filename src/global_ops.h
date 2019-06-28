//
// Created by mengqy on 2019/6/9.
//
#pragma  once
#ifndef BEYOND_GLOBAL_OPS_H
#define BEYOND_GLOBAL_OPS_H

#include <string>
#include <vector>
#include "shape.h"
#include "variable.h"
class variable;

namespace glops{
//    variable& dot_mul(variable& a, variable& b, bool a_transpose, bool b_transpose){
//        string dm_name = "(" + a.name + "dot_mul" + b.name + ")";
//        const shape& asp=a.get().get_shape();
//        const shape& bsp=b.get().get_shape();
//        vector<int> ssp(2, 0);
//        if (!a_transpose && !b_transpose){
//            ssp[0] = a.get().get_shape()[0];
//            ssp[1] = b.get().get_shape()[1];
//        } else if (a_transpose && !b_transpose){
//            ssp[0] = a.get().get_shape()[1];
//            ssp[1] = b.get().get_shape()[1];
//        } else if (a_transpose && b_transpose){
//            ssp[0] = a.get().get_shape()[1];
//            ssp[1] = b.get().get_shape()[0];
//        } else {
//            ssp[0] = a.get().get_shape()[0];
//            ssp[1] = b.get().get_shape()[0];
//        }
//        variable& res = get_variable(dm_name, ssp, false, true, 0.0f, 0.0f);
//        res.operation = new ops::dot_mul<real>(&a.get(), &b.get(), &res.get(), a_transpose, b_transpose);
//        res.add_input(&a);
//        res.add_input(&b);
//        return res;
//    }

//    variable& sigmoid(variable& a){
//        variable& res = get_variable("sigmoid(" + a.name + ")",
//                                     a.get().get_shape().dims, false, true, 0.0f, 0.0f);
//        res.add_input(&a);
//        res.operation = new ops::sigmoid<real>(&a.get(), &res.get());
//        return res;
//    }

//    variable& tanh(variable& a){
//        variable& res = get_variable("tanh(" + a.name + ")",
//                                     a.get().get_shape().dims, false, true, 0.0f, 0.0f);
//        res.add_input(&a);
//        res.operation = new ops::tanh<real>(&a.get(), &res.get());
//        return res;
//    }

//    variable& softmax_cross_entropy_with_logits(variable& predict, variable& label, variable& loss){
//        variable& res = get_variable("entropy(" + predict.name + ")",
//                                     predict.get().get_shape().dims, false, true, 0.0f, 0.0f);
//        loss.add_input(&predict);
//        loss.add_input(&label);
//        res.operation = new ops::softmax_cross_entropy_with_logits(&predict.get(), &res.get(), &loss.get());
//        return res;
//    }

    variable* add(variable* a, variable* b){
        string sum_name = "("+ a->name + "+" + b->name + ")";
        const shape& asp=a->get().get_shape();
        const shape& bsp=b->get().get_shape();
        variable *psum=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& sum = get_variable(sum_name, asp.dims, false, true, 0.0f, 0.0f);
            sum.add_input(a);
            sum.add_input(b);
            psum = &sum;
        } else {
            variable& sum = get_variable(sum_name, bsp.dims, false, true, 0.0f, 0.0f);
            sum.add_input(a);
            sum.add_input(b);
            psum = &sum;
        }
        psum->operation = new ops::add<real>(&a->get(), &b->get(), &psum->get());
        return psum;
    }

    variable* subtract(variable* a, variable* b){
        string sub_name = "("+ a->name + "-" + b->name + ")";
        const shape& asp=a->get().get_shape();
        const shape& bsp=b->get().get_shape();
        variable *psub=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& sub = get_variable(sub_name, asp.dims, false, true, 0.0f, 0.0f);
            sub.add_input(a);
            sub.add_input(b);
            psub = &sub;
        } else {
            variable& sub = get_variable(sub_name, bsp.dims, false, true, 0.0f, 0.0f);
            sub.add_input(a);
            sub.add_input(b);
            psub = &sub;
        }
        psub->operation = new ops::subtract<real>(&a->get(), &b->get(), &psub->get());
        return psub;
    }

//    variable& mul(variable& a, variable& b){
//        string mul_name = "("+ a.name + "*" + b.name + ")";
//        const shape& asp=a.get().get_shape();
//        const shape& bsp=b.get().get_shape();
//        variable *pmul=0;
//        if (asp.ndims() > bsp.ndims()) {
//            variable& mul_ = get_variable(mul_name, asp.dims, false, true, 0.0f, 0.0f);
//            mul_.add_input(&a);
//            mul_.add_input(&b);
//            pmul = &mul_;
//        } else {
//            variable& mul_ = get_variable(mul_name, bsp.dims, false, true, 0.0f, 0.0f);
//            mul_.add_input(&a);
//            mul_.add_input(&b);
//            pmul = &mul_;
//        }
//        pmul->operation = new ops::mul<real>(&a.get(), &b.get(), &pmul->get());
//        return *pmul;
//    }
//
//    variable& div(variable& a, variable& b){
//        string div_name = "("+ a.name + "*" + b.name + ")";
//        const shape& asp=a.get().get_shape();
//        const shape& bsp=b.get().get_shape();
//        variable *pdiv=0;
//        if (asp.ndims() > bsp.ndims()) {
//            variable& div_ = get_variable(div_name, asp.dims, false, true, 0.0f, 0.0f);
//            div_.add_input(&a);
//            div_.add_input(&b);
//            pdiv = &div_;
//        } else {
//            variable& div_ = get_variable(div_name, bsp.dims, false, true, 0.0f, 0.0f);
//            div_.add_input(&a);
//            div_.add_input(&b);
//            pdiv = &div_;
//        }
//        pdiv->operation = new ops::div<real>(&a.get(), &b.get(), &pdiv->get());
//        return *pdiv;
//    }
}

#endif //BEYOND_GLOBAL_OPS_H

//
// Created by mengqy on 2019/6/9.
//

#pragma once
#ifndef BEYOND_VARIABLE_H
#define BEYOND_VARIABLE_H
#include <string>
#include <vector>
#include <map>
#include "tensor.h"
#include "ops.h"
#include "global.h"
#include "initializers.h"
#include "node.h"

using namespace ops;


class variable: public node {
public:
    static int idx;
    static map<string, variable *> global_variables;
public:
    variable(const shape &sp, bool trainable = false, bool grad = true, string name = "V" + std::to_string(idx++))
    :node(name, sp, trainable, grad) {}

    variable(vector<int> sp, bool trainable = false, bool grad = true, string name = "V" + std::to_string(idx++))
    :node(name, sp, trainable, grad) {}

    ~variable() {}

    string to_str() {
        string tname = self.to_string();
        tname = tname.substr(1, tname.size() - 2);
        string str = "[" + tname + ",";

        if (inputs.size() == 0) {
            str += "inputs=None,";
        } else {
            str += "inputs=[";
            for (node *input: inputs) {
                str += input->name + ",";
            }
            str = str.substr(0, str.size() - 1) + "],";
        }

        if (!trainable || operation == 0) str += "op=None,";
        else {
            str += "op=" + operation->name + ",";
        }

        if (outputs.size() == 0) {
            str += "outputs=None,";
        } else {
            str += "outputs=[";
            for (node *output: outputs) {
                str += output->name + ",";
            }
            str = str.substr(0, str.size() - 1) + "],";
        }
        str = str.substr(0, str.size() - 1) + "]";
        return str;
    }

    static variable &get_variable(string name, const shape &sp, bool trainable = false, bool grad = true) {
        if (global_variables.count(name) == 1)
            return *global_variables[name];

        variable *v = new variable(sp, trainable, grad, name);
        global_variables[name] = v;
        if (trainable) {
            grad_tables[name] = new tensor<real>(sp);
        }
        return *v;
    }

    variable &operator+(variable &a) {
        return add(*this, a);
    }

    variable &operator-(variable &a) {
        return subtract(*this, a);
    }

    variable &operator*(variable &a) {
        return mul(*this, a);
    }

    variable &operator/(variable &a) {
        return div(*this, a);
    }

    static variable& dot_mul(variable& a, variable& b, bool a_transpose, bool b_transpose){
        string dm_name = "(" + a.name + "dot_mul" + b.name + ")";
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
        variable& res = get_variable(dm_name, ssp, false, true);
        res.operation = new ops::dot_mul<real>(&a.get(), &b.get(), &res.get(), a_transpose, b_transpose);
        res.add_input(&a);
        res.add_input(&b);
        return res;
    }

    static variable& sigmoid(variable& a){
        variable& res = get_variable("sigmoid(" + a.name + ")",
                                     a.get().get_shape().dims, false, true);
        res.add_input(&a);
        res.operation = new ops::sigmoid<real>(&a.get(), &res.get());
        return res;
    }

    static variable& tanh(variable& a){
        variable& res = get_variable("tanh(" + a.name + ")",
                                     a.get().get_shape().dims, false, true);
        res.add_input(&a);
        res.operation = new ops::tanh<real>(&a.get(), &res.get());
        return res;
    }

    static variable& softmax_cross_entropy_with_logits(variable& predict, variable& label, variable& loss){
        variable& res = get_variable("entropy(" + predict.name + ")",
                                     predict.get().get_shape().dims, false, true);
        loss.add_input(&predict);
        loss.add_input(&label);
        res.operation = new ops::softmax_cross_entropy_with_logits(&predict.get(), &res.get(), &loss.get());
        return res;
    }

    static variable& add(variable& a, variable& b){
        string sum_name = "("+ a.name + "+" + b.name + ")";
        const shape& asp=a.get().get_shape();
        const shape& bsp=b.get().get_shape();
        variable *psum=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& sum = get_variable(sum_name, asp.dims, false, true);
            sum.add_input(&a);
            sum.add_input(&b);
            psum = &sum;
        } else {
            variable& sum = get_variable(sum_name, bsp.dims, false, true);
            sum.add_input(&a);
            sum.add_input(&b);
            psum = &sum;
        }
        psum->operation = new ops::add<real>(&a.get(), &b.get(), &psum->get());
        return *psum;
    }

    static variable& subtract(variable& a, variable& b){
        string sub_name = "("+ a.name + "-" + b.name + ")";
        const shape& asp=a.get().get_shape();
        const shape& bsp=b.get().get_shape();
        variable *psub=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& sub = get_variable(sub_name, asp.dims, false, true);
            sub.add_input(&a);
            sub.add_input(&b);
            psub = &sub;
        } else {
            variable& sub = get_variable(sub_name, bsp.dims, false, true);
            sub.add_input(&a);
            sub.add_input(&b);
            psub = &sub;
        }
        psub->operation = new ops::subtract<real>(&a.get(), &b.get(), &psub->get());
        return *psub;
    }

    static variable& mul(variable& a, variable& b){
        string mul_name = "("+ a.name + "*" + b.name + ")";
        const shape& asp=a.get().get_shape();
        const shape& bsp=b.get().get_shape();
        variable *pmul=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& mul_ = get_variable(mul_name, asp.dims, false, true);
            mul_.add_input(&a);
            mul_.add_input(&b);
            pmul = &mul_;
        } else {
            variable& mul_ = get_variable(mul_name, bsp.dims, false, true);
            mul_.add_input(&a);
            mul_.add_input(&b);
            pmul = &mul_;
        }
        pmul->operation = new ops::mul<real>(&a.get(), &b.get(), &pmul->get());
        return *pmul;
    }

    static variable& div(variable& a, variable& b){
        string div_name = "("+ a.name + "*" + b.name + ")";
        const shape& asp=a.get().get_shape();
        const shape& bsp=b.get().get_shape();
        variable *pdiv=0;
        if (asp.ndims() > bsp.ndims()) {
            variable& div_ = get_variable(div_name, asp.dims, false, true);
            div_.add_input(&a);
            div_.add_input(&b);
            pdiv = &div_;
        } else {
            variable& div_ = get_variable(div_name, bsp.dims, false, true);
            div_.add_input(&a);
            div_.add_input(&b);
            pdiv = &div_;
        }
        pdiv->operation = new ops::div<real>(&a.get(), &b.get(), &pdiv->get());
        return *pdiv;
    }
};

int variable::idx = 0;
map<string, variable *> variable::global_variables;

typedef vector<variable *> graph;
typedef vector<graph> graphs;

template<typename T>
void init_variable(T low, T high, tensor<T> &tensor1, string fn_name) {
    if (fn_name == "uniform") {
        uniform_initializer(low, high, tensor1);
    }
}

template<typename T>
variable &get_variable(string name,
                       vector<int> sp,
                       bool trainable = true,
                       bool grad = true,
                       T low = 0.0f,
                       T high = 0.0f,
                       string initializer = "uniform") {
    if (variable::global_variables.count(name) == 1)
        return *(variable::global_variables[name]);

    variable *v = new variable(sp, trainable, grad, name);
    init_variable(low, high, v->get(), initializer);
    variable::global_variables[name] = v;
    if (trainable) {
        grad_tables[name] = new tensor<real>(sp);
    }
    return *v;
}


#endif

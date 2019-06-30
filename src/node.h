//
// Created by mengqy on 2019/6/28.
//

#ifndef BEYOND_NODE_H
#define BEYOND_NODE_H

#include "ops.h"
using namespace ops;

class node {
public:
    string          name;              // the name of the node or variable.
    vector<node *>  outputs;           // the output variable when this node as input variable.
    vector<node *>  inputs;            // the input variables of the node operation.
    op *            operation;         // the corresponding operation.
    tensor<real>    self;              // the corresponding tensor.
    bool            trainable;         // indicate whether to update grad_table.
    bool            grad;              // indicate whether to compute grad.

protected:
    node(){}
public:
    node(string &name, const shape &sp, bool trainable, bool grad)
            :name(name), self(sp, name) {
        operation = 0;
        this->trainable = trainable;
        this->grad = grad;
    }

    node(string &name, const vector<int> &sp, bool trainable, bool grad)
            :name(name), self(sp, name) {
        operation = 0;
        this->trainable = trainable;
        this->grad = grad;
    }

    ~node() {
        if (0 != operation) delete operation;
    }

    void forward() {
        if (0 == operation) return;
        operation->forward();
    }

    void backward() {
        if (0 == operation) return;
        operation->backward();
    }

    tensor<real> &get() {
        return self;
    }

    void add_input(node *input) {
        if (0 == input) {
            std::cerr << "node *input=0";
            return;
        }
        inputs.push_back(input);
        input->add_output(this);
    }

    string& get_name(){
        return self.name;
    }

private:
    void add_output(node *output) {
        outputs.push_back(output);
    }
};


#endif //BEYOND_NODE_H

//
// Created by mengqy on 2019/6/9.
//

#ifndef BEYOND_VARIABLE_H
#define BEYOND_VARIABLE_H

#include <string>
#include <vector>
#include <map>
#include "tensor.h"
#include "ops.h"
#include "global.h"
#include "initializers.h"

using namespace ops;

class node{
public:
    string name;
    vector<node*> outputs;
    vector<node*> inputs;
    op* operation=0;
    tensor<real> self;
    bool trainable=true;

public:
    node(string& name, const shape& sp)
    :name(name), self(sp){
        self.name = name;
    }

    node(string& name, vector<int>& sp)
    :name(name), self(sp){
        self.name = name;
    }

    ~node(){
        if (0 != operation) delete operation;
    }

    void forward(){
        if (!trainable) return;
        operation->forward();
    }

    void backward(){
        if (!trainable) return;
        operation->backward();
    }

    tensor<real>& get(){
        return self;
    }

    void add_input(node *input){
        if (0 == input){
            std::cout <<"input=0\n";
            return;
        }
        inputs.push_back(input);
        input->add_output(this);
    }

private:
    void add_output(node *output){
        outputs.push_back(output);
    }
};

class variable: public node{
public:
    static int idx;
    static map<string, variable*> global_variables;
public:
    variable(const shape& sp, string name="V" + std::to_string(idx++))
    :node(name, sp){}

    variable(vector<int> sp, string name="V" + std::to_string(idx++))
    :node(name, sp){}

    ~variable(){}

    string to_str(){
        string tname = self.to_string();
        tname = tname.substr(1, tname.size()-2);
        string str = "[" + tname + ",";

        if (inputs.size() == 0){
            str += "inputs=None,";
        } else {
            str += "inputs=[";
            for (node* input: inputs){
                str += input->name + ",";
            }
            str = str.substr(0, str.size()-1) + "],";
        }

        if (!trainable || operation == 0) str += "op=None,";
        else {
            str += "op=" + operation->name + ",";
        }

        if (outputs.size() == 0){
            str += "outputs=None,";
        }else{
            str += "outputs=[";
            for (node* output: outputs){
                str += output->name + ",";
            }
            str = str.substr(0, str.size()-1) +  "],";
        }
        str = str.substr(0, str.size()-1) + "]";
        return str;
    }

    variable& get_variable(string name, const shape& sp, bool trainable=true){
        if (global_variables.count(name) == 1)
            return *global_variables[name];

        variable *v = new variable(sp, name);
        global_variables[name] = v;
        if (trainable) {
            grad_tables[name] = new tensor<real>(sp);
        }
        return *v;
    }

    variable& operator+(variable& a){
        string des_name = name + "+" + a.name;
        variable& des = get_variable(des_name, self.get_shape());
        des.add_input(&a);
        des.add_input(this);
        return des;
    }

    variable& operator-(variable& a){
        string des_name = name + "-" + a.name;
        variable& des = get_variable(des_name, self.get_shape());
        des.add_input(&a);
        des.add_input(this);
        return des;
    }

    variable& operator*(variable& a){
        string des_name = name + "*" + a.name;
        variable& des = get_variable(des_name, self.get_shape());
        des.add_input(&a);
        des.add_input(this);
        return des;
    }

    variable& operator/(variable& a){
        string des_name = name + "/" + a.name;
        variable& des = get_variable(des_name, self.get_shape());
        des.add_input(&a);
        des.add_input(this);
        return des;
    }
};

int variable::idx = 0;
map<string, variable*> variable::global_variables;

typedef vector<variable*> graph;
typedef vector<graph> graphs;

template <typename T>
void init_variable(T low, T high, tensor<T> &tensor1, string fn_name){
    if (fn_name == "uniform"){
        uniform_initializer(low, high, tensor1);
    }
}

template <typename T>
variable& get_variable(string name,
                       vector<int> sp,
                       bool trainable=true,
                       T low=0.0f,
                       T high=0.0f,
                       string initializer="uniform"){
    if (variable::global_variables.count(name) == 1)
        return *(variable::global_variables[name]);

    variable *v = new variable(sp, name);
    init_variable(low, high, v->get(), initializer);
    variable::global_variables[name] = v;
    if (trainable) {
        grad_tables[name] = new tensor<real>(sp);
    }
    return *v;
}

#endif //BEYOND_VARIABLE_H

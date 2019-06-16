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
    bool is_training=true;

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
        if (!is_training) return;
        operation->forward();
    }

    void backward(){
        if (!is_training) return;
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

        if (!is_training || operation == 0) str += "op=None,";
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
        str = str.substr(0, str.size()-1) + "]\n";
        return str;
    }
};

int variable::idx = 0;

class graph{
public:
    static vector<vector<variable*>> graphs;
    static map<string, variable*> global_variables;

    static void construct_graphs(int iter_num){
        graphs.resize(iter_num);
    }
};

vector<vector<variable*>> graph::graphs;
map<string, variable*> graph::global_variables;

template <typename T>
void init_variable(T low, T high, tensor<T> &tensor1, string fn_name){
    if (fn_name == "uniform"){
        uniform_initializer(low, high, tensor1);
    }
}

template <typename T>
variable* get_variable(string name,
                       vector<int> sp,
                       T low=0.0f,
                       T high=0.0f,
                       string init_fn="uniform",
                       int tid=0,
                       bool is_training=true){
    if(tid >= 1){
        name = name + std::to_string(tid);
    }
    if (graph::global_variables.count(name) == 1)
        return graph::global_variables[name];

    variable *v = new variable(sp, name);
    init_variable(low, high, v->get(), init_fn);
    graph::global_variables[name] = v;
    if (tid == graph::graphs.size()){
        vector<variable*> _graph;
        graph::graphs.push_back(_graph);
    }
    graph::graphs[tid].push_back(v);
    if (is_training) {
        grad_tables[name] = new tensor<real>(sp);
    }
    return v;
}

#endif //BEYOND_VARIABLE_H

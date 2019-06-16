//
// Created by mengqy on 2019/6/9.
//

#ifndef BEYOND_GLOBAL_H
#define BEYOND_GLOBAL_H
#include <map>
#include "tensor.h"

map<string, tensor<real>*> grad_tables;

int global_steps = 0;

int total_steps = 0;

int current_epoch = 0;

int epoch_num = 1;

int every_epoch_num = 1;

int evaluate_steps = 1000;

int print_steps = 10;

int batch_size = 64;

real start_learning_rate = 0.01;

inline void add_to_trainable_variables(string& name, tensor<real> *grad){
    if (grad_tables.count(name) == 1){
        name = name + " has existed.";
        std::cerr << name << std::endl;
        exit(EXIT_FAILURE);
    }
    if (0 == grad){
        std::cerr << "add null pointer!" << std::endl;
        exit(EXIT_FAILURE);
    }
    grad_tables[name] = grad;
}

inline string partial_derivative_name(string& name){
    return "∂(loss)/∂(" + name + ")";
}

inline string partial_derivative_name(string name){
    return "∂(loss)/∂(" + name + ")";
}

void print_hyper_params(){
    std::cout << "-start learning rate: " << start_learning_rate << std::endl
              << "-batch size: " << batch_size << std::endl
              << "-epoch num: " << epoch_num << std::endl
              << "-evaluate steps: " << evaluate_steps << std::endl;

}

#endif //BEYOND_GLOBAL_H

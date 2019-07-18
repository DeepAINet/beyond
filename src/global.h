//
// Created by mengqy on 2019/6/9.
//

/**
 * Copyright 2019 by Contributors
 * \file  global.h
 * \brief Some variables commonly used in training neural networks.
 * \author Qingyang Meng.
 */

#ifndef BEYOND_GLOBAL_H
#define BEYOND_GLOBAL_H

#include <map>
#include "tensor.h"

// Gradient table for storing gradient information in each iteration.
map<string, tensor<real> *> grad_tables;

// The current number of iterations, each iteration being a batch of data.
int global_steps = 0;

// The target steps of training.
int train_steps_ = 0;

// The current epoch num.
int current_epoch = 0;

// The target num of training epochs.
int epoch_num_ = 1;

// Number of interval steps per model evaluation.
int eval_steps_ = 1000;

// Number of Interval steps to save the model or checkpoints.
int checkpoint_steps_ = 10000;

// Number of interval steps to print the log info (loss).
int print_steps = 10;

// Number of small batch data.
int batch_size_ = 64;

// Number of threads.
int num_threads_ = 8;

// The initial value of learning rate.
real start_learning_rate = 0.01;

// ID for weight naming.
int weight_idx = 0;

// ID for bias naming.
int bias_idx = 0;

inline void add_to_grad_tables(string &name, tensor<real> *grad) {
    if (grad_tables.count(name) == 1) {
        name = name + " has existed.";
        std::cerr << name << std::endl;
        exit(EXIT_FAILURE);
    }
    if (0 == grad) {
        std::cerr << "add null pointer!" << std::endl;
        exit(EXIT_FAILURE);
    }
    grad_tables[name] = grad;
}

inline string partial_derivative_name(string &name) {
    return "∂(loss)/∂(" + name + ")";
}

inline string partial_derivative_name(string name) {
    return "∂(loss)/∂(" + name + ")";
}

void print_hyper_params() {
    std::cout << "***********************************************" << std::endl
              << "\t-start learning rate: " << start_learning_rate << std::endl
              << "\t-batch size: " << batch_size_ << std::endl
              << "\t-epoch num: " << epoch_num_ << std::endl
              << "\t-evaluate steps: " << eval_steps_ << std::endl
              << "\t-print steps: " << print_steps << std::endl
              << "\t-num thread: " << num_threads_ << std::endl
              << "***********************************************" << std::endl;
}

#endif //BEYOND_GLOBAL_H

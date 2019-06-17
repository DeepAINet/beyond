//
// Created by mengqy on 2019/6/17.
//

#ifndef BEYOND_SESSION_H
#define BEYOND_SESSION_H

#include <thread>
#include <chrono>
#include "global.h"
#include "variable.h"
#include "log.h"

class session {
    graphs gs;
    vector<thread*> threads;
public:
    session():gs(num_threads), threads(num_threads, 0){
        print_hyper_params();
        clone_compute_graph();
    }

    session(int nthreads):gs(nthreads), threads(nthreads, 0){
        num_threads = nthreads;
        print_hyper_params();
        clone_compute_graph();
    }

    void clone_compute_graph(){
        std::cout << "clone compute graphs" << std::endl;
    }

    void run(){
        for (int i = 0; i < num_threads; ++i){
            threads[i] = new thread([=](){train_thread(i);});
        }

        for (int i = 0; i < num_threads; ++i){
            threads[i]->join();
        }

        for (int i = 0; i < num_threads; ++i){
            delete threads[i];
            threads[i] = 0;
        }
    }

    void update_grad_table(const vector<variable*>& g){
//        if (DEBUG) logger.info("update grad table.");
    }

    void grads2zeros(const vector<variable*>& g){
//        if (DEBUG) logger.info("grad table -> zero.");
        for (variable *v: g) zeros(v->get());
    }

    void forward(const vector<variable*>& g){
        for (variable *n: g) n->forward();
    }

    void backward(const vector<variable*>& g){
        for (auto iter = g.end() - 1; iter != g.begin() - 1; --iter)
            (*iter)->backward();
    }

    void print_loss(){
        if (global_steps != 0 && (global_steps % print_steps == 0)) {
            logger.info(std::to_string(current_epoch) + " epoch - total loss - " + std::to_string(0.0f) + " - " + std::to_string(print_steps) +" steps - " + std::to_string(logger.get_diff_time()) + " s");
        }
    }

    void evaluate(){
        if (global_steps != 0 && (global_steps % evaluate_steps == 0)){
            logger.info(std::to_string(current_epoch) + " epoch - EVAL - " + std::to_string(0.0f) + " - " + std::to_string(print_steps) +" steps - " + std::to_string(logger.get_diff_time()) + " s");
        }
    }

    void train_thread(int tid){
        if (DEBUG) chrono::seconds(10000);
        forward(gs[tid]);
        grads2zeros(gs[tid]);
        backward(gs[tid]);
        update_grad_table(gs[tid]);
        ++global_steps;
        print_loss();
        evaluate();
    }

    bool early_stop(){
        return false;
    }
};

#endif //BEYOND_SESSION_H

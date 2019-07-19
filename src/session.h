//
// Created by mengqy on 2019/6/17.
//

#ifndef BEYOND_SESSION_H
#define BEYOND_SESSION_H

#include <thread>
#include <chrono>
#include <map>
#include <queue>
#include <vector>
#include <set>
#include "variable.h"
#include "global.h"
#include "corpus.h"
#include "log.h"

typedef vector<variable *> graph;

typedef vector<graph> graphs;

typedef void (corpus<real>::*input_fn)(tensor<real> &, tensor<real> &);

class session {
    string corpus_file;
    string model_dir;
    input_fn train_input_fn;
    input_fn eval_input_fn;
    graphs gs;
    vector<thread *> threads;
public:
    session() : gs(num_threads_), threads(num_threads_, 0) {
        assert(num_threads_ >= 1);
        print_hyper_params();
        clone_compute_graph();
    }

    session(int nthreads) : gs(nthreads), threads(nthreads, 0) {
        num_threads_ = nthreads;
        assert(num_threads_ >= 1);
        print_hyper_params();
        clone_compute_graph();
    }

    void sort_graph() {
        map<string, int> ins;
        map<string, set < string>>
        in_nodes;

        for (auto iter = variable::global_variables.begin(); iter != variable::global_variables.end(); ++iter) {
            for (node *input: iter->second->inputs) {
                ++ins[iter->first];
                in_nodes[input->name].insert(iter->first);
            }
        }

        queue < variable * > compute_nodes;

        for (auto iter = variable::global_variables.begin(); iter != variable::global_variables.end(); ++iter) {
            if (ins[iter->first] == 0)
                compute_nodes.push(iter->second);
        }

        gs.resize(num_threads_);
        size_t size, i;
        variable *cnode;
        while (!compute_nodes.empty()) {
            size = compute_nodes.size();
            for (i = 0; i < size; ++i) {
                cnode = compute_nodes.front();
                compute_nodes.pop();
                gs[0].push_back(cnode);
                for (string e: in_nodes[cnode->name]) {
                    --ins[e];
                    if (ins[e] == 0) compute_nodes.push(variable::global_variables[e]);
                }
            }
        }

        if (DEBUG) {
            std::cout << gs[0].size() << " nodes\n";
            for (auto e: gs[0])
                std::cout << e->to_str() << std::endl << std::endl;
        }
    }


    void clone_compute_graph() {
        sort_graph();
        std::cout << "clone compute graphs" << std::endl;
    }

    void run_config(string corpus_file,
                    string model_dir,
                    input_fn train_input_fn,
                    input_fn eval_input_fn = 0,
                    int batch_size = 64,
                    int eval_steps = 1000,
                    int checkpoint_steps = 10000,
                    int train_steps = 100000,
                    int num_epoch = 1) {
        this->corpus_file = corpus_file;
        this->model_dir = model_dir;
        this->train_input_fn = train_input_fn;
        this->eval_input_fn = eval_input_fn;
        batch_size_ = batch_size;
        eval_steps_ = eval_steps;
        checkpoint_steps_ = checkpoint_steps;
        train_steps_ = train_steps;
        epoch_num_ = num_epoch;

        if (train_input_fn == 0) {
            throw new std::invalid_argument("train_input_fn is null pointer!. train_input_fn");
        }
    }

    void run() {
        corpus<real> corp(corpus_file);
        PLONG total_steps = (corp.size * epoch_num_) / batch_size_ + 1;
        total_steps = total_steps < train_steps_ ? total_steps : train_steps_;

        for (PLONG step_idx = 0; step_idx <= total_steps / num_threads_; ++step_idx) {
            for (int i = 0; i < num_threads_; ++i) {
                variable &x = get_variable("batch_x" + std::to_string(i), {batch_size_, corp.x_dim}, false, 0.0f, 0.0f);
                variable &y = get_variable("batch_y" + std::to_string(i), {batch_size_, 1}, false, 0.0f, 0.0f);
                (corp.*train_input_fn)(x.get(), y.get());
            }

            for (int i = 0; i < num_threads_; ++i) {
                threads[i] = new thread([=]() { train_thread(i); });
            }

            for (int i = 0; i < num_threads_; ++i) {
                threads[i]->join();
            }

            for (int i = 0; i < num_threads_; ++i) {
                delete threads[i];
                threads[i] = 0;
            }
        }
    }

    void update_grad_table(const vector<variable *> &g) {
//        if (DEBUG) logger.info("update grad table.");
    }

    void grads2zeros(const vector<variable *> &g) {
//        if (DEBUG) logger.info("grad table -> zero.");
        for (variable *v: g) zeros(v->get());
    }

    void forward(const vector<variable *> &g) {
        for (variable *n: g) n->forward();
    }

    void backward(const vector<variable *> &g) {
        for (auto iter = g.end() - 1; iter != g.begin() - 1; --iter)
            (*iter)->backward();
    }

    void print_loss() {
        if (global_steps != 0 && (global_steps % print_steps == 0)) {
            logger.info(std::to_string(current_epoch) + " epoch - total loss - " + std::to_string(0.0f) + " - " +
                        std::to_string(print_steps) + " steps - " + std::to_string(logger.get_diff_time()) + " s");
        }
    }

    void evaluate() {
        if (global_steps != 0 && (global_steps % eval_steps_ == 0)) {
            logger.info(std::to_string(current_epoch) + " epoch - EVAL - " + std::to_string(0.0f) + " - " +
                        std::to_string(print_steps) + " steps - " + std::to_string(logger.get_diff_time()) + " s");
        }
    }

    void train_thread(int tid) {
        if (DEBUG) chrono::seconds(10000);
        forward(gs[tid]);
        grads2zeros(gs[tid]);
        backward(gs[tid]);
        update_grad_table(gs[tid]);
        ++global_steps;
        print_loss();
        evaluate();
    }

    bool early_stop() {
        return false;
    }
};

#endif //BEYOND_SESSION_H

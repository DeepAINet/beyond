//
// Created by mengqy on 2019/6/12.
//

/**
 * Copyright 2019 by Contributors
 * \file  ops.h
 * \brief The operating units of the internal node of computing graph.
 *        Each operation unit mainly includes forward and backward computation.
 * \author Qingyang Meng.
 */

#ifndef BEYOND_OPS_H
#define BEYOND_OPS_H

#include <string>
#include "common.h"
#include "tensor.h"
#include "tensor_ops.h"

namespace ops {
    class op {
    public:
        string name;
    public:
        op(string name) : name(name) {}

        virtual ~op() {};

        virtual void forward() = 0;

        virtual void backward() = 0;
    };

    template<typename T>
    class dot_mul : public op {
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
        bool a_transpose;
        bool b_transpose;

    public:
        dot_mul(tensor<T> *a, tensor<T> *b, tensor<T> *des, bool a_transpose, bool b_transpose)
                : a(a), b(b), des(des), a_transpose(a_transpose), b_transpose(b_transpose), op("*") {}

        void forward() {
//            if (DEBUG) std::cout << a->name << op::name << b->name << " = "<< des->name << std::endl;
            tops::dot_mul(*a, *b, *des, a_transpose, b_transpose);
        }

        void backward() {

        }
    };

    template<typename T>
    class div : public op {
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        div(tensor<T> *a, tensor<T> *b, tensor<T> *des)
                : a(a), b(b), des(des), op("/") {}

        void forward() {
            tops::divide(*a, *b, *des);
        }

        void backward() {
        }
    };

    /**
     * Hadamard product of two matrices.
     * @tparam T
     */
    template<typename T>
    class mul : public op {
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        mul(tensor<T> *a, tensor<T> *b, tensor<T> *des)
                : a(a), b(b), des(des), op("⊙") {}

        void forward() {
            tops::mul(*a, *b, *des);
        }

        void backward() {
        }
    };

    template<typename T>
    class add : public op {
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        add(tensor<T> *a, tensor<T> *b, tensor<T> *des)
                : a(a), b(b), des(des), op("+") {}

        void forward() {
            tops::add(*a, *b, *des);
        }

        void backward() {

        }
    };

    template<typename T>
    class subtract : public op {
    private:
        tensor<T> *a;
        tensor<T> *b;
        tensor<T> *des;
    public:
        subtract(tensor<T> *a, tensor<T> *b, tensor<T> *des)
                : a(a), b(b), des(des), op("-") {}

        void forward() {

        }

        void backward() {

        }
    };

    template<typename T>
    class sigmoid : public op {
    public:
        tensor<real> *a;
        tensor<real> *des;
    public:
        sigmoid(tensor<real> *a, tensor<real> *des)
                : a(a), des(des), op("sigmoid") {}

        void forward() {
            if (DEBUG) std::cout << a->name << op::name << des->name << std::endl;
            tops::sigmoid(*a, *des);
        }

        void backward() {

        }
    };

    template<typename T>
    class tanh : public op {
    public:
        tensor<T> *a;
        tensor<real> *des;
    public:
        tanh(tensor<T> *a, tensor<real> *des)
                : a(a), des(des), op("tanh") {}

        void forward() {
            if (DEBUG) std::cout << a->name << op::name << des->name << std::endl;
            tops::tanh(*a, *des);
        }

        void backward() {

        }
    };

    template<typename T>
    class softmax : public op {
    public:
        tensor<T> *a;
        tensor<real> *res;
    public:
        softmax(tensor<T> *a, tensor<real> *res) : a(a), res(res) {}

        void forward() {
            tops::softmax(*a, *res);
        }

        void backward() {

        }
    };

    template<typename T>
    class log : public op {
    public:
        tensor<T> *a;
        tensor<real> *des;
    public:
        log(tensor<T> *a, tensor<real> *res) : a(a), des(des) {}

        void forward() {
//            tops::ln(*a, *des);
        }

        void backward() {

        }

    };

    class softmax_cross_entropy_with_logits : public op {
    public:
        tensor<real> *predict;
        tensor<real> *label;
        tensor<real> *loss;
    public:
        softmax_cross_entropy_with_logits(tensor<real> *p, tensor<real> *label, tensor<real> *loss = 0)
                : predict(p), label(label), loss(loss), op("softmax_cross_entropy_with_logits") {}

        void forward() {
        }

        void backward() {
        }
    };

    class batch_norm : public op {
    public:
        tensor<real> *a;
        tensor<real> *m;
        tensor<real> *v;
        tensor<real> *des;
    public:
        batch_norm(tensor<real> *a, tensor<real> *m, tensor<real> *v, tensor<real> *des)
                : a(a), m(m), v(v), des(des), op("batch_norm") {}

        void forward() {

        }

        void backward() {

        }
    };

}

#endif //BEYOND_OPS_H

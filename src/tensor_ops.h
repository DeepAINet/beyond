//
// Created by mengqy on 2019/6/9.
//

#ifndef BEYOND_TENSOR_OPS_H
#define BEYOND_TENSOR_OPS_H

#include <cmath>
#include "tensor.h"
#include "common.h"
#include "utils.h"

using namespace std;

namespace tops {

    template<typename T>
    void transpose(const tensor<T> &src, tensor<T> &des, vector<int> trans) {
        const shape &sp = src.get_shape();
        assert(sp.ndims() == trans.size());
        pair<int, int> tp = only_one_inverted_order(trans);

        if (tp.first == -1 && tp.second == -1) {
            std::cerr << "Not only one inverted order." << std::endl;
            exit(EXIT_FAILURE);
        }

        int left = tp.first, right = tp.second;
        PLONG dim = sp.bulks[left];
        PLONG bulk = (right == 0 ? sp.size : sp.bulks[right - 1]);
        PLONG nbulk = sp.size / bulk;
        int outer_dim = sp[left];
        int inner_dim = bulk / (dim * outer_dim);
        assert(sp.size % bulk == 0);
        assert(bulk % (dim * outer_dim) == 0);

        //std::cout << "dim:" << dim << std::endl
        //        << "bulk:" << bulk << std::endl
        //      << "num_bulk:" << nbulk << std::endl
        //    << "outer_dim:" << outer_dim << std::endl
        //  << "inner_dim:" << inner_dim << std::endl
        //<< std::endl;

        vector<int> des_shape(sp.dims);
        swap(*(des_shape.begin() + left), *(des_shape.begin() + right));
        des.reshape(des_shape);

        T *start_p = 0, *fp = 0, *mp = 0, *pd = des.data();
        PLONG sep = outer_dim * dim;
        for (PLONG i = 0; i < nbulk; ++i) {
            start_p = src.data() + i * bulk;
            for (int j = 0; j < outer_dim; ++j) {
                fp = start_p + j * dim;
                for (int k = 0; k < inner_dim; ++k) {
                    mp = fp + k * sep;
                    for (int l = 0; l < dim; ++l) {
                        *pd++ = *mp++;
                    }
                }
            }
        }
    }

    template<typename T>
    void add(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
        const shape &asp = a.get_shape();
        const shape &bsp = b.get_shape();
        if (!same_shape_backward(asp, bsp))
            throw new std::invalid_argument(
                    "tensor:" + a.name + " and " + "tensor:" + b.name + " don't have same backward shape.");

        T *pa = 0, *pb = 0;
        PLONG asize = 0, bsize = 0;
        if (asp.ndims() > bsp.ndims()) {
            des.reshape(asp);
            pa = a.data();
            pb = b.data();
            asize = a.size();
            bsize = b.size();
        } else {
            des.reshape(bsp);
            pa = b.data();
            pb = a.data();
            asize = b.size();
            bsize = a.size();
        }
        T *pd = des.data(), *pad = pb;

        for (PLONG i = 0; i < asize; ++i) {
            if (i % bsize == 0) pb = pad;
            *pd++ = (*pa++) + (*pb++);
        }
    }

    template<typename T>
    void subtract(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
        const shape &asp = a.get_shape();
        const shape &bsp = b.get_shape();
        if (!same_shape_backward(asp, bsp))
            throw new std::invalid_argument(
                    "tensor:" + a.name + " and " + "tensor:" + b.name + " don't have same backward shape.");

        assert(asp.ndims() >= bsp.ndims());
        des.reshape(asp);

        T *pa = a.data(), *pb = b.data(), *pd = des.data(), *pad = pb;
        PLONG asize = a.size(), bsize = b.size();

        for (PLONG i = 0; i < asize; ++i) {
            if (i % bsize == 0) pb = pad;
            *pd++ = (*pa++) - (*pb++);
        }
    }

    template<typename T>
    void mul(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
        const shape &asp = a.get_shape();
        const shape &bsp = b.get_shape();
        if (!same_shape_backward(asp, bsp))
            throw new std::invalid_argument(
                    "tensor:" + a.name + " and " + "tensor:" + b.name + " don't have same backward shape.");

        T *pa = 0, *pb = 0;
        PLONG asize = 0, bsize = 0;
        if (asp.ndims() > bsp.ndims()) {
            des.reshape(asp);
            pa = a.data();
            pb = b.data();
            asize = a.size();
            bsize = b.size();
        } else {
            des.reshape(bsp);
            pa = b.data();
            pb = a.data();
            asize = b.size();
            bsize = a.size();
        }
        T *pd = des.data(), *pad = pb;

        for (PLONG i = 0; i < asize; ++i) {
            if (i % bsize == 0) pb = pad;
            *pd++ = (*pa++) * (*pb++);
        }
    }

    template<typename T>
    void divide(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
        const shape &asp = a.get_shape();
        const shape &bsp = b.get_shape();
        if (!same_shape_backward(asp, bsp))
            throw new std::invalid_argument(
                    "tensor:" + a.name + " and " + "tensor:" + b.name + " don't have same backward shape.");

        assert(asp.ndims() >= bsp.ndims());
        des.reshape(asp);

        T *pa = a.data(), *pb = b.data(), *pd = des.data(), *pad = pb;
        PLONG asize = a.size(), bsize = b.size();

        for (PLONG i = 0; i < asize; ++i) {
            if (i % bsize == 0) pb = pad;
            *pd++ = (*pa++) / (*pb++);
        }
    }

    template<typename T>
    void conv(const tensor<T> &src, const tensor<T> &conv_kernel, tensor<T> &des, int stride) {
        assert(stride >= 1);
        const shape &ssp = src.get_shape();
        const shape &csp = conv_kernel.get_shape();
        assert(ssp[-1] == csp[-1]);
        return;
    }

    template<typename T>
    void pooling(const tensor<T> &src, tensor<T> &des, int stride) {
        return;
    }

    template<typename T>
    void tanh(const tensor<T> &src, tensor<T> &des) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");
        des.reshape(ssp);

        T *ps = src.data(), *pd = des.data();
        for (PLONG i = 0; i < ssp.size; ++i) {
            *pd++ = 1 - 2 / (1 + std::exp(2 * (*ps++)));
        }
    }

    template<typename T>
    void sigmoid(const tensor<T> &src, tensor<T> &des) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");
        des.reshape(ssp);

        T *ps = src.data(), *pd = des.data();
        for (PLONG i = 0; i < ssp.size; ++i) {
            *pd++ = 1 / (1 + std::exp(-(*ps++)));
        }
    }

    template<typename T>
    void relu(const tensor<T> &src, tensor<T> &des) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");
        des.reshape(ssp);

        T *ps = src.data(), *pd = des.data();
        for (PLONG i = 0; i < ssp.size; ++i) {
            if (*ps <= 0) *pd = 0;
            else *pd = *ps;
            ++ps;
            ++pd;
        }
    }

    void softmax(double *src, int len, double *des) {
        double *s = src, *d = des;
        double mx = -MAXFLOAT;
        for (int i = 0; i < len; ++i) {
            mx = max(*s, mx);
            ++s;
        }

        s = src;
        double exp_sum = 0.0;
        for (int i = 0; i < len; ++i) {
            *d = *s - mx;
            *d = std::exp(*d);
            exp_sum += *d;
            ++s;
            ++d;
        }

        d = des;
        for (int i = 0; i < len; ++i) {
            *d /= exp_sum;
            ++d;
        }
    }

    void softmax(real *src, int len, real *des) {
        real *s = src, *d = des;
        real mx = -MAXFLOAT;
        for (int i = 0; i < len; ++i) {
            mx = max(*s, mx);
            ++s;
        }

        s = src;
        real exp_sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            *d = *s - mx;
            *d = std::exp(*d);
            exp_sum += *d;
            ++s;
            ++d;
        }

        d = des;
        for (int i = 0; i < len; ++i) {
            *d /= exp_sum;
            ++d;
        }
    }

    template<typename T>
    void softmax(const tensor<T> &src, tensor<T> &des, int axis = 0) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");

        PLONG bulk_size = 0;
        if (axis == -1) {
            bulk_size = ssp.ndims() == 1 ? ssp.size : ssp.bulks[ssp.ndims() - 2];
            des.reshape(ssp);
        } else if (axis >= 0 && axis < ssp.ndims()) {
            bulk_size = axis == 0 ? ssp.size : ssp.bulks[axis - 1];
            if (axis >= 0) {
                vector<int> dsp(ssp.dims.begin(), ssp.dims.begin() + axis);
                dsp.push_back(ssp[axis] * ssp.bulks[axis]);
                des.reshape(dsp);
            }
        } else throw new invalid_argument("axis should be >= 0 or == -1");

        T *ps = src.data(), *pd = des.data();
        PLONG nbulk = ssp.size / bulk_size;
        assert(ssp.size % bulk_size == 0);
        for (PLONG i = 0; i < nbulk; ++i) {
            softmax(ps, bulk_size, pd);
            ps += bulk_size;
            pd += bulk_size;
        }
    }

    template<typename T>
    void mean(const tensor<T> &src, tensor<T> &des, int axis = -2, bool keepdims = false) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");
    }

    template<typename T>
    void sum(const tensor<T> &src, tensor<T> &des, int axis = -2, bool keepdims = false) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");
    }

    template<typename T>
    void max(const tensor<T> &src, tensor<T> &des, int axis = 0, bool keepdims = false) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");

        PLONG bulk_size = 0;
        if (axis == -1) {
            bulk_size = ssp.ndims() == 1 ? ssp.size : ssp.bulks[ssp.ndims() - 2];
            if (ssp.ndims() == 1) {
                if (keepdims) des.reshape({1, 1});
                else des.reshape({1});
            } else {
                vector<int> dsp(ssp.dims.begin(), ssp.dims.end() - 1);
                if (keepdims) dsp.push_back(1);
                des.reshape(dsp);
            }
        } else if (axis >= 0 && axis < ssp.ndims()) {
            bulk_size = axis == 0 ? ssp.size : ssp.bulks[axis - 1];
            if (axis == 0) {
                if (keepdims) des.reshape({1, 1});
                else des.reshape({1});
            } else {
                vector<int> dsp(ssp.dims.begin(), ssp.dims.begin() + axis);
                if (keepdims) dsp.push_back(1);
                des.reshape(dsp);
            }
        } else throw new invalid_argument("axis should be >= 0 or == -1");

        T *ps = src.data(), *pd = des.data();
        PLONG nbulk = ssp.size / bulk_size;
        assert(ssp.size % bulk_size == 0);
        for (PLONG i = 0; i < nbulk; ++i) {
            real mx = -MAXFLOAT;
            for (PLONG j = 0; j < bulk_size; ++j) {
                if (mx < *ps) {
                    mx = *ps;
                }
                ++ps;
            }
            *pd++ = mx;
        }
    }

    template<typename T>
    void min(const tensor<T> &src, tensor<T> &des, int axis = 0, bool keepdims = false) {
        const shape &ssp = src.get_shape();
        if (ssp.empty())
            throw new invalid_argument("tensor:" + src.name + " is empty.");

        PLONG bulk_size = 0;
        if (axis == -1) {
            bulk_size = ssp.ndims() == 1 ? ssp.size : ssp.bulks[ssp.ndims() - 2];
            if (ssp.ndims() == 1) {
                if (keepdims) des.reshape({1, 1});
                else des.reshape({1});
            } else {
                vector<int> dsp(ssp.dims.begin(), ssp.dims.end() - 1);
                if (keepdims) dsp.push_back(1);
                des.reshape(dsp);
            }
        } else if (axis >= 0 && axis < ssp.ndims()) {
            bulk_size = axis == 0 ? ssp.size : ssp.bulks[axis - 1];
            if (axis == 0) {
                if (keepdims) des.reshape({1, 1});
                else des.reshape({1});
            } else {
                vector<int> dsp(ssp.dims.begin(), ssp.dims.begin() + axis);
                if (keepdims) dsp.push_back(1);
                des.reshape(dsp);
            }
        } else throw new invalid_argument("axis should be >= 0 or == -1");

        T *ps = src.data(), *pd = des.data();
        PLONG nbulk = ssp.size / bulk_size;
        assert(ssp.size % bulk_size == 0);
        for (PLONG i = 0; i < nbulk; ++i) {
            real mn = MAXFLOAT;
            for (PLONG j = 0; j < bulk_size; ++j) {
                if (mn > *ps) {
                    mn = *ps;
                }
                ++ps;
            }
            *pd++ = mn;
        }
    }

    /**
     * Convert to one hot label tensor.
     * @tparam T: The generic type.
     * @param src: The original label tensor.
     * @param des: The one hot label tensor.
     * @param num_classes: the total num of different classes.
     * @example1:
     *     src: [[0, 1, 2, 3]]; shape=(4)
     *     num_classes: 4
     *     des: [[1, 0, 0, 0],
     *           [0, 1, 0, 0],
     *           [0, 0, 1, 0],
     *           [0, 0, 0, 1]]; shape=(4, 4)
     *
     * @example2:
     *     src: [[0, 1],
     *           [2, 3]]; shape=(2, 2)
     *     num_classes: 4
     *     des: [[[1, 0, 0, 0],
     *            [0, 1, 0, 0]],
     *           [[0, 0, 1, 0],
     *            [0, 0, 0, 1]]]; shape=(2, 2, 4)
     *
     * @example3:
     *     src: [[[0, 1],
     *            [2, 3]],
     *           [[1, 0],
     *            [3, 1]]]; shape=(2, 2, 2)
     *     num_classes: 4
     *     des: [[[[1, 0, 0, 0],
     *             [0, 1, 0, 0]],
     *            [[0, 0, 1, 0],
     *             [0, 0, 0, 1]]],
     *           [[[0, 1, 0, 0],
     *             [1, 0, 0, 0]],
     *            [[0, 0, 0, 1],
     *             [0, 1, 0, 0]]]]; shape=(2, 2, 2, 4)
     */
    template<typename T>
    void one_hot(const tensor<T> &src, const tensor<T> &des, int num_classes) {
        return;
    }

    /**
     * 2-D matrix multiplication.
     * @param a: The left matrix.
     * @param b: The right matrix.
     * @param a_transpose: Indicates whether the first matrix to be transposed.
     * @param b_transpose: Indicates whether the second matrix to be transposed.
     * @return The multiplication result.
     * @example1:
     *     a: [[1, 2, 3],
     *         [4, 5, 6]]
     *     b: [[3, 2],
     *         [1, 2],
     *         [4, 6]]
     *     a_transpose: false
     *     b_transpose: false
     *     return: [[17, 24],
     *              [41, 54]]
     * @example2:
     *    a: [[1, 4],
     *        [2, 5],
     *        [3, 6]]
     *    b: [[3, 2],
     *        [1, 2],
     *        [4, 6]]
     *    a_transpose: true
     *    b_transpose: false
     *    return: [[17, 24],
     *             [41, 54]]
     * @example3:
     *    a: [[1, 2, 3],
     *        [4, 5, 6]]
     *    b: [[3, 1, 4],
     *        [2, 2, 6]]
     *    a_transpose: false
     *    b_transpose: true
     *    return: [[17, 24],
     *             [41, 54]]
     * @example4:
     *    a: [[1, 4],
     *        [2, 5],
     *        [3, 6]]
     *    b: [[3, 1, 4],
     *        [2, 2, 6]]
     *    a_transpose: true
     *    b_transpose: true
     *    return: [[17, 24],
     *             [41, 54]]
     */
    template<typename T>
    void dot_mul(tensor<T> &a, tensor<T> &b, tensor<T> &des, bool a_transpose, bool b_transpose) {
        return;
    }

//    /**
//     * 2-dim tensor transpose op.
//     * @param a: origin tensor.
//     * @param des: output tensor.
//     */
//    template <typename T>
//    void transpose(const tensor<T> &a, tensor<T> &des) {
//        if (a.empty()) return;
//        assert(a.get_shape().ndims() == 2);
//
//        LONG col = a.get_shape()[1];
//        LONG row = a.get_shape()[0];
//        int dim[2] = {a.get_shape()[1], a.get_shape()[0]};
//        shape s(dim, 2);
//        des.reshape(s);
//
//        T *__pd = des.data();
//        T *__pa = a.data();
//        int i = 0, j = 0, k = 0;
//        for (i = 0; i < col; ++i){
//            k = 0;
//            for (j = 0; j < row; ++j) {
//                *__pd++ = (*(__pa + k));
//                k += col;
//            }
//            ++__pa;
//        }
//    }

//    /**
//     *
//     * @tparam T
//     * @param A
//     * @return
//     */
//    template <typename T>
//    tensor<T> transpose(const tensor<T> &A){
//        tensor<T> res;
//        transpose(A, res);
//        return res;
//    }
//
//    /**
//     * dot multiply -> a * transpose(b) = des.
//     * @param a
//     * @param b
//     * @param des
//     */
//    void __dot_mul_transpose(tensor<real> &a,
//                             tensor<real> &b,
//                             tensor<real> &des) {
//        real *__pb = b.data();
//        real *__pa = a.data();
//        real *__pd = des.data();
//        real __value = 0.0;
//        int col = a.get_shape()[1];
//        LONG i = 0, size = (LONG) des.size() * (LONG) a.get_shape()[1];
//        for (i = 0; i < size; ++i) {
//            __value += (*__pa++) * (*__pb++);
//            if ((i + 1) % col == 0) {
//                *__pd++ = __value;
//                __value = 0;
//                if ((i + 1) % b.size() == 0)
//                    __pb = b.data();
//                else __pa -= col;
//            }
//        }
//    }
//
//    /**
//     * transpose(a) * b = des.
//     * @param a
//     * @param b
//     * @param des
//     */
//    void __transpose_dot_mul(tensor<real> &a,
//                             tensor<real> &b,
//                             tensor<real> &des){
//
//        int __a_row = a.get_shape()[0];
//        int __a_col = a.get_shape()[1];
//        int __b_col = b.get_shape()[1];
//
//        int t[2] = {__a_col, __b_col};
//        shape __des_shape(t, 2);
//        des.reshape(__des_shape);
//
//        real *__pb = b.data();
//        real *__pa = a.data();
//        real *__pd = des.data();
//        real __value;
//
//        real **__pa_aux = new real*[__a_row];
//        real **__pb_aux = new real*[__a_row];
//        for (int i = 0; i < __a_row; ++i){
//            __pa_aux[i] = __pa + i * __a_col;
//            __pb_aux[i] = __pb + i * __b_col;
//        }
//
//        int i, j, k, l;
//        for (i = 0; i < __a_col; ++i){
//            for (j = 0; j < __b_col; ++j){
//                __value = 0.0;
//                for (k = 0; k < __a_row; ++k){
//                    __value += (*(__pa_aux[k])) * (*(__pb_aux[k] + j));
//                }
//                *__pd++ = __value;
//            }
//            for (l = 0; l < __a_row; ++l)
//                __pa_aux[l]++;
//        }
//
//        if (0 != __pa_aux) delete [] __pa_aux;
//        if (0 != __pb_aux) delete [] __pb_aux;
//    }
//
//    /**
//     * transpose(a) * transpose(b) = des.
//     * @param a
//     * @param b
//     * @param des
//     */
//    void __transpose_dot_mul_transpose(tensor<real> &a,
//                                       tensor<real> &b,
//                                       tensor<real> &des){
//        int __a_row = a.get_shape()[0];
//        int __a_col = a.get_shape()[1];
//        int __b_row = b.get_shape()[0];
//
//        int t[2] = {__a_col, __b_row};
//        shape __des_shape(t, 2);
//        des.reshape(__des_shape);
//
//        real *__pb = b.data();
//        real *__pa = a.data();
//        real *__pd = des.data();
//        real __value;
//
//
//        real **__pa_aux = new real*[__a_row];
//        for (int i = 0; i < __a_row; ++i){
//            __pa_aux[i] = __pa + i * __a_col;
//        }
//
//        int i, j, k, l;
//        for (i = 0; i < __a_col; ++i){
//            for (j = 0; j < __b_row; ++j){
//                __value = 0.0;
//                for (k = 0; k < __a_row; ++k){
//                    __value += (*(__pa_aux[k])) * (*__pb++);
//                }
//                *__pd++ = __value;
//            }
//            __pb = b.data();
//            for (l = 0; l < __a_row; ++l)
//                __pa_aux[l]++;
//        }
//
//        if (0 != __pa_aux) delete [] __pa_aux;
//    }
//
//    /**
//     * dot multiply -> a * b = des.
//     * @param a
//     * @param b
//     * @param des
//     */
//    void __dot_mul(tensor<real> &a,
//                   tensor<real> &b,
//                   tensor<real> &des) {
//        real *__pb = b.data();
//        real *__pa = a.data();
//        real *__pd = des.data();
//        real __value;
//        int __col = b.get_shape()[1];
//        int __row = des.get_shape()[0];
//        int __medium = a.get_shape()[1];
//
//        int i, j, k;
//        for (i = 0; i < __row; ++i) {
//            for (j = 0; j < __col; ++j) {
//                __value = 0.0;
//                for (k = 0; k < __medium; ++k) {
//                    __value += (*__pa++) * (*(__pb + k * __col + j));
//                }
//                (*__pd++) = __value;
//                if (j != __col - 1) __pa -= __medium;
//            }
//        }
//    }
//
//    /**
//     * a[ids, :] * transpose(b) = des.
//     * @param a
//     * @param b
//     * @param des
//     * @param ids
//     */
//    void __dot_mul_with_selected_ids_transpose(tensor<real> &a,
//                                               tensor<real> &b,
//                                               tensor<real> &des,
//                                               tensor<real> &ids) {
//        assert(ids.get_shape()[1] == 1);
//        real *__pids = ids.data();
//        real *__pa = 0;
//        real *__pb = b.data();
//        real *__pd = des.data();
//        real __value;
//
//        size_t __ids_size = ids.size();
//        int __a_col = a.get_shape()[1];
//        int __b_row = b.get_shape()[0];
//        int __b_col = b.get_shape()[1];
//        int i, j, k;
//        for (i = 0; i < __ids_size; ++i) {
//            __pa = a.data() + (LONG) (*__pids++) * __a_col;
//            for (j = 0; j < __b_row; ++j) {
//                __value = 0.0;
//                for (k = 0; k < __b_col; ++k) {
//                    __value += (*__pa++) * (*__pb++);
//                }
//                (*__pd++) = __value;
//                if (j != __b_row - 1) __pa -= __a_col;
//            }
//            __pb = b.data();
//        }
//    }
//
//    /**
//     * a[ids, :] * b = des.
//     * @param a
//     * @param b
//     * @param des
//     * @param ids
//     */
//    void __dot_mul_with_selected_ids(tensor<real> &a,
//                                     tensor<real> &b,
//                                     tensor<real> &des,
//                                     tensor<real> &ids) {
//        assert(ids.get_shape()[1] == 1);
//        real *__pids = ids.data();
//        real *__pa = 0;
//        real *__pb = b.data();
//        real *__pd = des.data();
//        real __value;
//
//        size_t __ids_size = ids.size();
//        int __a_col = a.get_shape()[1];
//        int __b_col = b.get_shape()[1];
//        int __b_row = b.get_shape()[0];
//        int i, j, k;
//        for (i = 0; i < __ids_size; ++i) {
//            __pa = a.data() + (LONG) (*__pids++) * __a_col;
//            for (j = 0; j < __b_col; ++j) {
//                __value = 0.0;
//                for (k = 0; k < __b_row; ++k) {
//                    __value += (*__pa++) * (*(__pb + k * __b_col + j));
//                }
//                (*__pd++) = __value;
//                if (j != __b_col - 1) __pa -= __a_col;
//            }
//        }
//    }
//
//    /**
//     * (a || a[ids, :]) * (b || transpose(b)) = des.
//     * @param a
//     * @param b
//     * @param des
//     * @param b_transpose
//     * @param a_rows
//     */
//    void dot_mul(tensor<real> &a,
//                 tensor<real> &b,
//                 tensor<real> &des,
//                 bool b_transpose,
//                 tensor<real> &a_rows = DEFAULT_FLOAT_TENSOR) {
//        if (a.empty() || b.empty()) return;
//
//        if (b_transpose) assert(a.get_shape()[1] == b.get_shape()[1]);
//        else assert(a.get_shape()[1] == b.get_shape()[0]);
//
//        if (b_transpose) {
//            if (a_rows.empty()) {
//                int t[2] = {a.get_shape()[0], b.get_shape()[0]};
//                shape s(t, 2);
//                des.reshape(s);
//            } else {
//                int t[2] = {a_rows.get_shape()[0], b.get_shape()[0]};
//                shape s(t, 2);
//                des.reshape(s);
//            }
//        } else {
//            if (a_rows.empty()) {
//                int t[2] = {a.get_shape()[0], b.get_shape()[1]};
//                shape s(t, 2);
//                des.reshape(s);
//            } else {
//                int t[2] = {a_rows.get_shape()[0], b.get_shape()[1]};
//                shape s(t, 2);
//                des.reshape(s);
//            }
//        }
//
//        if (a_rows.empty()) {
//            if (b_transpose) __dot_mul_transpose(a, b, des);
//            else __dot_mul(a, b, des);
//        } else {
//            if (b_transpose) __dot_mul_with_selected_ids_transpose(a, b, des, a_rows);
//            else __dot_mul_with_selected_ids(a, b, des, a_rows);
//        }
//    }
//
//    /**
//     * (a || transpose(a)) * (b || transpose(b)) = des.
//     * @param a
//     * @param b
//     * @param des
//     * @param a_transpose
//     * @param b_transpose
//     */
//    void dot_mul(tensor<real> &a,
//                 tensor<real> &b,
//                 tensor<real> &des,
//                 bool a_transpose,
//                 bool b_transpose){
//        if (a.empty() || b.empty()) return;
//
//        if (!a_transpose && b_transpose){
//            dot_mul(a, b, des, true);
//        } else if (!a_transpose && !b_transpose){
//            dot_mul(a, b, des, false);
//        } else if (a_transpose && !b_transpose){
//            assert(a.get_shape()[0] == b.get_shape()[0]);
//            __transpose_dot_mul(a, b, des);
//        } else {
//            assert(a.get_shape()[0] == b.get_shape()[1]);
//            __transpose_dot_mul_transpose(a, b, des);
//        }
//    }
//
//    /**
//     * look up embeddings[ids, :].
//     * @tparam T
//     * @param embeddings: 2-dim tensor.
//     * @param ids: the row ids of embeddings to look up. The data type is tensor.
//     * @param lookups: look up result.
//     */
//    template<typename T>
//    void embeddings_lookup(const tensor<T> &embeddings, const tensor<LONG> &ids, tensor<T> &lookups) {
//        if (embeddings.empty()) return;
//        assert(embeddings.get_shape().dims() == 2);
//
//        int dim = ids.get_shape().dims() + 1;
//        int *temp = new int[dim];
//        for (int i = 0; i < dim - 1; ++i)
//            temp[i] = ids.get_shape()[i];
//        temp[dim - 1] = embeddings.get_shape()[1];
//        shape s(temp, dim);
//        lookups.reshape(s);
//
//        LONG *pid = ids.data();
//        LONG embedding_col = embeddings.get_shape()[1];
//        T *pe = 0;
//        T *pl = lookups.data();
//        for (LONG i = 0; i < lookups.size(); ++i) {
//            if (i % embedding_col == 0) {
//                pe = embeddings.data() + (*pid++) * embedding_col;
//            }
//            (*pl++) = (*pe++);
//        }
//
//        delete [] temp;
//    }
//
//    /**
//    * look up embeddings[ids, :].
//    * @tparam T
//    * @param embeddings: 2-dim tensor.
//    * @param ids: the row ids of embeddings to look up. The data type is tensor.
//    * @param lookups: look up result.
//    */
//    template<typename T>
//    void embeddings_lookup(const tensor<T> &embeddings, const vector<LONG> &ids, tensor<T> &lookups) {
//        if (embeddings.empty()) return;
//        assert(embeddings.get_shape().dims() == 2);
//
//        lookups.reshape({(int)ids.size(), embeddings.get_shape()[1]});
//        int embedding_col = embeddings.get_shape()[1];
//        LONG idx = 0;
//        T *pe = 0;
//        T *pl = lookups.data();
//        for (LONG i = 0; i < lookups.get_shape()[0]; ++i) {
//            pe = embeddings.data() + (ids[idx++]) * embedding_col;
//            for (LONG j = 0; j < lookups.get_shape()[1]; ++j) {
//                (*pl++) = (*pe++);
//            }
//        }
//    }
//
//
//    /**
//     * reduce_mean.
//     * @tparam T
//     * @param src
//     * @param des
//     * @param axis
//     */
//    template<typename T>
//    void reduce_mean(const tensor<T> &src, tensor<T> &des, int axis) {
//        if (src.empty()) return;
//
//        int __src_dims = src.get_shape().dims();
//        if (1 == __src_dims) return;
//        assert(axis < __src_dims);
//        int *__des_dims = new int[__src_dims - 1];
//        int __idx = 0;
//        LONG __num_bulk = 1;
//        LONG __sub_bulk_size = 1;
//        LONG __axis_dim = src.get_shape()[axis];
//        for (int i = 0; i < __src_dims; ++i) {
//            if (i < axis) __num_bulk *= src.get_shape()[i];
//            if (i > axis) __sub_bulk_size *= src.get_shape()[i];
//            if (i == axis) continue;
//            __des_dims[__idx++] = src.get_shape()[i];
//        }
//        LONG __bulk_size = __axis_dim * __sub_bulk_size;
//        shape __des_shape(__des_dims, __idx);
//        des.reshape(__des_shape);
//
//        T *__ps;
//        T *__pd;
//        T sum = 0;
//        LONG i, k, j;
//        for (i = 0; i < __num_bulk; ++i) {
//            __ps = src.data() + i * __bulk_size;
//            __pd = des.data() + i * __sub_bulk_size;
//            for (k = 0; k < __sub_bulk_size; ++k) {
//                sum = 0;
//                for (j = 0; j < __axis_dim; ++j) {
//                    sum += *(__ps + j * __sub_bulk_size);
//                }
//                (*__pd++) = sum / __axis_dim;
//                ++__ps;
//            }
//        }
//
//        delete[] __des_dims;
//    }
//
//    /**
//     * reduce_sum.
//     * @tparam T
//     * @param src
//     * @param des
//     * @param axis
//     */
//    template<typename T>
//    void reduce_sum(const tensor<T> &src, tensor<T> &des, int axis) {
//        if (src.empty()) return;
//
//        int __src_dims = src.get_shape().dims();
//        if (1 == __src_dims) return;
//        assert(axis < __src_dims);
//        int *__des_dims = new int[__src_dims - 1];
//        int __idx = 0;
//        LONG __num_bulk = 1;
//        LONG __sub_bulk_size = 1;
//        LONG __axis_dim = src.get_shape()[axis];
//        for (int i = 0; i < __src_dims; ++i) {
//            if (i < axis) __num_bulk *= src.get_shape()[i];
//            if (i > axis) __sub_bulk_size *= src.get_shape()[i];
//            if (i == axis) continue;
//            __des_dims[__idx++] = src.get_shape()[i];
//        }
//        LONG __bulk_size = __axis_dim * __sub_bulk_size;
//        shape __des_shape(__des_dims, __idx);
//        des.reshape(__des_shape);
//
//        T *__ps;
//        T *__pd;
//        T sum = 0;
//        LONG i, k, j;
//        for (i = 0; i < __num_bulk; ++i) {
//            __ps = src.data() + i * __bulk_size;
//            __pd = des.data() + i * __sub_bulk_size;
//            for (k = 0; k < __sub_bulk_size; ++k) {
//                sum = 0;
//                for (j = 0; j < __axis_dim; ++j) {
//                    sum += *(__ps + j * __sub_bulk_size);
//                }
//                (*__pd++) = sum;
//                ++__ps;
//            }
//        }
//        delete[] __des_dims;
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void add_suffix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_suffix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bsize = b.size();
//        size_t __num_bulk = a.size() / __bsize;
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bsize; ++j)
//                (*__pd++) = (*__pa++) + (*__pb++);
//            __pb = b.data();
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template <typename T>
//    void add_prefix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des){
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_prefix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bulk_size = a.size() / b.size();
//        size_t __num_bulk = b.size();
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bulk_size; ++j)
//                (*__pd++) = (*__pa++) + (*__pb);
//            ++__pb;
//        }
//    }
//
//    /**
//     * a + b = des.
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void add(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        assert(__a_shape == __b_shape);
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t size = b.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) + (*__pb++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void add(const tensor<T> &a, T b, tensor<T> &des) {
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        LONG size = a.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) + b;
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param b
//     * @param a
//     * @param des
//     */
//    template<typename T>
//    void add(T b, const tensor<T> &a, tensor<T> &des) {
//        add(a, b, des);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void subtract_suffix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_suffix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bsize = b.size();
//        size_t __num_bulk = a.size() / __bsize;
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bsize; ++j)
//                (*__pd++) = (*__pa++) - (*__pb++);
//            __pb = b.data();
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template <typename T>
//    void subtract_prefix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des){
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_prefix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bulk_size = a.size() / b.size();
//        size_t __num_bulk = b.size();
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bulk_size; ++j)
//                (*__pd++) = (*__pa++) - (*__pb);
//            ++__pb;
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void subtract(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        assert(__a_shape == __b_shape);
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        LONG size = b.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) - (*__pb++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void subtract(const tensor<T> &a, T b, tensor<T> &des) {
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        LONG size = a.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) - b;
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param b
//     * @param a
//     * @param des
//     */
//    template<typename T>
//    void subtract(T b, const tensor<T> &a, tensor<T> &des) {
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        LONG size = a.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = b - (*__pa++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void multiply_suffix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_suffix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bsize = b.size();
//        size_t __num_bulk = a.size() / __bsize;
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bsize; ++j)
//                (*__pd++) = (*__pa++) * (*__pb++);
//            __pb = b.data();
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template <typename T>
//    void multiply_prefix(const tensor<T> &a, const tensor<T> &b, tensor<T> &des){
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_prefix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bulk_size = a.size() / b.size();
//        size_t __num_bulk = b.size();
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bulk_size; ++j)
//                (*__pd++) = (*__pa++) * (*__pb);
//            ++__pb;
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void multiply(const tensor<T> &a, const tensor<T> &b, tensor<T> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        assert(__a_shape == __b_shape);
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        LONG size = b.size();
//
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) * (*__pb++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void multiply(const tensor<T> &a, T b, tensor<T> &des) {
//        if (a.empty()) return;
//
//        des.reshape(a.get_shape());
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        size_t size = a.size();
//
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) * b;
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param b
//     * @param a
//     * @param des
//     */
//    template<typename T>
//    void multiply(T b, const tensor<T> &a,  tensor<T> &des) {
//        multiply(a, b, des);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void divide_suffix(const tensor<T> &a, const tensor<T> &b, tensor<real> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_suffix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bsize = b.size();
//        size_t __num_bulk = a.size() / __bsize;
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bsize; ++j)
//                (*__pd++) = (*__pa++) / (*__pb++);
//            __pb = b.data();
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template <typename T>
//    void divide_prefix(const tensor<T> &a, const tensor<T> &b, tensor<real> &des){
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        shape __des_shape = des.get_shape();
//        assert(shape::same_prefix_shape(__a_shape, __b_shape));
//        if (des.empty() || (__des_shape != __a_shape)) {
//            des.reshape(__a_shape);
//        }
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t __bulk_size = a.size() / b.size();
//        size_t __num_bulk = b.size();
//        LONG i, j;
//        for (i = 0; i < __num_bulk; ++i){
//            for (j = 0; j < __bulk_size; ++j)
//                (*__pd++) = (*__pa++) / (*__pb);
//            ++__pb;
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void divide(const tensor<T> &a, const tensor<T> &b, tensor<real> &des) {
//        if (a.empty() || b.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        shape __b_shape = b.get_shape();
//        assert(__a_shape == __b_shape);
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pb = b.data();
//        T *__pd = des.data();
//        size_t size = b.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) / (*__pb++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param b
//     * @param des
//     */
//    template<typename T>
//    void divide(const tensor<T> &a, T b, tensor<real> &des) {
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        size_t size = a.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = (*__pa++) / b;
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param b
//     * @param a
//     * @param des
//     */
//    template<typename T>
//    void divide(T b, const tensor<T> &a, tensor<real> &des) {
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        size_t size = a.size();
//        for (LONG i = 0; i < size; ++i)
//            *__pd++ = b / (*__pa++);
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param des
//     * @param def
//     */
//    template<typename T>
//    void cut_off(const tensor<T> &a, tensor<real> &des, T def=1000) {
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//
//        T *__pa = a.data();
//
//        real __norm2 = 0.0;
//        size_t size = a.size();
//        for (LONG i = 0; i < size; ++i){
//            __norm2 += (*__pa) * (*__pa);
//            ++__pa;
//        }
//
//        if (__norm2 <= a.size() * def) return;
//
//        __pa = a.data();
//        real *__pd = des.data();
//        __norm2 = sqrt(__norm2);
//        for (LONG i = 0; i < size; ++i){
//            *__pd++ = def * (*__pa++) / __norm2;
//        }
//    }
//
//    /**
//     * element-wise sigmoid function.
//     * @tparam T
//     * @param a
//     * @param des
//     */
//    template <typename T>
//    void sigmoid(const tensor<T> &a, tensor<real> &des){
//        shape __a_shape = a.get_shape();
//        shape __des_shape = des.get_shape();
//        if (__a_shape != __des_shape) des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        real *__pd = des.data();
//        size_t size = a.size();
//        for (LONG i = 0; i < size; ++i){
//            *__pd++ = 1 / (1 + std::exp(-(*__pa++)));
//        }
//    }
//
//    /**
//    * element-wise sigmoid function.
//    * @tparam T
//    * @param a
//    * @param des
//    */
//    template <typename T>
//    void tanh(const tensor<T> &a, tensor<real> &des){
//        shape __a_shape = a.get_shape();
//        shape __des_shape = des.get_shape();
//        if (__a_shape != __des_shape) des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        real *__pd = des.data();
//        for (LONG i = 0; i < a.size(); ++i){
//            *__pd++ = 1 - 2 / (1 + std::exp(2*(*__pa++)));
//        }
//    }
//
//    /**
//     * element-wise exp function.
//     * @tparam T
//     * @param a
//     * @param des
//     */
//    template <typename T>
//    void exp(const tensor<T> &a, tensor<T> &des){
//        shape __a_shape = a.get_shape();
//        shape __des_shape = des.get_shape();
//        if (__a_shape != __des_shape) des.reshape(__a_shape);
//
//        T *__pa = a.data();
//        T *__pd = des.data();
//        for (LONG i = 0; i < a.size(); ++i){
//            *__pd++ = std::exp((*__pa++));
//        }
//    }
//
//    /**
//     *
//     * @tparam T1
//     * @tparam T2
//     * @param a
//     * @param des
//     */
//    template <typename T1, typename T2>
//    void sqrt0(const tensor<T1> &a, tensor<T2> &des){
//        des.reshape(a.get_shape());
//
//        T1 *__pa = a.data();
//        T2 *__pd = des.data();
//
//        size_t __size = a.size();
//        for (LONG i = 0; i < __size; ++i){
//            *__pd++ = sqrt(*__pa++);
//        }
//    }
//
//    /**
//     *
//     * @tparam T1
//     * @tparam T2
//     * @tparam T3
//     * @param a
//     * @param p
//     * @param des
//     */
//    template <typename T1, typename T2, typename T3>
//    void pow0(const tensor<T1> &a, T2 p, tensor<T3> &des){
//        des.reshape(a.get_shape());
//
//        T1 *__pa = a.data();
//        T3 *__pd = des.data();
//
//        size_t __size = a.size();
//        for (LONG i = 0; i < __size; ++i){
//            *__pd++ = pow(*__pa++, p);
//        }
//    }
//
//    template <typename T>
//    void subtract_max(const tensor<T> &a, tensor<T> &des){
//        if (a.empty()) return;
//        assert(a.get_shape().dims() == 2);
//        int __row = a.get_shape()[0];
//        int __col = a.get_shape()[1];
//
//        des.reshape(a.get_shape());
//
//        T *__pa;
//        T *__pd;
//        T __max;
//        for (int i = 0; i < __row; ++i){
//            __pa = a.data() + i * __col;
//            for (int j = 0; j < __col; ++j){
//                if (j == 0 || __max < (*__pa))
//                    __max = *__pa;
//                ++__pa;
//            }
//            __pd = des.data() + i * __col;
//            __pa = a.data() + i * __col;
//            for (int j = 0; j < __col; ++j){
//                (*__pd++) = (*__pa++) - __max;
//            }
//        }
//    }
//
//    /**
//     * a[:, softmax(: - max(:))] = des
//     * @tparam T
//     * @param a
//     * @param des
//     */
//    template <typename T>
//    void softmax(const tensor<T> &a, tensor<real> &des){
//        shape __a_shape = a.get_shape();
//        des.reshape(__a_shape);
//        subtract_max(a, des);
//        exp(des, des);
//
//        tensor<real> __des_sum;
//        reduce_sum(des, __des_sum, 1);
//        divide_prefix(des, __des_sum, des);
//    }
//
//    /**
//     * element-wise ln function.
//     * @tparam T
//     * @param a
//     * @param des
//     */
//    template <typename T>
//    void ln(const tensor<T> &a, tensor<real> &des){
//        if (a.empty()) return;
//
//        des.reshape(a.get_shape());
//
//        real *__pd = des.data();
//        T *__pa = a.data();
//        size_t __a_size = a.size();
//        for (LONG i = 0; i < __a_size; ++i){
//            *__pd++ = log(*__pa++);
//        }
//    }
//
//    /**
//     *
//     * @tparam T
//     * @param a
//     * @param sub
//     * @param des
//     */
//    template <typename T>
//    void log_softmax(tensor<T> &a, tensor<T> &sub, tensor<real> &des){
//        subtract_max(a, sub);
//        exp(sub, des);
//        tensor<real> __des_sum;
//        reduce_sum(des, __des_sum, 1);
//
//        ln(__des_sum, __des_sum);
//        subtract_prefix(sub, __des_sum, des);
//    }
//
//    /**
//     * multi-bi-classes.
//     * @param weights
//     * @param bias
//     * @param inputs
//     * @param num_samples
//     */
//    void sigmoid_cross_entropy_with_logits(
//            tensor<real>& weights,
//            tensor<real>& bias,
//            tensor<real>& inputs,
//            tensor<real>& labels,
//            tensor<real>& samples=DEFAULT_FLOAT_TENSOR){
//        assert(labels.get_shape()[1] == 1);
//        assert(inputs.get_shape()[0] == labels.get_shape()[0]);
//        int __batch_size = inputs.get_shape()[0];
//        int __num_cols = weights.get_shape()[1];
//
//        real *__pw = weights.data();
//        real *__pb = bias.data();
//        real *__pi = inputs.data();
//        real *__pl = labels.data();
//        real *__ps = samples.data();
//        for (int i = 0; i < __batch_size; ++i){
//            ;
//        }
//    }
//
//    real softmax_cross_entropy_with_logits(tensor<real> &predict, tensor<real> &label){
//        tensor<real> __class_prob;
//        tensor<real> __sub_max;
//        log_softmax(predict, __sub_max, __class_prob);
//
//        real __soft_cross_entropy = 0.0;
//        real *__pl = label.data();
//        real *__pc = __class_prob.data();
//        for (LONG i = 0; i < label.size(); ++i){
//            if (*__pl++ != 0.0) __soft_cross_entropy -= *__pc++;
//        }
//
//        real loss = __soft_cross_entropy / label.get_shape()[0];
//        return loss;
//    }
//
//    /**
//     *
//     * @param y
//     * @param one_hot_label
//     * @param num_classes
//     */
//    void convert_to_one_hot(tensor<real>& y, tensor<real>& one_hot_label, int num_classes){
//        shape __y_shape = y.get_shape();
//        int __label_dims = __y_shape.dims() + 1;
//        int *a = new int[__label_dims];
//        for (int i = 0; i < __label_dims - 1; ++i)
//            a[i] = __y_shape[i];
//        a[__label_dims-1] = num_classes;
//
//        shape __label_shape(a, __label_dims);
//        one_hot_label.reshape(__label_shape);
//
//        real *__py = y.data();
//        real *__pl = 0;
//        int i, j;
//        size_t __y_size = y.size();
//        for (i = 0; i < __y_size; ++i){
//            __pl = one_hot_label.data() + i * num_classes;
//
//            for (j = 0; j < num_classes; ++j){
//                if (j == (*__py))
//                    *__pl++ = 1.0;
//                else *__pl++ = 0.0;
//            }
//            __py++;
//        }
//
//        if (0 != a) delete [] a;
//    }
//
//    template<typename T>
//    void batch_norm(const tensor<T> &a, tensor<T> &mean, tensor<T> &var, tensor<T> &des){
//        if (a.empty()) return;
//
//        shape __a_shape = a.get_shape();
//        int row = __a_shape[0];
//        int col = __a_shape[1];
//        des.reshape(__a_shape);
//
//        mean.reshape({col});
//        var.reshape({col});
//
//        int i, j;
//
//        real u = 0, delta = 0;
//        T *__pa;
//        T *__pd;
//        T *__pm;
//        T *__pv;
//
//        real t;
//        for (i = 0; i < col; ++i){
//            __pa = a.data() + i;
//            u = 0;
//            delta = 0;
//            for (j = 0; j < row; ++j){
//                u += (*__pa);
//                delta += (*__pa) * (*__pa);
//                __pa += col;
//            }
//            u /= row;
//
//            t = delta / row - u * u;
//            delta = 1 / sqrt(abs(t) + 0.000001f);
//
//            __pa = a.data() + i;
//            __pd = des.data() + i;
//            for (j = 0; j < row; ++j){
//                *__pd = ((*__pa) - u) * delta;
//                __pa += col;
//                __pd += col;
//            }
//
//            __pm = mean.data() + i;
//            __pv = var.data() + i;
//            *__pm = u;
//            *__pv = delta;
//        }
//    }
//
//    /**
//     *
//     * @tparam T1
//     * @tparam T2
//     * @param a
//     * @param scale
//     * @param bias
//     * @param des
//     */
//    template <typename T1, typename T2>
//    void redirect(const tensor<T1> &a, const tensor<T2> &scale, const tensor<T2> &bias, tensor<T2> &des){
//        assert(!a.empty());
//        assert(!scale.empty());
//        assert(!bias.empty());
//
//        multiply_suffix(a, scale, des);
//        add_suffix(des, bias, des);
//    }
//
//    void argmax(tensor<real>& a, tensor<real>& des){
//        assert(a.get_shape().dims() == 2);
//        des.reshape({a.get_shape()[0]});
//        real *pa, *pd = des.data(), mx, mx_idx=0;
//
//        int row = a.get_shape()[0], col = a.get_shape()[1];
//        for (int i = 0; i < row; ++i){
//            pa = a.data() + i * col;
//            mx = *pa++;
//            mx_idx = 0;
//            for (int j = 1; j < col; ++j){
//                if (mx < *pa){
//                    mx = *pa;
//                    mx_idx = j;
//                }
//                ++pa;
//            }
//            *pd++ = mx_idx;
//        }
//    }
}

#endif //BEYOND_TENSOR_OPS_H

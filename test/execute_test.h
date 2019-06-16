//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_EXECUTE_TEST_H
#define BEYOND_EXECUTE_TEST_H

#include "../src/log.h"
#include "../src/shape.h"
#include "../src/tensor.h"
#include "../src/initializers.h"
#include "../src/variable.h"
#include "../src/utils.h"
#include "../src/tensor_ops.h"
using namespace tops;

void logger_test(){
    logger.info("Hello, world!");
    logger.debug("Hello, world!");
    logger.error("Hello, world!");
}

void shape_test(){
    vector<int> vec = {1, 2, 3};
    std::cout << "shape(const vector<int>& dims):" << std::endl;
    shape sp(vec);
    std::cout << sp.to_str() << std::endl;

    std::cout << "shape(const shape& copy):" << std::endl;
    shape sp_copy(sp);
    std::cout << sp_copy.to_str() << std::endl;

    std::cout << "bool operator==(const shape& s):" << std::endl;
    if (sp_copy == sp){
        std::cout << "sp_copy == sp" << std::endl;
    }

    std::cout << "shape():" << std::endl;
    shape sp_assigned;
    std::cout << sp_assigned.to_str() << std::endl;

    std::cout << "shape& operator=(const shape& s):" << std::endl;
    sp_assigned = sp;
    std::cout << sp_assigned.to_str() << std::endl;

    std::cout << "shape& operator=(vector<int> s):" << std::endl;
    sp_assigned = {1, 2, 3};
    std::cout << sp_assigned.to_str() << std::endl;

    sp_assigned = {4, 5, 6, 7};
    std::cout << sp_assigned.to_str() << std::endl;

    std::cout << "shape& operator!=(const shape& s):" << std::endl;
    if (sp_assigned != sp){
        std::cout << "sp_assigned != sp" << std::endl;
    }
}

void tensor_test(){
    shape sp({1, 2, 3});
    tensor<float> tensor1(sp);
    uniform_initializer(1.0f, 2.0f, tensor1);
    std::cout << tensor1.show() << std::endl;

    const shape &shape1 = tensor1.get_shape();
    std::cout << shape1.to_str() << std::endl;

    tensor1.reshape({4, 5, 6});
    std::cout << tensor1.to_string() << std::endl;
    std::cout << tensor1.show() << std::endl;

    tensor1.reshape(sp);
    std::cout << tensor1.to_string() << std::endl;
    std::cout << tensor1.show() << std::endl;

    tensor<float> tensor2({4, 5, 6});
    uniform_initializer(1.0f, 2.0f, tensor2);
    std::cout << tensor2.to_string() << std::endl;
    std::cout << tensor2.show() << std::endl;

    tensor<float> tensor3(tensor2);
    std::cout << tensor3.to_string() << std::endl;
    std::cout << tensor3.show() << std::endl;

    std::cout << tensor3[0].to_string() << std::endl;
    std::cout << tensor3[0].show() << std::endl;
}

void variable_test(){
    variable *variable1 = get_variable("X", {2, 3, 4}, 0.1f, 0.2f);
    std::cout << variable1->to_str() << std::endl;
    std::cout << variable1->get().show() << std::endl;
}

void tops_transpose_test(){
    shape sp({2, 3, 4, 6});
    tensor<float> tensor1(sp), tensor2;
    uniform_initializer(1.0f, 2.0f, tensor1);
    tops::transpose(tensor1, tensor2, {0, 2, 1, 3});
    std::cout << "original tensor info:"
              << tensor1.to_string() 
              << std::endl;
    std::cout << "transpose shape to {0, 2, 1, 3}" 
              << std::endl;
    std::cout << "original tensor:"
              << std::endl;
    std::cout << tensor1.show() << std::endl;
    std::cout << "dest tensor info:"
              << tensor2.to_string() 
              << std::endl;
    std::cout << "dest tensor:"
              << std::endl
              << tensor2.show() 
              << std::endl;

    
    shape sp1({2, 3});
    tensor<float> tensor3(sp1), tensor4;
    uniform_initializer(1.0f, 2.0f, tensor3);
    tops::transpose(tensor3, tensor4, {1, 0});
    std::cout << "original tensor info:"
              << tensor3.to_string() 
              << std::endl;
    std::cout << "transpose shape to {1, 0}" 
              << std::endl;
    std::cout << "original tensor:"
              << std::endl;
    std::cout << tensor3.show() << std::endl;
    std::cout << "dest tensor info:"
              << tensor4.to_string() 
              << std::endl;
    std::cout << "dest tensor:"
              << std::endl
              << tensor4.show()
              << std::endl;
}

void only_one_inverted_order_test(){
    pair<int, int> res = only_one_inverted_order({0, 1, 3, 2});
    if (res.first == 3 && res.second == 2){
        std::cout << "{0, 1, 3, 2}" << std::endl;
    }

    res = only_one_inverted_order({0, 1, 4, 2});
    if (res.first == -1 && res.second == -1){
        std::cout << "!{0, 1, 4, 2}" << std::endl;
    }

    res = only_one_inverted_order({0, 1, 1, 2});
    if (res.first == -1 && res.second == -1){
        std::cout << "!{0, 1, 1, 2}" << std::endl;
    }

    res = only_one_inverted_order({0, 1, 4, 3, 2});
    if (res.first == 4 && res.second == 2){
        std::cout << "{0, 1, 4, 3, 2}" << std::endl;
    }
}

#endif //BEYOND_EXECUTE_TEST_H

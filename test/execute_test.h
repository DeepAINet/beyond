//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_EXECUTE_TEST_H
#define BEYOND_EXECUTE_TEST_H

#include "../src/log.h"
#include "../src/shape.h"
#include "../src/tensor.h"
#include "../src/initializers.h"

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
}

#endif //BEYOND_EXECUTE_TEST_H

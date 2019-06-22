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
#include "../src/session.h"
#include "../src/corpus.h"

using namespace tops;

void test_basic_info(string name){
    string up(name.size() + 10, '=');
    string content = "|    " + name + "    |";
    std::cout << std::endl << up << std::endl << content << std::endl << up << std::endl;
}

void test_func_name(string name){
    string up(name.size() + 10, '*');
    string down(name.size() + 10, '*');
    string content = "||   " + name + "   ||";
    std::cout << up << std::endl << content << std::endl << down << std::endl;
}

void logger_test(){
    test_basic_info("logger test");
    logger.info("Hello, world!");
    logger.debug("Hello, world!");
    logger.error("Hello, world!");
}

void shape_test(){
    test_basic_info("shape test");
    vector<int> vec = {1, 2, 3};
    std::cout << "shape(const vector<int>& dims):" << std::endl;
    shape sp(vec);
    std::cout << sp.to_str() << std::endl;

    std::cout << "test operator []:" << std::endl
              << "sp[0]:" << sp[0] << std::endl
              << "sp[-1]:" << sp[-1] << std::endl;

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
    test_basic_info("tensor test");
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
    test_basic_info("variable test");
    variable& variable1 = get_variable("X", {2, 3, 4}, true, 0.1f, 0.2f);
    std::cout << variable1.to_str() << std::endl;
    std::cout << variable1.get().show() << std::endl;

    variable& variable2 = get_variable("b", {3, 4}, true, 0.2f, 0.3f);
    variable& variable3 = variable1 + variable2;
    std::cout << variable3.name << std::endl;

    variable& variable4 = variable2 + variable1 + variable3;

    std::cout << variable1.to_str() << std::endl;
    std::cout << variable2.to_str() << std::endl;
    std::cout << variable3.to_str() << std::endl;
    std::cout << variable4.to_str() << std::endl;
}

void tops_transpose_test(){
    test_basic_info("tops::transpose test");
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
    test_basic_info("only_one_inverted_order test");
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

void same_shape_backward_test(){
    test_basic_info("same_shape_backward test");
    shape shape1({1, 2, 3, 4}), shape2({2, 3, 4});
    bool res = same_shape_backward(shape1, shape2);
    if (res){
        std::cout << "{1, 2, 3, 4}, {2, 3, 4} has same backward shape." << std::endl;
    }

    shape shape3({2, 3, 4}), shape4({2, 3, 4});
    res = same_shape_backward(shape3, shape4);
    if (res){
        std::cout << "{2, 3, 4}, {2, 3, 4} has same backward shape." << std::endl;
    }

    shape shape5({4}), shape6({4});
    res = same_shape_backward(shape5, shape6);
    if (res){
        std::cout << "{4}, {4} has same backward shape." << std::endl;
    }
}

void tops_add_test(){
    test_basic_info("tops::add test");
    tensor<real> a({2, 3, 4});
    tensor<real> b({3, 4});
    uniform_initializer(1.0f, 3.0f, a);
    uniform_initializer(1.0f, 3.0f, b);
    tensor<real> c;
    tops::add(a, b, c);
    std::cout << a.to_string() << std::endl
              << a.show() << std::endl
              << b.to_string() << std::endl
              << b.show() << std::endl
              << c.to_string() << std::endl
              << c.show() << std::endl;

    tensor<real> d({2, 3, 4});
    tensor<real> e({2, 3, 4});
    uniform_initializer(1.0f, 3.0f, d);
    uniform_initializer(1.0f, 3.0f, e);
    tensor<real> f;
    tops::add(d, e, f);
    std::cout << d.to_string() << std::endl
              << d.show() << std::endl
              << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tensor<real> g({4});
    tensor<real> h({4});
    uniform_initializer(1.0f, 2.0f, g);
    uniform_initializer(1.0f, 2.0f, h);
    tensor<real> m;
    tops::add(g, h, m);
    std::cout << g.to_string() << std::endl
              << g.show() << std::endl
              << h.to_string() << std::endl
              << h.show() << std::endl
              << m.to_string() << std::endl
              << m.show() << std::endl;
}

void tops_subtract_test(){
    test_basic_info("tops::substract test");
    tensor<real> a({2, 3, 4});
    tensor<real> b({3, 4});
    uniform_initializer(1.0f, 3.0f, a);
    uniform_initializer(1.0f, 3.0f, b);
    tensor<real> c;
    tops::subtract(a, b, c);
    std::cout << a.to_string() << std::endl
              << a.show() << std::endl
              << b.to_string() << std::endl
              << b.show() << std::endl
              << c.to_string() << std::endl
              << c.show() << std::endl;

    tensor<real> d({2, 3, 4});
    tensor<real> e({2, 3, 4});
    uniform_initializer(1.0f, 3.0f, d);
    uniform_initializer(1.0f, 3.0f, e);
    tensor<real> f;
    tops::subtract(d, e, f);
    std::cout << d.to_string() << std::endl
              << d.show() << std::endl
              << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tensor<real> g({4});
    tensor<real> h({4});
    uniform_initializer(1.0f, 2.0f, g);
    uniform_initializer(1.0f, 2.0f, h);
    tensor<real> m;
    tops::subtract(g, h, m);
    std::cout << g.to_string() << std::endl
              << g.show() << std::endl
              << h.to_string() << std::endl
              << h.show() << std::endl
              << m.to_string() << std::endl
              << m.show() << std::endl;
}

void tops_mul_test(){
    test_basic_info("tops::mul test");
    tensor<real> a({2, 3, 4});
    tensor<real> b({3, 4});
    uniform_initializer(1.0f, 3.0f, a);
    uniform_initializer(2.0f, 2.0f, b);
    tensor<real> c;
    tops::mul(a, b, c);
    std::cout << a.to_string() << std::endl
              << a.show() << std::endl
              << b.to_string() << std::endl
              << b.show() << std::endl
              << c.to_string() << std::endl
              << c.show() << std::endl;

    tensor<real> d({2, 3, 4});
    tensor<real> e({2, 3, 4});
    uniform_initializer(1.0f, 3.0f, d);
    uniform_initializer(2.0f, 2.0f, e);
    tensor<real> f;
    tops::mul(d, e, f);
    std::cout << d.to_string() << std::endl
              << d.show() << std::endl
              << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tensor<real> g({4});
    tensor<real> h({4});
    uniform_initializer(1.0f, 2.0f, g);
    uniform_initializer(2.0f, 2.0f, h);
    tensor<real> m;
    tops::mul(g, h, m);
    std::cout << g.to_string() << std::endl
              << g.show() << std::endl
              << h.to_string() << std::endl
              << h.show() << std::endl
              << m.to_string() << std::endl
              << m.show() << std::endl;
}

void tops_divide_test(){
    test_basic_info("tops::divide test");
    tensor<real> a({2, 3, 4});
    tensor<real> b({3, 4});
    uniform_initializer(1.0f, 3.0f, a);
    uniform_initializer(2.0f, 2.0f, b);
    tensor<real> c;
    tops::divide(a, b, c);
    std::cout << a.to_string() << std::endl
              << a.show() << std::endl
              << b.to_string() << std::endl
              << b.show() << std::endl
              << c.to_string() << std::endl
              << c.show() << std::endl;

//    tops::divide(b, a, c);
//    std::cout << a.to_string() << std::endl
//              << a.show() << std::endl
//              << b.to_string() << std::endl
//              << b.show() << std::endl
//              << c.to_string() << std::endl
//              << c.show() << std::endl;

    tensor<real> d({2, 3, 4});
    tensor<real> e({2, 3, 4});
    uniform_initializer(1.0f, 3.0f, d);
    uniform_initializer(2.0f, 2.0f, e);
    tensor<real> f;
    tops::divide(d, e, f);
    std::cout << d.to_string() << std::endl
              << d.show() << std::endl
              << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tensor<real> g({4});
    tensor<real> h({4});
    uniform_initializer(1.0f, 2.0f, g);
    uniform_initializer(2.0f, 2.0f, h);
    tensor<real> m;
    tops::divide(g, h, m);
    std::cout << g.to_string() << std::endl
              << g.show() << std::endl
              << h.to_string() << std::endl
              << h.show() << std::endl
              << m.to_string() << std::endl
              << m.show() << std::endl;
}

void tops_sigmoid_test(){
    test_basic_info("tops::sigmoid test");
    tensor<float> src({3, 4}), des;
    zeros(src);
    tops::sigmoid(src, des);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

}

void tops_relu_test(){
    test_basic_info("tops::relu test");
    tensor<float> src({3, 4}), des;
    uniform_initializer(-1.0f, 2.0f, src);
    tops::relu(src, des);

    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;
}

void tops_min_test(){
    test_basic_info("tops::min test");
    tensor<float> src({3, 4}), des;
    uniform_initializer(-1.0f, 2.0f, src);

    tops::min(src, des, 0, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::min(src, des, 1, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::min(src, des, -1, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::min(src, des, 1, false);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tensor<float> src0({2, 3, 4});
    uniform_initializer(-1.0f, 2.0f, src0);
    tops::min(src0, des, 0, true);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::min(src0, des, 1, true);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::min(src0, des, 2, false);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;
}

void tops_max_test(){
    test_basic_info("tops::max test");
    tensor<float> src({3, 4}), des;
    uniform_initializer(-1.0f, 2.0f, src);

    tops::max(src, des, 0, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::max(src, des, 1, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::max(src, des, -1, true);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::max(src, des, 1, false);
    std::cout << src.to_string() << std::endl
              << src.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tensor<float> src0({2, 3, 4});
    uniform_initializer(-1.0f, 2.0f, src0);
    tops::max(src0, des, 0, true);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::max(src0, des, 1, true);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;

    tops::max(src0, des, 2, false);
    std::cout << src0.to_string() << std::endl
              << src0.show() << std::endl
              << des.to_string() << std::endl
              << des.show() << std::endl;
}

void tops_softmax_test(){
    test_basic_info("tops::softmax test");
    vector<real> a(10, 0.0f), c(10);
    std::iota(a.begin(), a.end(), 0);
    test_func_name("tops::softmax(real *, 10, real *)");
    tops::softmax(a.data(), 10, c.data());
    for (int i = 0; i < 10; ++i)
        std::cout << c[i] << '\t';
    std::cout << std::endl;

    vector<double> b(10, 0.0), d(10);
    std::iota(b.begin(), b.end(), 0);
    test_func_name("tops::softmax(double *, 10, double *)");
    tops::softmax(b.data(), 10, d.data());
    for (int i = 0; i < 10; ++i)
        std::cout << d[i] << '\t';
    std::cout << std::endl;

    tensor<real> e({2, 3, 4});
    real *pe = e.data();
    for (int i = 0; i < 24; ++i)
        *pe++ = i;
    tensor<real> f;
    tops::softmax(e, f, -1);
    test_func_name("softmax(e, f, -1)");
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tops::softmax(e, f, 1);
    test_func_name("softmax(e, f, 1)");
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tops::softmax(e, f, 2);
    test_func_name("softmax(e, f, 2)");
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    tops::softmax(e, f, 0);
    test_func_name("softmax(e, f, 0)");
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl
              << f.to_string() << std::endl
              << f.show() << std::endl;

    test_func_name("softmax(e, e, 1)");
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl;
    tops::softmax(e, e, 1);
    std::cout << e.to_string() << std::endl
              << e.show() << std::endl;
}

void corpus_test(){
    corpus<real> corp("./test/test_data.txt");
    tensor<real> batch_x({3, 8}), batch_y({3, 1});
    corp.dense_input_fn(batch_x, batch_y);
    std::cout << batch_x.to_string() << std::endl
              << batch_x.show() << std::endl;
    std::cout << batch_y.to_string() << std::endl
              << batch_y.show() << std::endl;
}

void session_test(){
    test_basic_info("session test");
    eval_steps_ = 10;
    session session1;

    session1.run_config("./test/test_data.txt",
            "", &corpus<real>::dense_input_fn, 0, 2, 10, 10000, 100000000, 1000);
    session1.run();
}


void tests(){
    logger_test();
    shape_test();
    tensor_test();
    variable_test();
    same_shape_backward_test();
    tops_add_test();
    tops_subtract_test();
    tops_mul_test();
    tops_divide_test();
    tops_sigmoid_test();
    tops_relu_test();
    tops_min_test();
    tops_max_test();
    tops_softmax_test();
    corpus_test();
//    only_one_inverted_order_test();
//    tops_transpose_test();
    session_test();
}


#endif //BEYOND_EXECUTE_TEST_H

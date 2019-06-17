//
// Created by mengqy on 2019/6/4.
//
#include <iostream>
#include "test/execute_test.h"
#include "src/initializers.h"
using namespace std;
int main(){
    logger_test();
    shape_test();
    tensor_test();
    variable_test();
    only_one_inverted_order_test();
    tops_transpose_test();
    session_test();
}

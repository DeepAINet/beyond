//
// Created by mengqy on 2019/6/24.
//

#ifndef BEYOND_DNN_H
#define BEYOND_DNN_H

#include "../src/global.h"
#include "../src/variable.h"
#include "../src/nn.h"
#include "model.cc"

class dnn: public model{
public:
    vector<int>         hidden_dims;
    vector<string>      hidden_act_fn;
    int                 x_dim;
    int                 y_dim;

public:
    dnn(){}

    void train_input_fn(tensor<real>& x, tensor<real>& y){

    }

    void eval_input_fn(tensor<real>& x, tensor<real>& y){

    }

    void model_fn(){
//        variable& batch_x = get_variable("batch_x_0", {batch_size_, x_dim}, false, true, 0.0f, 0.0f);
//        variable& batch_y = get_variable("Y", {batch_size_, y_dim}, false, true, 0.0f, 0.0f);
//        variable* input = &batch_x, *output=0;
//        for (int hdim: hidden_dims){
//            output = &dense(*input, hdim);
//            input = output;
//        }
//        glops::softmax_entropy_with_logits(*input, batch_y);
    }

};


#endif //BEYOND_DNN_H

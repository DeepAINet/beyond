#include "../src/variable.h"
#include "../src/global.h"
#include "../src/nn.h"

class dnn{
public:
    vector<int> hidden_dims;
    vector<string> hidden_act_fn;
    int x_dim;
    int y_dim;

public:
    dnn(){}

    void train_input_fn(){

    }

    void eval_input_fn(){

    }

    void model_fn(){
        variable& batch_x = get_variable("X", {batch_size, x_dim}, false);
        variable& batch_y = get_variable("Y", {batch_size, y_dim}, false);
        variable* input = &batch_x, output=0;
        for (int hdim: hidden_dims){
            output = dense(*input, hdim);
            input = output;
        }
        softmax_entropy_with_logits(*input, batch_y);
    }

};

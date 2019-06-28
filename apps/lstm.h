#include "../src/common.h"
#include "../src/variable.h"
#include "../src/nn.h"
#include "model.cc"

typedef variable& (*activation)(variable&);

class lstm: public model{
private:
    int         x_dim;
    int         max_seq_len;
    int         hidden_dim;
    string      act_fn;
public:
    lstm(int x_dim, int max_seq_len, int hidden_dim, string act_fn="ReLU")
            :x_dim(x_dim), max_seq_len(max_seq_len), hidden_dim(hidden_dim), act_fn(act_fn){}

    void train_input_fn(tensor<real>& x, tensor<real>& y){

    }

    void eval_input_fn(tensor<real>& x, tensor<real>& y){

    }

    variable* get_gate(string gate_name,
                       int i,
                       variable* h,
                       activation act=&variable::sigmoid){
        assert(i >= 0);
        variable *wh=0;
        if (i > 0){
            string h_name = "w" + gate_name + 'h' + std::to_string(i);
            variable& wh_ = dense(*h, hidden_dim, false, h_name);
            wh = &wh_;
        }

        variable& x = get_variable("x"+std::to_string(i),
                                   {batch_size_, x_dim}, false, true, 0.0f, 0.0f);
        string x_name = "w" + gate_name + "x";
        string bias_name = "b" + gate_name;
        variable& wxb = dense(x, hidden_dim, true, x_name, bias_name);
        if (i > 0){
            variable& gate = (*act)(*wh + wxb);
            return &gate;
        }
        variable& gate = (*act)(wxb);
        return &gate;
    }

    void model_fn(){
        variable *f = 0, *i = 0, *cell = 0, *o = 0, *c = 0, *h = 0;
        for (int k = 0; k < max_seq_len; ++k){
            f = get_gate("f", k, h);
            i = get_gate("i", k, h);
            cell = get_gate("c", k, h, &variable::tanh);
            o = get_gate("o", k, h);

            if (0 == k){
                variable& c_ = (*i) * (*cell);
                c = &c_;
            } else {
                variable& c_ = (*f) * (*c) + (*i) * (*cell);
                c = &c_;
            }

            variable& h_ = (*o) * variable::tanh(*c);
            h = &h_;
        }
    }

    void classify_model_fn(){

    }

    void series_label_model_fn(){

    }
};

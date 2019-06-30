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
                       activation act=&glops::sigmoid){
        assert(i >= 0);
        variable *wh=0;
        if (i > 0){
            string h_name = "W" + gate_name + 'h';
            variable& wh_ = dense(*h, hidden_dim, false, h_name);
            wh = &wh_;
        }

        variable& x = get_variable("batch_x"+std::to_string(i),
                                   {batch_size_, x_dim}, false, true, 0.0f, 0.0f);
        string x_name = "W" + gate_name + "x";
        string bias_name = "B" + gate_name;
        variable& wxb = dense(x, hidden_dim, true, x_name, bias_name);
        if (i > 0){
            variable& gate = (*act)(*wh + wxb);
            gate.name = gate_name + std::to_string(i);
            return &gate;
        }
        variable& gate = (*act)(wxb);
        gate.name = gate_name + std::to_string(i);
        return &gate;
    }

    void model_fn(){
        variable *f = 0, *i = 0, *cell = 0, *o = 0, *c = 0, *h = 0;
        for (int k = 0; k < max_seq_len; ++k){
            f = get_gate("f", k, h);
            i = get_gate("i", k, h);
            cell = get_gate("c", k, h, &glops::tanh);
            o = get_gate("o", k, h);

            if (0 == k){
                variable& c_ = (*i) * (*cell);
                c = &c_;
            } else {
                variable& c_ = (*f) * (*c) + (*i) * (*cell);
                c = &c_;
            }

            variable& h_ = (*o) * glops::tanh(*c);
            h = &h_;
            h->name = "H" + std::to_string(k);
        }
    }

    void classify_model_fn(){

    }

    void series_label_model_fn(){

    }
};

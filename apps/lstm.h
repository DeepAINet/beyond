#include "../src/common.h"
#include "../src/variable.h"
#include "../src/nn.h"
#include "model.cc"

typedef variable &(*activation)(variable &);

class lstm : public model {
private:
    int             x_dim;
    int             max_seq_len;
    int             hidden_dim;
    string          act_fn;
public:
    lstm(int x_dim, int max_seq_len, int hidden_dim, string act_fn = "ReLU")
    : x_dim(x_dim), max_seq_len(max_seq_len), hidden_dim(hidden_dim), act_fn(act_fn) {}

    void train_input_fn(tensor<real> &x, tensor<real> &y) {
        // get input batch
    }

    void eval_input_fn(tensor<real> &x, tensor<real> &y) {

    }

    /**
     *
     * @param name: the name of the gate.
     * @param seq_idx: the position in the sequence.
     * @param h: the hidden level output variable at one position of the sequence.
     * @param act: the activation function.
     * @return:
     */
    variable* get_gate(string name, int seq_idx, variable *h, activation act = &glops::sigmoid) {
        assert(seq_idx > 0 || (seq_idx == 0 && h == 0));
        variable &x = get_variable("batch_x" + std::to_string(seq_idx),
                                   {batch_size_, x_dim}, false, false, 0.0f, 0.0f);
        string weights_name = "W" + name + "x";
        string bias_name = "B" + name;
        variable &wxb = dense(x, hidden_dim, true, weights_name, bias_name);

        if (seq_idx == 0){
            variable &gate = (*act)(wxb);
            gate.name = name + std::to_string(seq_idx);
            return &gate;
        }

        weights_name = "W" + name + 'h';
        variable& wh = dense(*h, hidden_dim, false, weights_name);
        variable& gate = (*act)(wh + wxb);
        gate.name = name + std::to_string(seq_idx);
        return &gate;
    }

    void model_fn() {
        variable *f = 0, *i = 0, *cell = 0, *o = 0, *c = 0, *H = 0;
        for (int k = 0; k < max_seq_len; ++k) {
            f = get_gate("f", k, H);
            i = get_gate("i", k, H);
            cell = get_gate("c", k, H, &glops::tanh);
            o = get_gate("o", k, H);

            if (0 == k) {
                variable &c_ = (*i) * (*cell);
                c = &c_;
            } else {
                variable &c_ = (*f) * (*c) + (*i) * (*cell);
                c = &c_;
            }

            variable &h_ = (*o) * glops::tanh(*c);
            H = &h_;
            H->name = "H" + std::to_string(k);
        }
    }

    void classify_model_fn() {

    }

    void series_label_model_fn() {

    }
};

#include "../src/variable.h"
#include "../src/global_variables.h"
#include "../src/log.h"

class transformer {
public:
    int embedding_dim;
    int max_seq_len;


public:

    variable &get_token_embeddings(int vocab_size, int embedding_dim, bool zero_padded=true){
        variable token_embeddings = get_variable("token_embeddings",
                                                 {vocab_size, embedding_dim},
                                                 true, true,
                                                 -1.0f * std::sqrt(1.0f/(float)embedding_dim),
                                                 1.0f * std::sqrt(1.0f/(float)embedding_dim),
                                                 "uniform");
        if (zero_padded) {
            real *pt = token_embeddings.get().data();
            for (int i = 0; i < embedding_dim; ++i)
                *pt++ = 0.0f;
        }
    }

    variable &scaled_dot_product_attention(variable &query,
                                           variable &key,
                                           variable &value,
                                           real dropout_rate,
                                           bool causality=false,
                                           bool training=true){



    }

    variable &multihead_attention(variable &queries,
                                  variable &keys,
                                  variable &values,
                                  int num_heads=8,
                                  real dropout_rate=0.5f,
                                  bool training=true,
                                  bool casualty=true){

    }

    /**
     * 初始化pos_embeddings.
     * @param pos_embeddings
     */
    void init_pos_embeddings(variable& pos_embeddings){
        variable position_embeddings = get_variable("pos_embeddings",
                                                    {max_seq_len, embedding_dim},
                                                    false, false, 0.0f, 0.0f);
        shape sp = pos_embeddings.get().get_shape();
        int pos_max = sp[0];
        real *pp = pos_embeddings.get().data();
        for(int pos_idx = 0; pos_idx < pos_max; ++pos_idx){
            for (int j = 0; j < embedding_dim; ++j){
                *pp++ = j % 2 == 0 ? (float)std::sin(pos_idx / std::pow(10000, (float)j / (float)embedding_dim)) : (float)std::cos(pos_idx / std::pow(10000, (float)(j - 1) / (float)embedding_dim));
            }
        }
        logger.info("Position embeddings has been initialized successfully!");
    }

    void mask(variable &input, variable &batch_position_embeddings){
        real *pi = input.get().data(), *pb = batch_position_embeddings.get().data();
        shape sp = input.get().get_shape();
        for (int batch_idx = 0; batch_idx < sp[0]; ++batch_idx){
            for (int pos_idx = 0; pos_idx < max_seq_len; ++pos_idx){
                bool masked = true;
                for (int embedding_idx = 0; embedding_idx < embedding_dim; ++embedding_idx){
                    if (*pi++ != 0.0f) masked = false;
                }
                if (masked) {
                    for (int embedding_idx = 0; embedding_idx < embedding_dim; ++embedding_idx)
                        *pb++ = 0.0f;
                } else pb += embedding_dim;
            }
        }
    }

    variable &positional_encoding(variable& input, bool masked=true){
        shape sp = input.get().get_shape();
        assert(sp.ndims() == 3);
        int batch_size = sp[0];
        variable batch_position_embeddings = get_variable("batch_position_embeddings",
                                                          {batch_size, max_seq_len, embedding_dim},
                                                          false, false, 0.0f, 0.0f);
        init_pos_embeddings(batch_position_embeddings);
        if (masked) mask(input, batch_position_embeddings);
        return batch_position_embeddings;
    }




//    variable &encode(){
//
//        return ;
//    }

//    variable &decode(){
//
//    }

};

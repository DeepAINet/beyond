//
// Created by mengqy on 2019/7/22.
//

#ifndef BEYOND_GLOBAL_VARIABLES_H
#define BEYOND_GLOBAL_VARIABLES_H

class GlobalVariables{
public:
    map<string, variable*> global_variables;
private:
    GlobalVariables *instance;

    GlobalVariables(){
        instance = 0;
    }

public:
    static GlobalVariables* get(){
        if (0 == instance){
            instance = new GlobalVariables();
            return instance;
        }else return instance;
    }

    ~GlobalVariables(){
        for (auto iter = global_variables.begin(); iter != global_variables.end(); ++iter){
            if (iter->second != 0) {
                delete iter->second;
                iter->second = 0;
            }
        }
        if (0 != instance) {
            delete instance;
            instance = 0;
        }
    }
};

/**
 *
 * @tparam T
 * @param name : string, the name of the variable.
 * @param sp : vector<int>, the shape of the variable.
 * @param trainable : bool, indicates whether to be optimized, e.g. params to be estimated .
 * @param grad: bool, indicates whether to compute the gradient of the variable.
 * @param low: T, the low value.
 * @param high: T, the high value.
 * @param initializer: function pointer, the initializing function.
 * @return
 */
template<typename T>
variable &get_variable(string name,
                       vector<int> sp,
                       bool trainable = true,
                       bool grad = true,
                       T low = 0.0f,
                       T high = 0.0f,
                       string initializer = "uniform") {
    if (GlobalVariables.get()->global_variables.count(name) == 1)
        return *(GlobalVariables.get()->global_variables[name]);

    variable *v = new variable(sp, trainable, grad, name);
    init_variable(low, high, v->get(), initializer);
    GlobalVariables.get()->global_variables[name] = v;
    if (trainable) {
        grad_tables[name] = new tensor<real>(sp);
    }
    return *v;
}

#endif //BEYOND_GLOBAL_VARIABLES_H

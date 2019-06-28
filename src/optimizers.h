//
// Created by mengqy on 2019/6/14.
//

#ifndef BEYOND_OPTIMIZERS_H
#define BEYOND_OPTIMIZERS_H

class optimizer{
public:
    string name;
    virtual optimize()=0;
};

class AdaGradOptimizer:public optimizer{

};

class AdamOptimizer:public optimizer{

};

class RMSPropOptimizer:public optimizer{

};

#endif //BEYOND_OPTIMIZERS_H

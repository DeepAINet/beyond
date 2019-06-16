//
// Created by mengqy on 2019/6/12.
//

#ifndef BEYOND_OPS_H
#define BEYOND_OPS_H

#include <string>
#include "common.h"

namespace ops{

    class op {
    public:
        string name="op";
    public:
        virtual ~op() {};
        virtual void forward()=0;
        virtual void backward()=0;
    };

}

#endif //BEYOND_OPS_H

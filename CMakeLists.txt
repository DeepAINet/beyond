cmake_minimum_required(VERSION 3.12)
project(Beyond)

set(CMAKE_CXX_STANDARD 11)

add_executable(Beyond
        main.cpp
        src/common.h
        src/initializers.h
        src/log.h
        src/shape.h
        src/shape.cpp
        src/tensor.h
        src/variable.h
        src/global.h
        src/global_ops.h
        src/ops.h
        src/session.h)



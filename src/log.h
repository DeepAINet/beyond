//
// Created by mengqy on 2019/6/4.
//

#ifndef BEYOND_LOG_H
#define BEYOND_LOG_H

#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

char tmp[1000];

class Logger {
private:
    ofstream out;
    bool console;
    int last_second;
    time_t now;
    time_t last;

public:
    Logger() {
        console = true;
        last_second = 0;
    }

    Logger(string name) {
        out.open(name);
        if (!out.is_open()) {
            std::cerr << "Can't open file named " << name << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~Logger() {
        if (out.is_open()) out.close();
    }

    void info(string text, bool flush = false) {
        get_time();
        if (!flush) {
            if (console)
                std::cout << tmp << "- Beyond - INFO - " << text << std::endl;

            if (out.is_open())
                out << tmp << "- Beyond - INFO - " << text << std::endl;

        } else {
            std::cout << '\r' << tmp << "- Beyond - INFO - " << text << std::flush;
        }
    }

    void debug(string text) {
        get_time();
        if (console)
            std::cout << tmp << "- Beyond - DEBUG - " << text << std::endl;
        if (out.is_open())
            out << tmp << "- Beyond - DEBUG - " << text << std::endl;
    }

    void error(string text) {
        get_time();
        if (console)
            std::cout << tmp << "- Beyond - ERROR - " << text << std::endl;
        if (out.is_open())
            out << tmp << "- Beyond - ERROR - " << text << std::endl;
    }

    void get_time() {
        now = time(nullptr);
        tm *t = localtime(&now);
        sprintf(tmp, "%d-%02d-%02d %02d:%02d:%02d ", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec);
    }

    float get_diff_time() {
        now = time(nullptr);
        float res = difftime(now, last);
        last = now;
        return res;
    }
};

Logger logger;

#endif //BEYOND_LOG_H

#ifndef __MPPA_UTILS_H
#define __MPPA_UTILS_H

#include <iostream>
#include <chrono>
#include <vector>

struct work_area_t {
    int x_init,
        y_init,
        x_final,
        y_final;
    std::vector<int> dist_to_border;
    work_area_t(int x_init_, int y_init_, int x_final_, int y_final_, std::initializer_list<int> vector_) :
                x_init{x_init_}, 
                y_init{y_init_}, 
                x_final{x_final_}, 
                y_final{y_final_}, 
                dist_to_border{vector_}{}
};

std::chrono::time_point<std::chrono::steady_clock> mppa_master_get_time(void);
std::chrono::time_point<std::chrono::steady_clock> mppa_slave_get_time(void);
double mppa_master_diff_time(std::chrono::time_point<std::chrono::steady_clock> begin, std::chrono::time_point<std::chrono::steady_clock> end);
double mppa_slave_diff_time(std::chrono::time_point<std::chrono::steady_clock> begin, std::chrono::time_point<std::chrono::steady_clock> end);

#endif // __MPPA_UTILS_H
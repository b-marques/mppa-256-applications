#include "mppa_utils.h"

std::chrono::time_point<std::chrono::steady_clock> mppa_master_get_time(void) {
  return std::chrono::steady_clock::now();
}

std::chrono::time_point<std::chrono::steady_clock> mppa_slave_get_time(void) {
  return std::chrono::steady_clock::now();
}

double mppa_master_diff_time(std::chrono::time_point<std::chrono::steady_clock> begin, std::chrono::time_point<std::chrono::steady_clock> end) {
  std::chrono::duration<double> elapsed_seconds = end - begin;
  //   std::cout << begin.tv_sec << ", " << begin.tv_usec << std::endl;
  //   std::cout << end.tv_sec << ", " << end.tv_usec << std::endl;
  //   std::cout << (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) << std::endl;
  // return (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
  return elapsed_seconds.count();
}
double mppa_slave_diff_time(std::chrono::time_point<std::chrono::steady_clock> begin, std::chrono::time_point<std::chrono::steady_clock> end) {
  std::chrono::duration<double> elapsed_seconds = end - begin;
  return elapsed_seconds.count();
}
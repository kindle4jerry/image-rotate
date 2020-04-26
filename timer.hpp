/*
 * \file timer.h
 * \brief  print elapsed time
 */
#ifndef _TIMER_HPP_
#define _TIMER_HPP_
#include <iostream>
#include <string>
#include <chrono>
using namespace std;
class Timer{
  private:
    std::chrono::steady_clock::time_point last;
  public:
    Timer():last{std::chrono::steady_clock::now()} { }

    double printDiff(const std::string& msg = "Timer diff: ") {
      auto now{std::chrono::steady_clock::now()};
      std::chrono::duration<double, std::milli> diff{now - last};
      std::cout << msg << diff.count() << " ms\n";
      last = std::chrono::steady_clock::now();
      return diff.count();
    }

};
#endif 

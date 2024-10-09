#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

class TIMER
{
  private:
    std::chrono::high_resolution_clock::time_point start_time, stop_time;
    std::string indent = "      ";

  public:
    TIMER() = default;
    ~TIMER() = default;

    void indent_set(const std::string &s)
    {
        indent = s;
    }

    void tik()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void tok()
    {
        stop_time = std::chrono::high_resolution_clock::now();
    }

    void print_time(const std::string &str)
    {
        auto duration = std::chrono::duration<double, std::micro>(stop_time - start_time).count() / 1000.0;
        std::cout << indent << str << " use: " << std::fixed << std::setprecision(3) << duration << " ms\n";
    }

    void print_fps(const std::string &str)
    {
        auto duration = std::chrono::duration<double, std::micro>(stop_time - start_time).count() / 1000.0;
        std::cout << indent << str << " fps: " << std::fixed << std::setprecision(2) << 1000.0 / duration << "\n";
    }

    double get_time()
    {
        return std::chrono::duration<double, std::milli>(stop_time - start_time).count();
    }
};

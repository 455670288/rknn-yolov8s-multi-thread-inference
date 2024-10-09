#include "coreNum.hpp"

const int RK3588 = 3; //设定要使用的npu核数

int get_core_num()
{
    static int core_num = 0;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);

    int temp = core_num % RK3588;
    core_num++;
    return temp;
}

#ifndef CORENUM_HPP
#define CORENUM_HPP

#include <stdio.h>
#include "rknn_api.h"
#include <mutex>

// 设置模型需要绑定的核心
// Set the core of the model that needs to be bound
int get_core_num();
#endif
/**
 * @file utils.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Utility function definitions.
 * @version 0.1
 * @date 2025-06-26
 * 
 * Copyright (c) Riley Horrix 2025
 */
#include <chrono>

#include "common/Utils.h"

using namespace Dae;

uint64_t Utils::micros() {
    const std::chrono::system_clock::duration epoch = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(epoch).count());
}
/**
 * @file utils.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Definition of the Utils class.
 * @version 0.1
 * @date 2025-09-24
 * 
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#define UTILS_DEFINED

#include <cstdint>
#include <json.h>

#include "common/Logging.h"

namespace Dae {

/**
 * @brief Static utility functions.
 */
class Utils {
public: 
    /**
     * @brief Get the current time since unix epoch in microseconds.
     * 
     * @return uint64_t Microseconds since epoch.
     */
    static uint64_t micros();
};

} // Closing namespace Dae
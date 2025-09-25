/**
 * @file Configurable.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Implementation of the Configurable class.
 * @version 0.1
 * @date 2025-09-24
 * 
 * Copyright (c) Riley Horrix 2025
 */
#include <iostream>
#include <fstream>

#include "common/Logging.h"
#include "common/Configurable.h"

using namespace Dae;

Configurable::json Configurable::global;

Configurable::Configurable(const std::string& key): config(global[key]) {}

Configurable::~Configurable() {}

int Configurable::initialize(const std::string& filepath) {
    // Open the configuration file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        stl_warn(errno, "Failed to open configuration file at '%s'", filepath.c_str());
        return ST_FOPEN_FAIL;
    }

    // Parse configurations without throwing
    global = json::parse(file, nullptr, false, true, true);
    if (global.is_discarded()) {
        warn("Failed to parse JSON in file '%s'", filepath.c_str());
        return ST_PARSE_FAIL;
    }

    return 0;
}

double Configurable::confNum(const std::string& key, const double defaultVal) {
    return getOrDefault<double>(config, key, defaultVal);
}

std::string Configurable::confStr(const std::string& key, const std::string& defaultVal) {
    return getOrDefault<std::string>(config, key, defaultVal);
}
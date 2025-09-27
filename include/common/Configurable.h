/**
 * @file Configurable.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Definition of Configurable class.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#include <string>
#include <json.h>

#include "common/Logging.h"
#include "common/Common.h"

namespace Dae {

/**
 * @brief Simple JSON wrapper for configurable classes.
 */
class Configurable {
public:
    /**
     * @brief Status codes for Configurable class.
     */
    enum Status { ST_GOOD = 0, ST_FOPEN_FAIL, ST_PARSE_FAIL };

    /**
     * @brief Construct a new Configurable object.
     *
     * @param key Configuration object key.
     */
    explicit Configurable(const std::string& key);

    /**
     * @brief Destroy the Configurable object.
     */
    ~Configurable();

    /**
     * @brief Initialize the global configuration object from a configuration
     * file. This should be called before initializing any other object that
     * extends the Configurable class.
     *
     * @param filepath Path to the `.json` configuration file.
     * @return int Status code. 0 for success.
     */
    static int initialize(const std::string& filepath);

    /**
     * @brief Configure the class from it's configuration.
     *
     * Called on initialisation.
     *
     * This function should be overridden by derived classes.
     */
    virtual void configure(void) = 0;

    /**
     * @brief Configure a key in the class' configuration.
     *
     * @param key The value key.
     * @param value The value to change to.
     * @return double The previous value.
     */
    void cnf(const std::string& key, const double value);

    /**
     * @brief Configure a key in the class' configuration.
     *
     * @param key The value key.
     * @param value The value to change to.
     * @return std::string The previous value.
     */
    void cnf(const std::string& key, const std::string& value);

protected:
    /// @brief The string key of this configurable object.
    std::string key;

    /**
     * @brief Extract an double value from the configuration.
     *
     * @param key String key.
     * @param defaultVal Default value if it is not found.
     * @return double The value or default.
     */
    double confNum(const std::string& key, const double defaultVal = 0.0);

    /**
     * @brief Extract an string value from the configuration.
     *
     * @param key String key.
     * @param defaultVal Default value if it is not found.
     * @return std::string The value or default.
     */
    std::string confStr(const std::string& key,
                        const std::string& defaultVal = "");

private:
    using json = nlohmann::json;

    /// @brief Global json instance.
    static json global;

    /// @brief Class json instance.
    json config;

    /**
     * @brief Helper function to extract a field from a json object.
     *
     * @tparam T The type of the field.
     * @param json The json object.
     * @param key The string key.
     * @param defaultValue A default value.
     * @return T The found object or the default value;
     */
    template <typename T>
    static T getOrDefault(const nlohmann::json& json, const std::string& key,
                          const T& defaultValue) {
        if (json.contains(key) && !json[key].is_null()) {
            try {
                return json[key].get<T>();
            } catch (const nlohmann::json::exception& e) {
                info("Failed to parse %s as a " MSTR(T) ", \"%s\"", key.c_str(),
                     e.what());
                return defaultValue;
            }
        }
        return defaultValue;
    }
};

} // namespace Dae
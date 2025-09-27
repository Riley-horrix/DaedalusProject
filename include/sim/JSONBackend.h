/**
 * @file JSONBackend.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Definition of the JSONBackend class.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#include <vector>
#include <arpa/inet.h>

#include "common/Configurable.h"
#include "sim/PhysicsBackend.h"

namespace Dae {

/**
 * @brief The JSONBackend class implements the ArduPilot SITL interface.
 *
 * More details can be found here, https://github.com/ArduPilot/ardupilot/blob/master/libraries/SITL/examples/JSON/readme.md
 *
 * The external physics simulation process should host a UDP server at a known
 * address and port number.
 *
 * The JSONBackend will then connect to that UDP server and stream a binary
 * control signal to it, while receiving the vehicle telemetry in JSON format.
 *
 * The binary control format is specified as follows:
 * ```cpp
 *  struct ControlPacket {
 *      uint16_t magic = 18458;
 *      uint16_t frame_rate;
 *      uint32_t frame_count;
 *      uint16_t pwm[16];
 * };
 * ```
 * Every value in the control packet should be in network order.
 *
 * The JSON format is specified as follows:
 * ```json
 * {
 *      "timestamp" : physics time (s),
 *      "imu" : {
 *          "gyro" : [roll, pitch, yaw] (rad/s) in body frame,
 *          "accel_body" : [x, y, z] (m/s/s) in body frame,
 *      },
 *      "position" : [North, East, Down] in Earth frame,
 *      "velocity" : [North, East, Down] in Earth frame,
 *      "quaternion" : [w, x, y, z]
 * }
 * ```
 */
class JSONBackend : public PhysicsBackend, public Configurable {
public:
    /**
     * @brief Status codes for the JSON backend.
     */
    enum Status { ST_GOOD = 0, ST_SOCKET_FAIL, ST_BIND_FAIL, ST_MOVED_OUT };

    /**
     * @brief Construct a new JSONBackend object.
     *
     * @param key Configuration key.
     */
    JSONBackend(const std::string& key = "JSONBackend");

    /**
     * @brief Destroy the JSONBackend object.
     */
    ~JSONBackend();

    /**
    * @brief Copy construct a new JSONBackend object.
    *
    * @param other The other class instance.
    */
    JSONBackend& operator=(const JSONBackend& other) = delete;

    /**
    * @brief Copy construct a new JSONBackend object.
    *
    * @param other The other class instance.
    */
    JSONBackend(const JSONBackend& other) = delete;

    /**
    * @brief Move construct a new JSONBackend object.
    *
    * @param other The rvalue class instance.
    */
    JSONBackend(JSONBackend&& other);

    /**
    * @brief Move assignment construct a new JSONBackend object.
    *
    * @param other The rvalue class instance.
    */
    JSONBackend& operator=(JSONBackend&& other);

    /// @copydoc Dae::PhysicsBackend::iterate
    std::unique_ptr<Telemetry>
    iterate(const std::unique_ptr<Control> ctrl) override;

private:
    using json = nlohmann::json;

/**
     * @brief JSON Protocol binary packet.
     */
#pragma pack(push, 1)
    struct ControlPacket {
        uint16_t magic = htons(18458);
        uint16_t frame_rate;
        uint32_t frame_count;
        uint16_t pwm[16];
    };
#pragma pack(pop)

    // Configs

    /// @brief Timeout to wait for telemetry to be received (s). Config
    /// 'telem_timeout'
    double TELEM_TIMEOUT = 10;

    /// @brief Timeout to wait between trying to receive from the server (s).
    /// Config 'receive_timeout'.
    double RECEIVE_TIMEOUT = 0.01;

    /// @brief Port that the UDP server is hosted on. Config 'port'.
    uint16_t SERVER_PORT = 9002;

    /// @brief Address that the UDP server is hosted on. Config 'addr'.
    std::string SERVER_ADDR = "127.0.0.1";

    // Constants

    /// @brief Size of the input buffer for telemetry.
    static constexpr int BUFFER_SIZE = 1 << 10;

    // State

    /// @brief Socket file descriptor for UDP server.
    int sockfd;

    /// @brief The UDP server address.
    sockaddr_in serverAddr;

    /// @brief Buffer for sending the control packet.
    ControlPacket control;

    /// @brief The input buffer for telemetry.
    char telemBuffer[BUFFER_SIZE];

    // Methods

    /**
     * @brief Validate that the json object contains the `field`, and if it is
     * populated, then set the `value` to whatever is in the json field.
     *
     * @param value The value to extract into.
     * @param field The json field to extract from.
     * @param j The json object containing the value.
     * @return bool Status flag.
     */
    bool validateAndGetJson(double* value, const std::string& field, json j);

    /**
     * @brief Validate that the json array within the json object at `field`,
     * contains `len` values, and then moves the numbers into the `values`
     * array.
     *
     * @param values The values to extract into.
     * @param field The json array name.
     * @param j The json object containing the array.
     * @param len Number of elements in the array.
     * @return bool Status flag.
     */
    bool validateAndGetJsonArr(double* values, const std::string& field, json j,
                               const unsigned int len);

    /// @copydoc Dae::Configurable::configure
    void configure(void) override;
};

} // namespace Dae
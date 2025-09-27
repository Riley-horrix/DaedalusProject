/**
 * @file JSONBackend.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Implementation of the JSONBackend class.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cinttypes>

#include "common/Logging.h"
#include "common/Utils.h"
#include "sim/JSONBackend.h"

using namespace Dae;

JSONBackend::JSONBackend(const std::string& key) : Configurable(key) {
    configure();

    // Create the UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1) {
        stl_error(errno, "Failed to initialise UDP socket");
        statusCode = ST_SOCKET_FAIL;
    }

    // Configure the server address
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port   = htons(SERVER_PORT);
    if (inet_pton(AF_INET, SERVER_ADDR.c_str(), &serverAddr.sin_addr) != 1) {
        error("Failed to convert server network address '%s'",
              SERVER_ADDR.c_str());
    }

    info("JSONBackend connected to address %s on port %" PRIu16,
         SERVER_ADDR.c_str(), SERVER_PORT);
}

JSONBackend::~JSONBackend() { close(sockfd); }

JSONBackend::JSONBackend(JSONBackend&& other)
    : PhysicsBackend(std::move(other)), Configurable(std::move(other)),
      TELEM_TIMEOUT(other.TELEM_TIMEOUT),
      RECEIVE_TIMEOUT(other.RECEIVE_TIMEOUT), SERVER_PORT(other.SERVER_PORT),
      SERVER_ADDR(std::move(other.SERVER_ADDR)), sockfd(other.sockfd),
      serverAddr(other.serverAddr), control(other.control) {
    // Leave the other instance in a safe state.
    other.sockfd     = -1;
    other.statusCode = ST_MOVED_OUT;
}

JSONBackend& JSONBackend::operator=(JSONBackend&& other) {
    if (this != &other) {
        // Clean up existing socket
        if (sockfd >= 0) {
            close(sockfd);
            sockfd = -1;
        }

        PhysicsBackend::operator=(std::move(other));
        Configurable::operator=(std::move(other));

        TELEM_TIMEOUT   = other.TELEM_TIMEOUT;
        RECEIVE_TIMEOUT = other.RECEIVE_TIMEOUT;
        SERVER_PORT     = other.SERVER_PORT;
        SERVER_ADDR     = std::move(other.SERVER_ADDR);

        sockfd     = other.sockfd;
        serverAddr = other.serverAddr;

        other.sockfd = -1;
        other.statusCode = ST_MOVED_OUT;
    }

    return *this;
}

std::unique_ptr<PhysicsBackend::Telemetry>
JSONBackend::iterate(const std::unique_ptr<Control> ctrl) {
    // Fill up the control packet
    control.frame_rate  = htons(static_cast<uint16_t>(frameRate));
    control.frame_count = htonl(static_cast<uint32_t>(frameCount));

    int pwmLen = sizeof(ctrl->pwm) / sizeof(ctrl->pwm[0]);

    for (int i = 0; i < pwmLen; i++) {
        // Normalise [-1, 1] to [1000, 2000]
        control.pwm[i] =
            htons(static_cast<uint16_t>(std::round(ctrl->pwm[i] * 500) + 1500));
    }

    // Send the control packet over the UDP port to the physics backend
    ssize_t bytesSent =
        sendto(sockfd, &control, sizeof(control), 0,
               reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr));

    if (bytesSent < 0) {
        stl_error(errno, "Failed to send control packet");
        return nullptr;
    }

    // Listen for response for configured timeout
    uint64_t   start       = Utils::micros();
    uint64_t   timeout     = static_cast<uint64_t>(TELEM_TIMEOUT * 1e6);
    useconds_t recvTimeout = static_cast<useconds_t>(RECEIVE_TIMEOUT * 1e6);

    while (Utils::micros() < start + timeout) {
        // Receive the most recent data message
        ssize_t receivedBytes = recvfrom(sockfd, &telemBuffer, BUFFER_SIZE - 1,
                                         MSG_DONTWAIT, NULL, NULL);

        // Received an error
        if (receivedBytes <= 0) {
            if ((receivedBytes < 0 &&
                 (errno == EAGAIN || errno == EWOULDBLOCK)) ||
                receivedBytes == 0) {
                // This is fine and just means physics backend has not started
                // yet or no data
                if (usleep(recvTimeout) != 0) {
                    stl_warn(
                        errno,
                        "Failed to usleep while waiting for physics backend");
                    return nullptr;
                }
                continue;
            }
            stl_error(errno, "Received error when receiving from UDP server");
            return nullptr;
        }

        // NULL terminate received data
        telemBuffer[receivedBytes] = '\0';

        // Received a message so read the json.
        json msg = json::parse(telemBuffer, nullptr, false, true, true);

        if (msg.is_discarded()) {
            warn("Failed to parse telemetry message : %s", telemBuffer);
            continue;
        }

        // debug("Received JSON : %s", msg.dump().c_str());

        // Validate and set telemetry
        std::unique_ptr<Telemetry> telem = std::make_unique<Telemetry>();
        if (!validateAndGetJson(&telem->timestamp, "timestamp", msg)) {
            warn("JSON telemetry does not contain timestamp");
            continue;
        }

        if (!msg.contains("imu") || !msg["imu"].is_object()) {
            warn("JSON telemetry does not contain imu object");
            continue;
        }

        if (!validateAndGetJsonArr(telem->accel, "accel_body", msg["imu"], 3)) {
            warn("JSON telemetry does not contain timestamp");
            continue;
        }

        if (!validateAndGetJsonArr(telem->gyro, "gyro", msg["imu"], 3)) {
            warn("JSON telemetry does not contain timestamp");
            continue;
        }

        if (!validateAndGetJsonArr(telem->position, "position", msg, 3)) {
            warn("JSON telemetry does not contain position");
            continue;
        }

        if (!validateAndGetJsonArr(telem->velocity, "velocity", msg, 3)) {
            warn("JSON telemetry does not contain velocity");
            continue;
        }

        if (!validateAndGetJsonArr(telem->quaternion, "quaternion", msg, 4)) {
            warn("JSON telemetry does not contain quaternion");
            continue;
        }

        frameCount++;

        return telem;
    }

    warn("Physics backend telemetry request timed out");

    return nullptr;
}

bool JSONBackend::validateAndGetJson(double* value, const std::string& field,
                                     json j) {
    if (j.contains(field) && j[field].is_number()) {
        *value = j[field];
        return true;
    }

    return false;
}

bool JSONBackend::validateAndGetJsonArr(double*            values,
                                        const std::string& field, json j,
                                        const unsigned int len) {
    if (j.contains(field) && j[field].is_array() && j[field].size() == len) {
        for (unsigned int i = 0; i < len; i++) {
            const auto& number = j[field][i];

            if (!number.is_number()) {
                return false;
            }

            values[i] = number;
        }
        return true;
    }

    return false;
}

void JSONBackend::configure(void) {
    TELEM_TIMEOUT   = confNum("telem_timeout", TELEM_TIMEOUT);
    RECEIVE_TIMEOUT = confNum("receive_timeout", RECEIVE_TIMEOUT);
    SERVER_ADDR     = confStr("addr", SERVER_ADDR);
    SERVER_PORT     = static_cast<uint16_t>(
        confNum("port", static_cast<double>(SERVER_PORT)));
}
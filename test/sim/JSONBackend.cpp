/**
 * @file JSONBackend.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Testing file for JSONBackend class.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */
#include <catch2/catch_all.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <json.h>
#include <thread>

#include "sim/JSONBackend.h"

using namespace Dae;

TEST_CASE("JSONBackend can initialise and start with no running server",
          "[JSONBackend]") {
    JSONBackend backend;

    REQUIRE(backend);
}

TEST_CASE(
    "JSONBackend can send control packets successfully and decode telemetry",
    "[JSONBackend]") {

    // Initialise a UDP server
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    REQUIRE(sockfd != 0);

    sockaddr_in server;
    memset(&server, 0, sizeof(server));
    server.sin_family      = AF_INET;
    server.sin_port        = htons(9002);
    server.sin_addr.s_addr = INADDR_ANY;

    REQUIRE(bind(sockfd, reinterpret_cast<sockaddr*>(&server),
                 sizeof(server)) != -1);

    JSONBackend backend;
    REQUIRE(backend);

    backend.cnf("telem_timeout", 1.0);
    backend.cnf("receive_timeout", 0.1);

    // Send control to the backend
    std::unique_ptr<PhysicsBackend::Control> ctrl =
        std::make_unique<PhysicsBackend::Control>();
    int size = sizeof(ctrl->pwm) / sizeof(ctrl->pwm[0]);
    for (int i = 0; i < size; i++) {
        ctrl->pwm[i] = ((i * 2.0) / (size - 1)) - 1.0;
    }

    usleep(10000);

    // Invoke a single iteration on the Physics backend and let it timeout
    REQUIRE(backend.iterate(std::move(ctrl)) == nullptr);
    info("^^^ this is meant to happen");

    usleep(10000);

    // Read control
    const int BUFFER_SIZE = 1 << 10;
    char      buffer[BUFFER_SIZE];
    debug("Server doing read now");
    sockaddr_in client;
    memset(&client, 0, sizeof(client));
    socklen_t sockSize = sizeof(client);
    ssize_t   recvBytes =
        recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                 reinterpret_cast<sockaddr*>(&client), &sockSize);

    usleep(10000);

    debug("Recv bytes %ld", recvBytes);

#pragma pack(push, 1)
    struct ControlPacket {
        uint16_t magic;
        uint16_t frame_rate;
        uint32_t frame_count;
        uint16_t pwm[16];
    };
#pragma pack(pop)

    ControlPacket* packet = reinterpret_cast<ControlPacket*>(&buffer);

    // Size of control packet
    REQUIRE(recvBytes == sizeof(ControlPacket));

    REQUIRE(packet->magic == htons(18458));
    REQUIRE(packet->frame_rate == htons(50));
    REQUIRE(packet->frame_count == 0);

    size = sizeof(packet->pwm) / sizeof(packet->pwm[0]);
    for (int i = 0; i < size; i++) {
        double sent = ((i * 2.0) / (size - 1)) - 1.0;
        REQUIRE(packet->pwm[i] ==
                htons(static_cast<uint16_t>(std::round(sent * 500 + 1500))));
    }

    // Send back a telemetry response
    nlohmann::json telem;
    telem["timestamp"]         = 0.1;
    telem["imu"]["accel_body"] = {1.0, 2.0, 3.0};
    telem["imu"]["gyro"]       = {-1.0, -2.0, -3.0};
    telem["position"]          = {100, 1000, -500};
    telem["velocity"]          = {1, 10, -5};
    telem["quaternion"]        = {1, 0.12, 0.34, 0.56};

    std::string json = telem.dump(0);
    ssize_t     sentBytes =
        sendto(sockfd, json.c_str(), json.size(), 0,
               reinterpret_cast<sockaddr*>(&client), sizeof(client));
    debug("Sent bytes : %ld", sentBytes);
    if (sentBytes < 0) {
        stl_error(errno, "Bad things happening");
    }

    usleep(10000);

    std::unique_ptr<PhysicsBackend::Telemetry> telemPtr =
        backend.iterate(std::make_unique<PhysicsBackend::Control>());

    REQUIRE(telemPtr != nullptr);

    REQUIRE(sentBytes > 0);

    REQUIRE(telemPtr->timestamp == 0.1);

    REQUIRE(telemPtr->accel[0] == 1.0);
    REQUIRE(telemPtr->accel[1] == 2.0);
    REQUIRE(telemPtr->accel[2] == 3.0);

    REQUIRE(telemPtr->gyro[0] == -1.0);
    REQUIRE(telemPtr->gyro[1] == -2.0);
    REQUIRE(telemPtr->gyro[2] == -3.0);

    REQUIRE(telemPtr->position[0] == 100);
    REQUIRE(telemPtr->position[1] == 1000);
    REQUIRE(telemPtr->position[2] == -500);

    REQUIRE(telemPtr->velocity[0] == 1);
    REQUIRE(telemPtr->velocity[1] == 10);
    REQUIRE(telemPtr->velocity[2] == -5);

    REQUIRE(telemPtr->quaternion[0] == 1);
    REQUIRE(telemPtr->quaternion[1] == 0.12);
    REQUIRE(telemPtr->quaternion[2] == 0.34);
    REQUIRE(telemPtr->quaternion[3] == 0.56);
}
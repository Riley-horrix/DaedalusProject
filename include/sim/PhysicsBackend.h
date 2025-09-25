/**
 * @file PhysicsBackend.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Definition of PhysicsBackend class.
 * @version 0.1
 * @date 2025-09-24
 * 
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#include <memory>

namespace Dae {

/**
 * @brief Interface class for physics backend implementations. 
 * 
 * This interface is used to communicate commands and telemetry to / from a
 * connected physics backend.
 */
class PhysicsBackend {
public:
    /**
     * @brief Telemetry data struct to hold data about the vehicle from the 
     * physics simulation.
     */
    struct Telemetry{
        /// @brief Physics timestamp (s)
        double timestamp;
        /// @brief Gyroscope readings in body frame (rad/s) [roll, pitch, yaw]
        double gyro[3];
        /// @brief Accelerometer readings in body frame (m/s^s) [x, y, z]
        double accel[3];
        /// @brief Position in Earth frame (m) [North, East, Down]
        double position[3];
        /// @brief Velocity in Earth frame (m/s) [North, East, Down]
        double velocity[3];
        /// @brief Attitude quaternion [w, x, y, z]
        double quaternion[4];
    };

    /**
     * @brief A control data struct containing PWM control signals to be sent
     * to the physics backend.
     */
    struct Control{
        /// @brief PWM control signal array. Should be [-1, 1]
        double pwm[16];
    };

    /**
     * @brief Construct a new Physics Backend object.
     */
    PhysicsBackend(void);

    /**
     * @brief Destroy the Physics Backend object.
     */
    virtual ~PhysicsBackend(void);

    /**
     * @brief Set the requested FrameRate of the physics backend.
     * 
     * @param hz Frame rate in frames per second.
     */
    inline void setFrameRate(double hz);

    /**
     * @brief Run one iteration of the physics backend with the provided 
     * control signal.
     * 
     * @param ctrl Pointer to the desired control signal.
     * @return std::unique_ptr<Telemetry> Vehicle telemetry after the physics 
     * step.
     */
    virtual std::unique_ptr<Telemetry> iterate(const std::unique_ptr<Control> ctrl) = 0;

    /// @copydoc Dae::PhysicsBackend::status
    explicit operator bool();

    /**
     * @brief Get the current status of the component.
     * 
     * @return The component status.
     */
    inline bool status(void);

    /**
     * @brief Get the current status code of the component.
     * 
     * @return int The status code.
     */
    inline int getStatus(void);

protected:
    /// @brief Requested frame rate of physics backend.
    double frameRate = 50;

    /// @brief The current iteration frame.
    double frameCount = 0;

    /// @brief Status code of the component. 0 represents a good status.
    int statusCode = 0;
};

} // Closing namespace Dae
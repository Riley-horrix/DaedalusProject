/**
 * @file PhysicsBackend.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Implementation of PhysicsBackend class.
 * @version 0.1
 * @date 2025-09-24
 *
 * Copyright (c) Riley Horrix 2025
 */

#include "sim/PhysicsBackend.h"

using namespace Dae;

PhysicsBackend::PhysicsBackend(void) {}

PhysicsBackend::~PhysicsBackend(void) {}

void PhysicsBackend::setFrameRate(double hz) { frameRate = hz; }

PhysicsBackend::operator bool(void) { return status(); }

bool PhysicsBackend::status(void) { return statusCode == 0; }

int PhysicsBackend::getStatus(void) { return statusCode; }
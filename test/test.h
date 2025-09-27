/**
 * @file test.h
 * @author rileyhorrix (riley@horrix.com)
 * @brief Test utils.
 * @version 0.1
 * @date 2025-09-26
 *
 * Copyright (c) Riley Horrix 2025
 */
#pragma once

#include <catch2/catch_all.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include "common/Logging.h"

namespace Dae {

class CatchTestListener : public Catch::EventListenerBase {
public:
    using EventListenerBase::EventListenerBase; // inherit ctor required by Catch2

    /**
     * @brief .
     *
     * @param info
     */
    void testCaseStarting(Catch::TestCaseInfo const& info) override;
};

} // namespace Dae

// Register listener
CATCH_REGISTER_LISTENER(Dae::CatchTestListener);
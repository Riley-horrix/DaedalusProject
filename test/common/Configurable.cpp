/**
 * @file Configurable.cpp
 * @author rileyhorrix (riley@horrix.com)
 * @brief Testing file for Configurable class.
 * @version 0.1
 * @date 2025-09-24
 * 
 * Copyright (c) Riley Horrix 2025
 */
#include <catch2/catch_all.hpp>
#include <string>

#include "common/Configurable.h"

using namespace Dae;

class Component : public Configurable {
public:
    Component() : Configurable("Component") { configure(); }

    double a;
    double b;
    double c;

    std::string astr;
    std::string bstr;
    std::string cstr;

    void configure(void) override {
        a = confNum("a", 10);
        b = confNum("b", 10);
        c = confNum("c", 10);

        astr = confStr("a_str", "10");
        bstr = confStr("b_str", "goat");
        cstr = confStr("c_str", "balloon");
    }
};

TEST_CASE("Configurable can configure classes", "[Configurable]") {
    Configurable::initialize("test/common/Configurable.json");

    Component comp;

    SECTION("Configurable can configure numbers") {
        REQUIRE(comp.a == 100);
        REQUIRE(comp.b == 0);
        REQUIRE(comp.c == 10);
    }

    SECTION("Configurable can configure strings") {
        REQUIRE(comp.astr == "100");
        REQUIRE(comp.bstr == "");
        REQUIRE(comp.cstr == "balloon");
    }
}
#include <catch2/catch_all.hpp>

#include "test/test.h"

using namespace Dae;

void CatchTestListener::testCaseStarting(Catch::TestCaseInfo const& info){
    testInfo(info.tagsAsString().c_str(), "%s", info.name.c_str());
}
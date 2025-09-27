cmake_minimum_required(VERSION 3.28)
project(daedalus LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch Catch2 if it not on the host system
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, downloading...")
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.10.0  # Change to latest stable version
    )
    FetchContent_MakeAvailable(catch2)
endif()

# Set Flags
set(WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Werror
)
add_compile_options(${WARNINGS})

# Include CMake files
include(CMakeSources.cmake)

# Include directories
include_directories(.)
include_directories(include)
include_directories(lib/include)

# Build core library
add_library(daedalus_core STATIC ${SRC_FILES})

# Build main application
add_executable(daedalus src/daedalus.cpp)
target_link_libraries(daedalus PRIVATE daedalus_core)

# Build testing application
add_executable(test ${TEST_FILES} test/test.cpp)
target_link_libraries(test PRIVATE daedalus_core)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode!")
    target_compile_definitions(daedalus_core PRIVATE DEBUG)
    target_compile_definitions(daedalus PRIVATE DEBUG)
    target_compile_definitions(test PRIVATE DEBUG)
endif()

# Link Catch2 to test application
if(TARGET Catch2::Catch2)
    target_link_libraries(test PRIVATE Catch2::Catch2WithMain)
else()
    # If FetchContent was used, target is catch2
    target_link_libraries(test PRIVATE catch2Main)
endif()

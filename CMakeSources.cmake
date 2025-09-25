set(SRC_FILES 
    # Common
    ${CMAKE_SOURCE_DIR}/src/common/Logging.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Utils.cpp
    ${CMAKE_SOURCE_DIR}/src/common/Configurable.cpp

    # Sim
    ${CMAKE_SOURCE_DIR}/src/sim/PhysicsBackend.cpp
    ${CMAKE_SOURCE_DIR}/src/sim/JSONBackend.cpp
)

set(TEST_FILES
    # Common
    ${CMAKE_SOURCE_DIR}/test/common/Configurable.cpp

    # Sim
    ${CMAKE_SOURCE_DIR}/test/sim/JSONBackend.cpp
)
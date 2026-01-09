cmake_minimum_required(VERSION 3.0)
project(test_hbm_info)

set(CMAKE_BUILD_TYPE Release)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for header file search
include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
)

# Add library search path for linker
link_directories(/usr/hobot/lib)

# Define the executable target and its source files
add_executable(test_hbm_info
    test_hbm_info.cc
)

# Find required packages
find_package(Threads REQUIRED)

# Libraries to link with the target
SET(LINK_libs dnn hbucp Threads::Threads)

# Link libraries to the executable target
target_link_libraries(test_hbm_info ${LINK_libs})

# Set output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Print build info
message(STATUS "Building test_hbm_info - HBM Model Info Test Tool")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

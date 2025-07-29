cmake_minimum_required(VERSION 3.10)
project(styletts_ggml VERSION 1.0.0 LANGUAGES CXX)

# Compiler and flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Add debug information for debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -DDEBUG")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Architecture detection
include(CheckCXXCompilerFlag)

# Check compiler flag support
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
check_cxx_compiler_flag("-mfma" HAS_FMA)
check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    if(HAS_AVX2)
        add_compile_options(-mavx2)
    endif()
    if(HAS_FMA)
        add_compile_options(-mfma)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    if(HAS_MCPU_NATIVE)
        add_compile_options(-mcpu=native)
    endif()
endif()

# GGML path
set(GGML_PATH ${CMAKE_SOURCE_DIR}/ggml)

# See https://github.com/ggerganov/ggml/pull/682
add_definitions(-DGGML_MAX_NAME=128)

# Check if GGML sources exist
if(NOT EXISTS ${GGML_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "GGML sources not found in: ${GGML_PATH}")
endif()

# Add GGML as a subproject
message(STATUS "Building GGML from sources...")
add_subdirectory(${GGML_PATH} ggml_build)

# Get GGML include directory
set(GGML_INCLUDE_DIR ${GGML_PATH}/include)
if(NOT EXISTS ${GGML_INCLUDE_DIR})
    # If include is at the root of GGML
    set(GGML_INCLUDE_DIR ${GGML_PATH})
endif()

# Include directories
include_directories(${GGML_INCLUDE_DIR})

# Source files
set(SOURCES styletts_ggml.cpp)

#add_subdirectory(thirdparty)

# Create executable
add_executable(styletts_ggml ${SOURCES})

# Link libraries
#target_link_libraries(styletts_ggml ggml zip)
target_link_libraries(styletts_ggml ggml)

# Platform-specific options
if(APPLE)
    target_link_libraries(styletts_ggml "-framework Accelerate")
elseif(UNIX AND NOT APPLE)
    target_link_libraries(styletts_ggml pthread)
endif()

# Install target
install(TARGETS styletts_ggml DESTINATION bin)

# Get GGML git hash
execute_process(
    COMMAND git -C ${GGML_PATH} rev-parse --short HEAD
    OUTPUT_VARIABLE GGML_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "GGML Git commit: ${GGML_GIT_HASH}")
target_compile_definitions(styletts_ggml PRIVATE GGML_COMMIT_HASH="${GGML_GIT_HASH}")

# Optional targets
add_custom_target(install_deps
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Updating GGML submodule"
)

add_custom_target(test_model
    COMMAND python3 ${CMAKE_SOURCE_DIR}/convert_model.py --template --output test_model.bin
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating test model"
)

# Testing
enable_testing()
add_test(NAME basic_test 
         COMMAND styletts_ggml test_model.bin "Hello world"
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Build info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "GGML path: ${GGML_PATH}")

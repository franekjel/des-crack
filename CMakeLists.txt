cmake_minimum_required(VERSION 3.10)
project(des-crack LANGUAGES C CUDA)

include_directories(/opt/cuda/include)

set(CMAKE_COLOR_MAKEFILE ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})


include_directories(include)
file(GLOB SOURCES "main.c" "*.c" "*.cu" "*.h")
add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

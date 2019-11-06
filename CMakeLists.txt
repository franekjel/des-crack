cmake_minimum_required(VERSION 3.10)
project(des-crack LANGUAGES C)
enable_language(CUDA)
include_directories(/opt/cuda/include)

set(CMAKE_C_STANDARD 11)
set(CMAKE_COMPILER_IS_GNUCC TRUE)
set(CMAKE_C_FLAGS -mbmi -O2)
set(CMAKE_COLOR_MAKEFILE ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS -arch=sm_61;-G;-g;-rdc=true)

file(GLOB SOURCES "main.c" "*.c" "*.cu" "*.h")
add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

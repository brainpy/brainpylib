cmake_minimum_required(VERSION 3.12...3.18)
project(brainpylib LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
find_package(pybind11 REQUIRED)

include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ./pybind11/include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
        gpu_ops
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_event_csr_matvec.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_csr_matvec.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_event_info.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_event_sum.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_atomic_prod.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_atomic_sum.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/kernel_helper_descriptor.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_jitconn_event_matvec.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_jitconn_event_matvec_atomic.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_jitconn_matvec.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_jitconn_matvec_atomic.cu
        ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_jitconn_matmat.cu
)
install(TARGETS gpu_ops DESTINATION brainpylib)
pybind11_add_module(
        cpu_ops
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_event_prod.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_event_sum.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_atomic_prod.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_atomic_sum.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_jitconn_event_matvec.cpp
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_jitconn_event_matvec_atomic.cpp
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_jitconn_matvec.cpp
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_jitconn_matvec_atomic.cpp
)
install(TARGETS cpu_ops DESTINATION brainpylib)

# GPU op library
#if (BRAINPY_CUDA)
#  enable_language(CUDA)
#  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#  pybind11_add_module(
#    gpu_ops
#    ${CMAKE_CURRENT_LIST_DIR}/lib/event_sum_gpu.cc.cu
#    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
#  install(TARGETS gpu_ops DESTINATION brainpylib)
#else()
#   message(STATUS "Building without CUDA")
#   # CPU op library
#  pybind11_add_module(
#    cpu_ops
#    ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc
#    ${CMAKE_CURRENT_LIST_DIR}/lib/event_sum_cpu.cc
#    ${CMAKE_CURRENT_LIST_DIR}/lib/atomic_sum_cpu.cc
#    )
#  install(TARGETS cpu_ops DESTINATION brainpylib)
# endif()
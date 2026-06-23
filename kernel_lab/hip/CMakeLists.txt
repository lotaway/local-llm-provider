cmake_minimum_required(VERSION 3.28)

project(kernel_lab LANGUAGES CXX HIP)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
    "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

add_library(
    kernel_lab_core
    STATIC
    src/kernel_lab.hip
)

target_include_directories(
    kernel_lab_core
    PUBLIC
    include
)

target_link_libraries(
    kernel_lab_core
    PUBLIC
    ${TORCH_LIBRARIES}
)

add_library(
    kernel_lab_ext
    MODULE
    torch_ext/setup.cpp
)

target_link_libraries(
    kernel_lab_ext
    PRIVATE
    kernel_lab_core
    pybind11::module
)

add_executable(
    test_add
    tests/test_add.hip
)

target_link_libraries(
    test_add
    PRIVATE
    kernel_lab_core
)
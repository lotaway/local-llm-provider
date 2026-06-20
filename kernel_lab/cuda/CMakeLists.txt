cmake_minimum_required(VERSION 3.28)

project(kernel_lab LANGUAGES CXX CUDA)

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

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
    "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(
    kernel_lab_ext
    MODULE
    torch_ext/setup.cpp
    src/kernel_lab.cu
)

target_include_directories(
    kernel_lab_ext
    PRIVATE
    include
)

target_link_libraries(
    kernel_lab_ext
    PRIVATE
    ${TORCH_LIBRARIES}
    pybind11::module
)
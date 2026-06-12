cmake_minimum_required(VERSION 3.28)

project(kernel_lab LANGUAGES CXX HIP)

find_package(Python REQUIRED)
find_package(pybind11 REQUIRED)

add_library(
    kernel_lab_ext
    MODULE
    torch_ext/setup.cpp
    src/kernel_lab.hip
)

target_include_directories(
    kernel_lab_ext
    PRIVATE
    include
)
cmake_minimum_required(VERSION 3.28)

project(kernel_lab LANGUAGES CXX HIP)

include_directories(
    /opt/rocm/include
)

add_executable(
    kernel_lab
    kernel_lab_hip.cpp
)
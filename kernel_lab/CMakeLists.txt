cmake_minimum_required(VERSION 3.28)

project(kernel_lab LANGUAGES CXX HIP)

add_executable(
    kernel_lab
    kernel_lab_hip.cpp
)
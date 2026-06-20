#!/bin/bash
cmake -B ../build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
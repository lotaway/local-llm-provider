#!/bin/bash
# mamba create -n ai python=3.12.12 # Use it to create conda virtual environment in first time
cd "$(dirname "$0")"
source .env
mamba run -n ai python main.py

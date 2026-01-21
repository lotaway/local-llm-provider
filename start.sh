#!/bin/bash
cd "$(dirname "$0")"
source .env
mamba run -n python3.12 python main.py

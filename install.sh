#!/bin/bash
pip install -r requirements1.txt
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements2.txt
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python==0.3.16

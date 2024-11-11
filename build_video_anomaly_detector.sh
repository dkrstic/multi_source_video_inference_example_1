#!/bin/sh
g++ onnx2tensorrt.cpp main.cpp -std=c++17 -I/usr/include/ -I/usr/include/x86_64-linux-gnu/ -I/usr/local/cuda-12.0/targets/x86_64-linux/include/ `pkg-config --cflags --libs opencv` -lnvinfer -lnvonnxparser -L/usr/lib/ -L/usr/lib/x86_64-linux-gnu/ -lzmq -L/usr/local/cuda-12.0/lib64 -lcudart -o vdd

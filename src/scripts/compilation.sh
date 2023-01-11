#!/bin/bash

BUILDDIR="../../build"
WD=$PWD

cd $BUILDDIR/..
make all
make build/cnn_cuda_matrix_multiplication.o
cd $WD

echo "Compiling matrix_multiplication_benchmark.cu"
nvcc -ljpeg \
    matrix_multiplication_benchmark.cu \
    "$BUILDDIR/"cnn_cuda_matrix_multiplication.o \
    "$BUILDDIR/"cuda_utils.o \
    -o benchmark-matrix-multiplication

echo "Compiling convolution_benchmark.cu"
nvcc -ljpeg \
    convolution_benchmark.cu \
    "$BUILDDIR/"cnn_cuda_convolution.o \
    "$BUILDDIR/"cuda_utils.o \
    -o benchmark-convolution
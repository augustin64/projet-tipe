#include <stdio.h>
#include <stdlib.h>

#include "include/mnist.h"



unsigned int* cudaReadMnistLabels(char* filename) {
    FILE* ptr;

    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_items;
    unsigned int* labels;
    unsigned int* labels_cuda;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&number_of_items, sizeof(uint32_t), 1, ptr);
    number_of_items = swap_endian(number_of_items);

    unsigned char buffer[number_of_items];
    fread(buffer, sizeof(unsigned char), number_of_items, ptr);

    labels = (unsigned int*)malloc(sizeof(unsigned int)*number_of_items);

    for (int i=0; i < (int)number_of_items; i++) {
        labels[i] = (unsigned int)buffer[i];
    }

    cudaMalloc(&labels_cuda, (size_t)sizeof(labels));
    cudaMemcpy(labels_cuda, &labels, sizeof(labels), cudaMemcpyHostToDevice);
    free(labels);
    return labels_cuda;
}


void check_cuda_compatibility() {
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
    } else {
        printf("CUDA-capable device is detected\n");
    }
}
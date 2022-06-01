#include <stdio.h>
#include <stdlib.h>

#include "include/cuda_utils.h"

int*** copy_images_cuda(int*** images, int nb_images, int width, int height) {
    int*** images_cuda;
    cudaMalloc(&images_cuda, (size_t)sizeof(int**)*nb_images);
    cudaMemcpy(images_cuda, &images, (size_t)sizeof(int**)*nb_images, cudaMemcpyHostToDevice);

    for (int i=0; i < nb_images; i++) {
        cudaMalloc(&images_cuda[i], sizeof(int**)*nb_images);
        cudaMemcpy(images_cuda[i], &images[i], sizeof(int**)*nb_images, cudaMemcpyHostToDevice);
            for (int j=0; j < height; j++) {
                cudaMalloc((int**)&images_cuda[i][j], sizeof(int*)*width);
                cudaMemcpy(images_cuda[i][j], &images[i][j], sizeof(int*)*width, cudaMemcpyHostToDevice);
            }
    }
    return images_cuda;
}


unsigned int* copy_labels_cuda(unsigned int* labels) {
    unsigned int* labels_cuda;
    cudaMalloc(&labels_cuda, (size_t)sizeof(labels));
    cudaMemcpy(labels_cuda, &labels, sizeof(labels), cudaMemcpyHostToDevice);
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
#include <stdio.h>
#include <stdlib.h>


int*** copy_images_cuda(int*** images, int nb_images, int width, int height) {
    int*** images_cuda;
    cudaMalloc((int****)&images_cuda, sizeof(int**)*nb_images);
    cudaMemcpy((int****)&images_cuda, sizeof(int**)*nb_images, images);

    for (int i=0; i < nb_images; i++) {
        cudaMalloc((int***)&images_cuda[i], sizeof(int**)*nb_images);
        cudaMemcpy((int***)&images_cuda[i], sizeof(int**)*nb_images, images[i]);
            for (int j=0; j < height; j++) {
                cudaMalloc((int**)&images_cuda[i][j], sizeof(int*)*width);
                cudaMemcpy((int**)&images_cuda[i][j], sizeof(int*)*width, images[i][j]);
            }
    }
    return images_cuda;
}




unsigned int* copy_labels_cuda(unsigned int* labels) {
    unsigned int* labels_cuda;
    cudaMalloc((unsigned int**)&labels_cuda, sizeof(labels));
    cudaMemcpy((unsigned int**)&labels_cuda, sizeof(labels), labels);
    return labels_cuda;
}
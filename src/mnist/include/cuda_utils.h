#include <stdio.h>
#include <stdlib.h>

#ifndef DEF_CUDA_UTILS_H
#define DEF_CUDA_UTILS_H

int*** copy_images_cuda(int*** images, int nb_images, int width, int height);
unsigned int* copy_labels_cuda(unsigned int* labels);
void check_cuda_compatibility();

#endif
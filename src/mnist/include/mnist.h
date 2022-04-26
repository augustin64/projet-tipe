#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef DEF_MNIST_H
#define DEF_MNIST_H

uint32_t swap_endian(uint32_t val);
uint32_t read_mnist_labels_nb_images(char* filename);
int** read_image(unsigned int width, unsigned int height, FILE* ptr);
int* read_mnist_images_parameters(char* filename);
int* read_mnist_labels_parameters(char* filename);
int*** read_mnist_images(char* filename);
unsigned int* read_mnist_labels(char* filename);

#endif
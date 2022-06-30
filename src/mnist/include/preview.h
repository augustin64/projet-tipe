#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef DEF_PREVIEW_H
#define DEF_PREVIEW_H

/*
* Affiche un chiffre de taille width x height
*/
void print_image(unsigned int width, unsigned int height, int** image);

/* 
* Prévisualise un chiffre écrit à la main
*/
void preview_images(char* images_file, char* labels_file);

int main(int argc, char *argv[]);

#endif

               
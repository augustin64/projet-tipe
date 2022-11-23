#ifndef JPEG_DEF_H
#define JPEG_DEF_H

// keep images in ram vs re-read and decompress each time
// #define STORE_IMAGES_TO_RAM
// Note: in use dataset is 90Go once decompressed, use with caution

/*
* Struct used to describe a single JPEG image
*/
typedef struct imgRawImage {
    unsigned int numComponents; // Nombre de composantes (S, R, G, B, ...)
    unsigned long int width, height; // Taille de l'image
    unsigned char* lpData; // Données de l'image
} imgRawImage;

/*
* Struct used to describe a full JPEG dataset
*/
typedef struct jpegDataset {
    unsigned int numComponents; // Nombre de composantes (S, R, G, B, ...)
    unsigned int numImages; // Nombre d'images (fichiers)
    unsigned int numCategories; // Nombre de catégories (dossiers)

    unsigned int width; // Largeur des images
    unsigned int height; // Hauteur des images

    unsigned int* labels; // Labels
    unsigned char** images; // Images en cache, vaut NULL si STORE_IMAGES_TO_RAM n'est pas défini
	char** fileNames; // Noms de fichiers
} jpegDataset;

/*
* Load a single JPEG image from its location
*/
imgRawImage* loadJpegImageFile(char* lpFilename);

/*
* Load a complete dataset from its path
*/
jpegDataset* loadJpegDataset(char* folderPath);

/*
* Count the number of directories available directly under a specific path
*/
unsigned int countDirectories(char* path);

/*
* Counts recursively the number of files available in a directory (and all subdirs)
*/
unsigned int countFiles(char* path);

/*
* Adds the names of files available under a directory to a char* array
*/
void addFilenamesToArray(char* path, char** array, int* index);

/*
* Free a dataset
*/
void free_dataset(jpegDataset* dataset);

/*
* Returns the value of the label for a given directory
* (Generated with Python)
*/
unsigned int getLabel(char* string);

#endif
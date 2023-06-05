#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <jerror.h>
#include <jpeglib.h>

#include "../common/include/utils.h"
#include "../common/include/colors.h"

#include "include/jpeg.h"

// How to load a JPEG using libjpeg: https://www.tspi.at/2020/03/20/libjpegexample.html
imgRawImage* loadJpegImageFile(char* lpFilename) {
    struct jpeg_decompress_struct info;
    struct jpeg_error_mgr err;

    imgRawImage* lpNewImage;

    unsigned long int imgWidth, imgHeight;
    int numComponents;

    unsigned long int dwBufferBytes;
    unsigned char* lpData;

    unsigned char* lpRowBuffer[1];

    FILE* fHandle;

    fHandle = fopen(lpFilename, "rb");
    if(fHandle == NULL) {
        fprintf(stderr, "%s:%u: Failed to read file %s\n", __FILE__, __LINE__, lpFilename);
        return NULL; /* ToDo */
    }

    info.err = jpeg_std_error(&err);
    jpeg_create_decompress(&info);

    jpeg_stdio_src(&info, fHandle);
    jpeg_read_header(&info, TRUE);

    jpeg_start_decompress(&info);
    imgWidth = info.output_width;
    imgHeight = info.output_height;
    numComponents = info.num_components;

    #ifdef DEBUG
        fprintf(
            stderr,
            "%s:%u: Reading JPEG with dimensions %lu x %lu and %u components\n",
            __FILE__, __LINE__,
            imgWidth, imgHeight, numComponents
        );
    #endif

    dwBufferBytes = imgWidth * imgHeight * 3; /* We only read RGB, not A */
    lpData = (unsigned char*)malloc(sizeof(unsigned char)*dwBufferBytes);

    lpNewImage = (imgRawImage*)malloc(sizeof(imgRawImage));
    lpNewImage->numComponents = numComponents;
    lpNewImage->width = imgWidth;
    lpNewImage->height = imgHeight;
    lpNewImage->lpData = lpData;

    /* Read scanline by scanline */
    while(info.output_scanline < info.output_height) {
        lpRowBuffer[0] = (unsigned char *)(&lpData[3*info.output_width*info.output_scanline]);
        jpeg_read_scanlines(&info, lpRowBuffer, 1);
    }

    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    fclose(fHandle);

    return lpNewImage;
}


int storeJpegImageFile(imgRawImage* lpImage, char* lpFilename) {
	struct jpeg_compress_struct info;
	struct jpeg_error_mgr err;

	unsigned char* lpRowBuffer[1];

	FILE* fHandle;

	fHandle = fopen(lpFilename, "wb");
	if(fHandle == NULL) {
		#ifdef DEBUG
			fprintf(stderr, "%s:%u Failed to open output file %s\n", __FILE__, __LINE__, lpFilename);
		#endif
		return 1;
	}

	info.err = jpeg_std_error(&err);
	jpeg_create_compress(&info);

	jpeg_stdio_dest(&info, fHandle);

	info.image_width = lpImage->width;
	info.image_height = lpImage->height;
	info.input_components = 3;
	info.in_color_space = JCS_RGB;

	jpeg_set_defaults(&info);
	jpeg_set_quality(&info, 100, TRUE);

	jpeg_start_compress(&info, TRUE);

	/* Write every scanline ... */
	while(info.next_scanline < info.image_height) {
		lpRowBuffer[0] = &(lpImage->lpData[info.next_scanline * (lpImage->width * 3)]);
		jpeg_write_scanlines(&info, lpRowBuffer, 1);
	}

	jpeg_finish_compress(&info);
	fclose(fHandle);

	jpeg_destroy_compress(&info);
	return 0;
}


jpegDataset* loadJpegDataset(char* folderPath) {
    jpegDataset* dataset = (jpegDataset*)malloc(sizeof(jpegDataset));
    imgRawImage* image;

    // We start by counting the number of images and categories
    dataset->numCategories = countDirectories(folderPath);
	dataset->numImages = countFiles(folderPath);

	dataset->images = NULL;
	dataset->labels = (unsigned int*)malloc(sizeof(unsigned int)*dataset->numImages);
	dataset->fileNames = (char**)malloc(sizeof(char*)*dataset->numImages);

	DIR* dirp;
    struct dirent* entry;
    char* concatenated_path;
    int index = 0;
    int prev_index = index;
	
	dirp = opendir(folderPath);
    while ((entry = readdir(dirp)) != NULL) {
        if (strcmp(entry->d_name, ".")&&strcmp(entry->d_name, "..")) {
            if (entry->d_type == DT_DIR) {
                prev_index = index;
                concatenated_path = malloc(strlen(folderPath)+strlen(entry->d_name)+2);
                sprintf(concatenated_path, "%s/%s", folderPath, entry->d_name);
                addFilenamesToArray(concatenated_path, dataset->fileNames, &index);
                for (int i=prev_index; i < index; i++) {
                    dataset->labels[i] = getLabel(entry->d_name);
                }
                free(concatenated_path);
            }
        }
    }
    dataset->images = (unsigned char**)malloc(sizeof(unsigned char*)*dataset->numImages);
    for (int i=0; i < (int)dataset->numImages; i++) {
        dataset->images[i] = NULL;
    }

    // Lecture des caractéristiques des images
    image = loadJpegImageFile(dataset->fileNames[0]);
    dataset->width = image->width;
    dataset->height = image->height;
    dataset->numComponents = image->numComponents;

    free(image->lpData);
    free(image);

	closedir(dirp);
	return dataset;
}

unsigned int countDirectories(char* path) {
    unsigned int directories = 0;
    DIR* dirp;
    struct dirent* entry;

    dirp = opendir(path);
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            directories++;
        }
    }
    closedir(dirp);
    return directories;
}

unsigned int countFiles(char* path) {
    unsigned int files = 0;
    DIR* dirp;
	char* next_dir;
    struct dirent* entry;

    dirp = opendir(path);
    while ((entry = readdir(dirp)) != NULL) {
        if (strcmp(entry->d_name, ".")&&strcmp(entry->d_name, "..")) {
            if (entry->d_type == DT_REG) {
                files++;
            } else if (entry->d_type == DT_DIR) {
                next_dir = (char*)malloc(strlen(path)+strlen(entry->d_name)+2);
                sprintf(next_dir, "%s/%s", path, entry->d_name);
                files += countFiles(next_dir);
                free(next_dir);
            }
        }
    }
	closedir(dirp);
	return files;
}

void addFilenamesToArray(char* path, char** array, int* index) {
    int i = *index;

    DIR* dirp;
    struct dirent* entry;
    char* filename;

    dirp = opendir(path); /* There should be error handling after this */
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type == DT_REG) { /* If the entry is a regular file */
            filename = (char*)malloc(strlen(path)+strlen(entry->d_name)+2);
            sprintf(filename, "%s/%s", path, entry->d_name);
            array[i] = filename;
            i++;
        }
    }
    *index = i;
    closedir(dirp);
}

void free_dataset(jpegDataset* dataset) {
    for (int i=0; i < (int)dataset->numImages; i++) {
        free(dataset->fileNames[i]);
    }
    free(dataset->fileNames);
    free(dataset->labels);
    free(dataset->images);
    free(dataset);
}

unsigned int getLabel(char* string) {
    if (!strcmp(string, "Alabama")) {
        return 0;
    } if (!strcmp(string, "Alaska")) {
        return 1;
    } if (!strcmp(string, "Arizona")) {
        return 2;
    } if (!strcmp(string, "Arkansas")) {
        return 3;
    } if (!strcmp(string, "California")) {
        return 4;
    } if (!strcmp(string, "Colorado")) {
        return 5;
    } if (!strcmp(string, "Connecticut")) {
        return 6;
    } if (!strcmp(string, "Delaware")) {
        return 7;
    } if (!strcmp(string, "Florida")) {
        return 8;
    } if (!strcmp(string, "Georgia")) {
        return 9;
    } if (!strcmp(string, "Hawaii")) {
        return 10;
    } if (!strcmp(string, "Idaho")) {
        return 11;
    } if (!strcmp(string, "Illinois")) {
        return 12;
    } if (!strcmp(string, "Indiana")) {
        return 13;
    } if (!strcmp(string, "Iowa")) {
        return 14;
    } if (!strcmp(string, "Kansas")) {
        return 15;
    } if (!strcmp(string, "Kentucky")) {
        return 16;
    } if (!strcmp(string, "Louisiana")) {
        return 17;
    } if (!strcmp(string, "Maine")) {
        return 18;
    } if (!strcmp(string, "Maryland")) {
        return 19;
    } if (!strcmp(string, "Massachusetts")) {
        return 20;
    } if (!strcmp(string, "Michigan")) {
        return 21;
    } if (!strcmp(string, "Minnesota")) {
        return 22;
    } if (!strcmp(string, "Mississippi")) {
        return 23;
    } if (!strcmp(string, "Missouri")) {
        return 24;
    } if (!strcmp(string, "Montana")) {
        return 25;
    } if (!strcmp(string, "Nebraska")) {
        return 26;
    } if (!strcmp(string, "Nevada")) {
        return 27;
    } if (!strcmp(string, "New Hampshire")) {
        return 28;
    } if (!strcmp(string, "New Jersey")) {
        return 29;
    } if (!strcmp(string, "New Mexico")) {
        return 30;
    } if (!strcmp(string, "New York")) {
        return 31;
    } if (!strcmp(string, "North Carolina")) {
        return 32;
    } if (!strcmp(string, "North Dakota")) {
        return 33;
    } if (!strcmp(string, "Ohio")) {
        return 34;
    } if (!strcmp(string, "Oklahoma")) {
        return 35;
    } if (!strcmp(string, "Oregon")) {
        return 36;
    } if (!strcmp(string, "Pennsylvania")) {
        return 37;
    } if (!strcmp(string, "Rhode Island")) {
        return 38;
    } if (!strcmp(string, "South Carolina")) {
        return 39;
    } if (!strcmp(string, "South Dakota")) {
        return 40;
    } if (!strcmp(string, "Tennessee")) {
        return 41;
    } if (!strcmp(string, "Texas")) {
        return 42;
    } if (!strcmp(string, "Utah")) {
        return 43;
    } if (!strcmp(string, "Vermont")) {
        return 44;
    } if (!strcmp(string, "Virginia")) {
        return 45;
    } if (!strcmp(string, "Washington")) {
        return 46;
    } if (!strcmp(string, "West Virginia")) {
        return 47;
    } if (!strcmp(string, "Wisconsin")) {
        return 48;
    } if (!strcmp(string, "Wyoming")) {
        return 49;
    }
    printf_warning("Catégorie non reconnue ");
    printf("%s\n", string);
    return -1; // Dossier non reconnu
}
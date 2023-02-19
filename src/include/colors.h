#include <stdio.h>

#ifndef DEF_COLORS_H
#define DEF_COLORS_H

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#ifdef __CUDACC__
extern "C"
#endif
/*
* Affiche le texte demandé, précédé d'un [ERROR] en rouge
*/
void printf_error(char* string);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Affiche le texte demandé, précédé d'un [WARNING] en orange
*/
void printf_warning(char* string);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Affiche le texte demandé, précédé d'un [INFO] en bleu
*/
void printf_info(char* string);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Affiche un timing en heures minutes secondes millisecondes en limitant la précision aux deux unités les plus significatives
*/
void printf_time(float time);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Affiche une quantité de mémoire de manière humainement lisible
*/
void printf_memory(size_t size);
#endif
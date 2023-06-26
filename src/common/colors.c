#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "include/colors.h"

void printf_error(char* string) {
    printf(BOLDRED "[ ERROR ]" RESET " %s", string);
}

void printf_warning(char* string) {
    printf(BOLDYELLOW "[WARNING]" RESET " %s", string);
}

void printf_info(char* string) {
    printf(BOLDBLUE "[ INFO  ]" RESET " %s", string);
}

void printf_time(clock_t time) {
    double real_time = (double) time / CLOCKS_PER_SEC;

    int hours = real_time/3600;
    int minutes = ((int)real_time %3600)/60;
    int seconds = ((int)real_time) %60;
    int milliseconds = (real_time - (int)real_time)*1000;

    if (hours != 0) {
        printf("%dh %dmn", hours, minutes);
    } else if (minutes != 0) {
        printf("%dmn %ds", minutes, seconds);
    } else if (seconds != 0) {
        printf("%ds %dms", seconds, milliseconds);
    } else {
        printf("%dms", milliseconds);
    }
}

void printf_memory(size_t size) {
    size_t gigabytes = size/(1024*1024*1024);
    size_t megabytes = size/(1024*1024) %1024;
    size_t kilobytes = size/1024 %1024;
    size_t bytes = size %1024;

    bool is_null = true;

    if (gigabytes != 0) {
        printf("%ldGB", gigabytes);
        is_null = false;
    }
    if (megabytes != 0) {
        if (!is_null) {
            printf(" ");
        }
        printf("%ldMB", megabytes);
        is_null = false;
    }
    if (kilobytes != 0) {
        if (!is_null) {
            printf(" ");
        }
        printf("%ldkB", kilobytes);
        is_null = false;
    }
    if (bytes != 0) {
        if (!is_null) {
            printf(" ");
        }
        printf("%ldB", bytes);
        is_null = false;
    }
    if (is_null) {
        printf("OB");
    }
}
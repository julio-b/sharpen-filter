#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

// malloc and memcpy wrappers for nvprof
void *nmalloc(size_t size);
void *nmemcpy(void *dest, const void *src, size_t n);

#endif

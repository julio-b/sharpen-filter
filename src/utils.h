#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define EXIT_WITH_ERR_MSG(ERR_MSG) {\
	fprintf(stderr, ERR_MSG); \
	exit(EXIT_FAILURE); \
	}

// malloc and memcpy wrappers for nvprof
void *nmalloc(size_t size);
void *nmemcpy(void *dest, const void *src, size_t n);

#endif

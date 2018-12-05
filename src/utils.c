#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvToolsExt.h>
#include "utils.h"

void *nmalloc(size_t size)
{
	nvtxRangePush("cpu malloc");
	void *p = malloc(size);
	nvtxRangePop();
	return p;
}

void *nmemcpy(void *dest, const void *src, size_t n)
{
	nvtxRangePush("cpu memcpy");
	void *p = memcpy(dest, src, n);
	nvtxRangePop();
	return p;

}

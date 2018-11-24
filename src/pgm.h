#include <stdbool.h>

struct pgm {
	int width;
	int height;
	int maxval;
	int *pixels;
};

struct pgm *read_pgm(char *filename);
bool save_pgm(struct pgm *img, char *filename);

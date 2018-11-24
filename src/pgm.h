#include <stdbool.h>
#include <stdlib.h>

struct pgm {
	int width;
	int height;
	int maxval;
	int *pixels;
};

struct pgm *read_pgm(char *filename);
struct pgm *copy_pgm(struct pgm *img);
bool save_pgm(struct pgm *img, char *filename);

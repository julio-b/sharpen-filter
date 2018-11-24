struct pgm {
	int width;
	int height;
	int maxval;
	int *pixels;
};

struct pgm *read_pgm(char *filename);

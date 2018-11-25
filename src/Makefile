.PHONY : clean run all build
.DEFAULT_GOAL := run

SRC    = pgm.c
CUSRC  = sharpen-filter.cu
OBJ    = $(SRC:.c=.o) $(CUSRC:.cu=.o)
LIBS   = 
EXE    = sharpen-filter

#CC      = /usr/bin/gcc
CC      = /opt/cuda/bin/nvcc
NVCC    = /opt/cuda/bin/nvcc
#CFLAGS  = -Wall
CFLAGS  = 
LIBPATH = -L.
LDFLAGS = -o $(EXE) $(LIBPATH) $(LIBS)
CFDEBUG = -Wall -g -DDEBUG $(LDFLAGS)
RM      = /bin/rm -f
OUTIMGS = /tmp/sharpen_baboon.*.pgm

all: build

build: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(LDFLAGS) $+

%.o: %.cu
	$(NVCC) -c $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

run: build
	./sharpen-filter && feh $(OUTIMGS)

clean:
	$(RM) $(OBJ) $(EXE) $(OUTIMGS) core a.out

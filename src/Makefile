.PHONY : clean run all build
.DEFAULT_GOAL := nvprof

SRC    = pgm.c utils.c
CUSRC  = sharpen-filter.cu
OBJ    = $(SRC:.c=.o) $(CUSRC:.cu=.o)
LIBS   = -lnvToolsExt
EXE    = sharpen-filter

#CC      = /usr/bin/gcc
CC      = /opt/cuda/bin/nvcc
NVCC    = /opt/cuda/bin/nvcc
#CFLAGS  = -Wall
CFLAGS  = 
LIBPATH = -L. -L/opt/cuda/lib64
LDFLAGS = -o $(EXE) $(LIBPATH) $(LIBS)
CFDEBUG = -Wall -g -DDEBUG $(LDFLAGS)
RM      = /bin/rm -f
OUTIMGS = baboon_sharpen.*.pgm

all: build

build: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(LDFLAGS) $+

%.o: %.cu
	$(NVCC) -c $< -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

run: build
	./$(EXE) && feh $(OUTIMGS)

nvprof: build
	nvprof ./$(EXE)

clean:
	$(RM) $(OBJ) $(EXE) $(OUTIMGS) core a.out

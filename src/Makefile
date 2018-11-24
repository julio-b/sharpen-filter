.PHONY : clean run
.DEFAULT_GOAL := run

INCL   = 
SRC    = pgm.c sharpen-filter.c
OBJ    = $(SRC:.c=.o)
LIBS   = 
EXE    = sharpen-filter

CC      = /usr/bin/gcc
CFLAGS  = -Wall
LIBPATH = -L.
LDFLAGS = -o $(EXE) $(LIBPATH) $(LIBS)
CFDEBUG = -Wall -g -DDEBUG $(LDFLAGS)
RM      = /bin/rm -f

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

$(EXE): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ)

$(OBJ): $(INCL)

debug:
	$(CC) $(CFDEBUG) $(SRC)

run: $(EXE)
	./$(EXE)

clean:
	$(RM) $(OBJ) $(EXE) core a.out

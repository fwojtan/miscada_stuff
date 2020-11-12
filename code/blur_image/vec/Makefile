CC = icc

CFLAGS = -I. -std=c11 -O2 -qopt-report-phase=vec -qopt-report=3
LIBS = -lm

DEPS = proto.h
OBJ  = main.o io.o filters.o
EXE = blur

.PHONY: all clean

all: $(EXE) Makefile

$(EXE): $(OBJ) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBS)

clean:
	-rm -f $(OBJ) $(EXE)

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mprefer-vector-width=512 -std=gnu11 -Wall -pthread
MAXB    ?= 32
DEPS    := src/common.h src/threads.h src/bench.h

all: gemma3t gemma bitnet gemma4

gemma3t: src/gemma3t.c src/q4.h src/ternary.h src/gemma3.h $(DEPS)
	$(CC) $(CFLAGS) -DMAXB=$(MAXB) -Isrc -o $@ $< -lm
gemma: src/gemma.c src/q4.h src/gemma3.h $(DEPS)
	$(CC) $(CFLAGS) -DMAXB=$(MAXB) -Isrc -o $@ $< -lm
bitnet: src/bitnet.c src/ternary.h $(DEPS)
	$(CC) $(CFLAGS) -DMAXB=$(MAXB) -Isrc -o $@ $< -lm
gemma4: src/gemma4.c $(DEPS)   # OpenMP only for load-time quantization
	$(CC) $(CFLAGS) -fopenmp -DMAXB=$(MAXB) -Isrc -o $@ $< -lm

clean:
	rm -f gemma3t gemma bitnet gemma4

.PHONY: all clean

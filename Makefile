CC := clang
CFLAGS := -O3 -mcpu=native -std=c11 -Wall -Wextra -Wpedantic -Isrc
LDFLAGS := -lm -framework Metal -framework Foundation
SRC := src/gguf.c src/model.c src/tokenizer.c src/forward.c src/sampler.c src/metal.m

.PHONY: all clean test
all: dsm
dsm: $(SRC) src/main.c metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ $(SRC) src/main.c $(LDFLAGS)
test_gguf: src/gguf.c tests/test_gguf.c
	$(CC) $(CFLAGS) -o $@ src/gguf.c tests/test_gguf.c
test: test_gguf
	./test_gguf
clean:
	rm -f dsm test_gguf

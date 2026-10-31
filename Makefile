CC := clang
CFLAGS := -O3 -mcpu=native -std=c11 -Wall -Wextra -Wpedantic -Isrc
LDFLAGS := -lm -framework Metal -framework Foundation
SRC := src/gguf.c src/quants.c src/model.c src/tokenizer.c src/forward.c src/sampler.c src/metal.m

.PHONY: all clean test check-quants
all: dsm
dsm: $(SRC) src/main.c metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ $(SRC) src/main.c $(LDFLAGS)
test_gguf: src/gguf.c tests/test_gguf.c
	$(CC) $(CFLAGS) -o $@ src/gguf.c tests/test_gguf.c
test: test_gguf
	./test_gguf
check-quants: tools/ggml_dequant_check.c src/gguf.c src/quants.c
	$(CC) $(CFLAGS) -I../llama.cpp/ggml/include -o $@ tools/ggml_dequant_check.c src/gguf.c src/quants.c -L../llama.cpp/build/bin -lggml-base -Wl,-rpath,@loader_path/../llama.cpp/build/bin -lm
clean:
	rm -f dsm test_gguf

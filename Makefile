CC := clang
CFLAGS := -O3 -mcpu=native -std=c11 -Wall -Wextra -Wpedantic -Isrc
LDFLAGS := -lm -lpthread -framework Metal -framework Foundation
SRC := src/gguf.c src/quants.c src/model.c src/tokenizer.c src/forward.c src/sampler.c src/json.c src/server.c src/metal.m

.PHONY: all clean test check-quants check-metal-gemv check-prefill check-rewind check-attn bench-prefill llama-ref check-ref
all: dsm
dsm: $(SRC) src/main.c metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ $(SRC) src/main.c $(LDFLAGS)
test_gguf: src/gguf.c tests/test_gguf.c
	$(CC) $(CFLAGS) -o $@ src/gguf.c tests/test_gguf.c
test_quants: src/gguf.c src/quants.c tests/test_quants.c
	$(CC) $(CFLAGS) -o $@ src/gguf.c src/quants.c tests/test_quants.c -lm
test_server_parse: $(SRC) tests/test_server_parse.c
	$(CC) $(CFLAGS) -o $@ $(SRC) tests/test_server_parse.c $(LDFLAGS)
gguf-inspect: tools/gguf_inspect.c src/gguf.c src/quants.c src/model.c
	$(CC) $(CFLAGS) -o $@ tools/gguf_inspect.c src/gguf.c src/quants.c src/model.c -lm
test: test_gguf test_quants test_server_parse
	./test_gguf
	./test_quants
	./test_server_parse
check-quants: tools/ggml_dequant_check.c src/gguf.c src/quants.c
	$(CC) $(CFLAGS) -I../llama.cpp/ggml/include -o $@ tools/ggml_dequant_check.c src/gguf.c src/quants.c -L../llama.cpp/build/bin -lggml-base -Wl,-rpath,@loader_path/../llama.cpp/build/bin -lm
check-metal-gemv: tools/metal_gemv_check.m $(SRC) metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ tools/metal_gemv_check.m $(SRC) $(LDFLAGS)
check-prefill: tools/prefill_check.c $(SRC) metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ tools/prefill_check.c $(SRC) $(LDFLAGS)
check-rewind: tools/rewind_check.c $(SRC) metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ tools/rewind_check.c $(SRC) $(LDFLAGS)
check-attn: tools/attn_check.m metal/dsm.metal
	$(CC) -O2 -fobjc-arc -Wall -o $@ tools/attn_check.m -framework Metal -framework Foundation
bench-prefill: tools/prefill_bench.c $(SRC) metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ tools/prefill_bench.c $(SRC) $(LDFLAGS)
llama-ref: tools/llama_ref.c
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -I/opt/homebrew/opt/llama.cpp/include -I/opt/homebrew/Cellar/ggml/0.15.3/include -o $@ tools/llama_ref.c -L/opt/homebrew/opt/llama.cpp/lib -lllama -L/opt/homebrew/Cellar/ggml/0.15.3/lib -lggml -lggml-base -Wl,-rpath,/opt/homebrew/opt/llama.cpp/lib -Wl,-rpath,/opt/homebrew/Cellar/ggml/0.15.3/lib -lm
check-ref: tools/ref_check.c $(SRC) metal/dsm.metal
	$(CC) $(CFLAGS) -o $@ tools/ref_check.c $(SRC) $(LDFLAGS)
clean:
	rm -f dsm test_gguf test_quants test_server_parse check-quants check-metal-gemv check-prefill check-rewind check-attn bench-prefill llama-ref check-ref

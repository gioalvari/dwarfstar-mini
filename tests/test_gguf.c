#include "gguf.h"

#include <stdio.h>

int main(int argc,char **argv){const char *path=argc>1?argv[1]:"models/Qwen3-1.7B-Q8_0.gguf";gguf_file f;if(!gguf_open(&f,path)){fprintf(stderr,"%s\n",f.error);return 1;}const gguf_tensor *emb=gguf_tensor_find(&f,"token_embd.weight"),*q=gguf_tensor_find(&f,"blk.0.attn_q.weight");int ok=f.n_tensors==310&&emb&&q&&emb->dims[0]==2048&&emb->dims[1]==151936&&emb->type==GGML_TYPE_Q8_0&&q->dims[0]==2048&&q->dims[1]==2048;printf("GGUF v%u: %llu tensors; token_embd=(%llu,%llu), blk.0.q=(%llu,%llu)\n",f.version,(unsigned long long)f.n_tensors,(unsigned long long)emb->dims[0],(unsigned long long)emb->dims[1],(unsigned long long)q->dims[0],(unsigned long long)q->dims[1]);gguf_close(&f);return ok?0:1;}

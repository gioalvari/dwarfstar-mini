/* DSM-vs-llama.cpp logit comparator.  See tools/llama_ref.c for DSMREF v2. */
#include "forward.h"
#include "metal.h"
#include "model.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REF_VERSION = 2, REF_HEADER_BYTES = 32, TOP_K = 5 };

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab;
    uint32_t n_positions;
    uint32_t record_bytes;
    uint32_t reserved;
} ref_header;

typedef struct {
    double kl;
    float max_abs;
    uint32_t ref_top1;
    uint32_t dsm_top1;
    uint32_t ref_top5[TOP_K];
    uint32_t dsm_top5[TOP_K];
} comparison;

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s MODEL.gguf IDS.txt LLAMA.bin [--backend metal|cpu]\n"
            "       [--floor F] [--prefill] [--dump-dsm DSM.bin | --dsm-dump DSM.bin]\n",
            argv0);
    fprintf(stderr, "       %s --compare-dumps REF.bin TEST.bin [--floor F]\n", argv0);
}

static bool append_id(int32_t **ids, size_t *len, size_t *cap, long value) {
    if (value < 0 || value > INT32_MAX) return false;
    if (*len == *cap) {
        size_t next = *cap ? *cap * 2 : 256;
        int32_t *replacement = realloc(*ids, next * sizeof(**ids));
        if (!replacement) return false;
        *ids = replacement;
        *cap = next;
    }
    (*ids)[(*len)++] = (int32_t)value;
    return true;
}

static bool read_ids(const char *path, int32_t **ids, size_t *len) {
    FILE *file = fopen(path, "r");
    size_t cap = 0;
    if (!file) return false;
    for (;;) {
        int character;
        do character = fgetc(file); while (character == '[' || character == ']' ||
                                            character == ',' || character == ' ' ||
                                            character == '\t' || character == '\r' ||
                                            character == '\n');
        if (character == EOF) break;
        ungetc(character, file);
        char text[64], *end = NULL;
        if (fscanf(file, "%63[-0123456789]", text) != 1) {
            fclose(file);
            return false;
        }
        errno = 0;
        long value = strtol(text, &end, 10);
        if (errno || !end || *end || !append_id(ids, len, &cap, value)) {
            fclose(file);
            return false;
        }
    }
    bool ok = !ferror(file);
    fclose(file);
    return ok;
}

static bool valid_header(const ref_header *header) {
    return !memcmp(header->magic, "DSMREF1", 8) &&
           header->version == REF_VERSION &&
           header->header_bytes == REF_HEADER_BYTES && header->vocab &&
           header->record_bytes == sizeof(uint32_t) + header->vocab * sizeof(float);
}

static uint32_t argmax(const float *values, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i)
        if (values[i] > values[best]) best = i;
    return best;
}

static void top_k(const float *values, uint32_t n, uint32_t out[TOP_K]) {
    for (uint32_t k = 0; k < TOP_K; ++k) out[k] = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t k = TOP_K;
        while (k && (out[k - 1] == UINT32_MAX || values[i] > values[out[k - 1]])) --k;
        if (k == TOP_K) continue;
        memmove(out + k + 1, out + k, (TOP_K - k - 1) * sizeof(*out));
        out[k] = i;
    }
}

static uint32_t overlap(const uint32_t a[TOP_K], const uint32_t b[TOP_K]) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < TOP_K; ++i)
        for (uint32_t j = 0; j < TOP_K; ++j)
            if (a[i] == b[j]) {
                ++result;
                break;
            }
    return result;
}

static comparison compare(const float *reference, const float *dsm, uint32_t vocab) {
    comparison result = {0};
    double ref_max = -INFINITY, dsm_max = -INFINITY, ref_sum = 0, dsm_sum = 0;
    for (uint32_t i = 0; i < vocab; ++i) {
        if (reference[i] > ref_max) ref_max = reference[i];
        if (dsm[i] > dsm_max) dsm_max = dsm[i];
        float difference = fabsf(reference[i] - dsm[i]);
        if (difference > result.max_abs) result.max_abs = difference;
    }
    for (uint32_t i = 0; i < vocab; ++i) {
        ref_sum += exp((double)reference[i] - ref_max);
        dsm_sum += exp((double)dsm[i] - dsm_max);
    }
    for (uint32_t i = 0; i < vocab; ++i) {
        double p = exp((double)reference[i] - ref_max) / ref_sum;
        double q = exp((double)dsm[i] - dsm_max) / dsm_sum;
        result.kl += p * log(p / q);
    }
    result.ref_top1 = argmax(reference, vocab);
    result.dsm_top1 = argmax(dsm, vocab);
    top_k(reference, vocab, result.ref_top5);
    top_k(dsm, vocab, result.dsm_top5);
    return result;
}

static int compare_kl(const void *left, const void *right) {
    const double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static int compare_dumps(const char *reference_path, const char *test_path,
                         double floor) {
    FILE *reference_file = fopen(reference_path, "rb");
    FILE *test_file = fopen(test_path, "rb");
    ref_header reference_header, test_header;
    float *reference = NULL, *test = NULL;
    double *kls = NULL;
    int status = 1;
    if (!reference_file || !test_file ||
        fread(&reference_header, sizeof(reference_header), 1, reference_file) != 1 ||
        fread(&test_header, sizeof(test_header), 1, test_file) != 1 ||
        !valid_header(&reference_header) || !valid_header(&test_header) ||
        reference_header.vocab != test_header.vocab ||
        reference_header.n_positions != test_header.n_positions) {
        fprintf(stderr, "check-ref: incompatible dump headers\n");
        goto done;
    }
    reference = malloc((size_t)reference_header.vocab * sizeof(*reference));
    test = malloc((size_t)reference_header.vocab * sizeof(*test));
    kls = malloc((size_t)reference_header.n_positions * sizeof(*kls));
    if (!reference || !test || !kls) {
        fprintf(stderr, "check-ref: out of memory\n");
        goto done;
    }
    double max_kl = 0, sum_overlap = 0;
    float max_abs = 0;
    uint32_t top1_matches = 0;
    for (uint32_t row = 0; row < reference_header.n_positions; ++row) {
        uint32_t reference_position, test_position;
        if (fread(&reference_position, sizeof(reference_position), 1, reference_file) != 1 ||
            fread(reference, sizeof(*reference), reference_header.vocab, reference_file) != reference_header.vocab ||
            fread(&test_position, sizeof(test_position), 1, test_file) != 1 ||
            fread(test, sizeof(*test), test_header.vocab, test_file) != test_header.vocab ||
            reference_position != test_position) {
            fprintf(stderr, "check-ref: dump records differ or are truncated at row %u\n", row);
            goto done;
        }
        comparison result = compare(reference, test, reference_header.vocab);
        kls[row] = result.kl;
        if (result.kl > max_kl) max_kl = result.kl;
        if (result.max_abs > max_abs) max_abs = result.max_abs;
        top1_matches += result.ref_top1 == result.dsm_top1;
        sum_overlap += overlap(result.ref_top5, result.dsm_top5);
        printf("position=%u kl=%.8g max_abs_dlogit=%.8g top1=%s top5_overlap=%u/%u\n",
               reference_position, result.kl, result.max_abs,
               result.ref_top1 == result.dsm_top1 ? "match" : "different",
               overlap(result.ref_top5, result.dsm_top5), TOP_K);
    }
    qsort(kls, reference_header.n_positions, sizeof(*kls), compare_kl);
    double median = kls[reference_header.n_positions / 2];
    double gate_kl = fmax(1e-3, 3.0 * floor);
    double top1_percent = 100.0 * top1_matches / reference_header.n_positions;
    double top5_percent = 100.0 * sum_overlap / (TOP_K * reference_header.n_positions);
    bool pass = median <= gate_kl && max_kl <= 1e-2 && top1_percent >= 99.0;
    printf("summary positions=%u kl_median=%.8g kl_max=%.8g top1_match=%.4f%% "
           "top5_overlap=%.4f%% max_abs_dlogit=%.8g gate_kl=%.8g %s\n",
           reference_header.n_positions, median, max_kl, top1_percent, top5_percent,
           max_abs, gate_kl, pass ? "PASS" : "FAIL");
    status = pass ? 0 : 1;
done:
    if (reference_file) fclose(reference_file);
    if (test_file) fclose(test_file);
    free(kls);
    free(test);
    free(reference);
    return status;
}

int main(int argc, char **argv) {
    const char *model_path, *ids_path, *reference_path, *backend = "metal";
    const char *dsm_dump_path = NULL, *write_dsm_path = NULL;
    double floor = 0;
    bool prefill = false;
    int32_t *ids = NULL;
    size_t n_ids = 0;
    FILE *reference_file = NULL, *dsm_file = NULL;
    ref_header reference_header, dsm_header;
    dsm_model model = {0};
    dsm_metal *metal = NULL;
    dsm_forward *forward = NULL;
    float *reference = NULL, *dsm = NULL;
    double *kls = NULL;
    int status = 1;

    if (argc >= 4 && !strcmp(argv[1], "--compare-dumps")) {
        double floor = 0;
        if (argc == 6 && !strcmp(argv[4], "--floor")) floor = strtod(argv[5], NULL);
        else if (argc != 4) {
            usage(argv[0]);
            return 2;
        }
        if (floor < 0 || !isfinite(floor)) {
            fprintf(stderr, "check-ref: --floor must be finite and non-negative\n");
            return 2;
        }
        return compare_dumps(argv[2], argv[3], floor);
    }
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }
    model_path = argv[1];
    ids_path = argv[2];
    reference_path = argv[3];
    for (int arg = 4; arg < argc; ++arg) {
        if (!strcmp(argv[arg], "--backend") && arg + 1 < argc) backend = argv[++arg];
        else if (!strcmp(argv[arg], "--floor") && arg + 1 < argc) {
            floor = strtod(argv[++arg], NULL);
            if (floor < 0 || !isfinite(floor)) {
                fprintf(stderr, "check-ref: --floor must be finite and non-negative\n");
                return 2;
            }
        }
        else if (!strcmp(argv[arg], "--dsm-dump") && arg + 1 < argc) dsm_dump_path = argv[++arg];
        else if (!strcmp(argv[arg], "--dump-dsm") && arg + 1 < argc) write_dsm_path = argv[++arg];
        else if (!strcmp(argv[arg], "--prefill")) prefill = true;
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (strcmp(backend, "metal") && strcmp(backend, "cpu")) {
        fprintf(stderr, "check-ref: unknown backend %s\n", backend);
        return 2;
    }
    if (dsm_dump_path && write_dsm_path) {
        fprintf(stderr, "check-ref: --dsm-dump and --dump-dsm cannot be combined\n");
        return 2;
    }
    if (!read_ids(ids_path, &ids, &n_ids) || !n_ids || n_ids > UINT32_MAX) {
        fprintf(stderr, "check-ref: cannot parse token id file %s\n", ids_path);
        goto done;
    }
    reference_file = fopen(reference_path, "rb");
    if (!reference_file || fread(&reference_header, sizeof(reference_header), 1,
                                 reference_file) != 1 || !valid_header(&reference_header)) {
        fprintf(stderr, "check-ref: invalid llama dump %s\n", reference_path);
        goto done;
    }
    if (!dsm_model_open(&model, model_path)) {
        fprintf(stderr, "check-ref: %s\n", model.error);
        goto done;
    }
    if (reference_header.vocab != model.vocab) {
        fprintf(stderr, "check-ref: vocabulary mismatch: llama=%u dsm=%u\n",
                reference_header.vocab, model.vocab);
        goto done;
    }
    for (size_t i = 0; i < n_ids; ++i)
        if (ids[i] < 0 || (uint32_t)ids[i] >= model.vocab) {
            fprintf(stderr, "check-ref: token %zu (%d) is outside the vocabulary\n", i, ids[i]);
            goto done;
        }
    if (dsm_dump_path) {
        dsm_file = fopen(dsm_dump_path, "rb");
        if (!dsm_file || fread(&dsm_header, sizeof(dsm_header), 1, dsm_file) != 1 ||
            !valid_header(&dsm_header) || dsm_header.vocab != model.vocab ||
            dsm_header.n_positions != reference_header.n_positions) {
            fprintf(stderr, "check-ref: invalid DSM dump %s\n", dsm_dump_path);
            goto done;
        }
    } else {
        char error[256];
        if (!strcmp(backend, "metal")) {
            if (!dsm_metal_init(&metal, &model, (uint32_t)n_ids + 1, error, sizeof(error))) {
                fprintf(stderr, "check-ref: Metal initialization failed: %s\n", error);
                goto done;
            }
        } else if (!dsm_forward_init(&forward, &model, (uint32_t)n_ids + 1, 8)) {
            fprintf(stderr, "check-ref: CPU forward initialization failed\n");
            goto done;
        }
        if (write_dsm_path) {
            dsm_file = fopen(write_dsm_path, "wb");
            dsm_header = (ref_header){.magic = "DSMREF1", .version = REF_VERSION,
                .header_bytes = REF_HEADER_BYTES, .vocab = model.vocab,
                .n_positions = reference_header.n_positions,
                .record_bytes = sizeof(uint32_t) + model.vocab * sizeof(float)};
            if (!dsm_file || fwrite(&dsm_header, sizeof(dsm_header), 1, dsm_file) != 1) {
                fprintf(stderr, "check-ref: cannot write DSM dump %s\n", write_dsm_path);
                goto done;
            }
        }
    }
    reference = malloc((size_t)model.vocab * sizeof(*reference));
    dsm = malloc((size_t)model.vocab * sizeof(*dsm));
    kls = malloc((size_t)reference_header.n_positions * sizeof(*kls));
    if (!reference || !dsm || !kls) {
        fprintf(stderr, "check-ref: out of memory\n");
        goto done;
    }

    if (prefill && dsm_dump_path) { fprintf(stderr, "check-ref: --prefill cannot read a DSM dump\n"); goto done; }
    if (prefill && reference_header.n_positions != 1) { fprintf(stderr, "check-ref: --prefill requires one llama logit record\n"); goto done; }
    if (prefill && !(metal && dsm_metal_prefill(metal, ids, (uint32_t)n_ids, dsm))) { fprintf(stderr, "check-ref: DSM prefill failed\n"); goto done; }
    if (prefill && dsm_dump_path) { fprintf(stderr, "check-ref: --prefill cannot read a DSM dump\n"); goto done; }
    if (prefill && reference_header.n_positions != 1) { fprintf(stderr, "check-ref: --prefill requires one llama logit record\n"); goto done; }
    if (prefill && !(metal && dsm_metal_prefill(metal, ids, (uint32_t)n_ids, dsm))) { fprintf(stderr, "check-ref: DSM prefill failed\n"); goto done; }
    double max_kl = 0, sum_overlap = 0;
    float max_abs = 0;
    uint32_t top1_matches = 0, first_divergence = UINT32_MAX;
    uint32_t next_position = 0;
    for (uint32_t row = 0; row < reference_header.n_positions; ++row) {
        uint32_t reference_position, dsm_position;
        if (fread(&reference_position, sizeof(reference_position), 1, reference_file) != 1 ||
            fread(reference, sizeof(*reference), model.vocab, reference_file) != model.vocab ||
            reference_position >= n_ids) {
            fprintf(stderr, "check-ref: truncated or invalid llama record %u\n", row);
            goto done;
        }
        if (dsm_dump_path) {
            if (fread(&dsm_position, sizeof(dsm_position), 1, dsm_file) != 1 ||
                fread(dsm, sizeof(*dsm), model.vocab, dsm_file) != model.vocab ||
                dsm_position != reference_position) {
                fprintf(stderr, "check-ref: DSM record %u does not match llama position\n", row);
                goto done;
            }
        } else if (!prefill) {
            while (next_position <= reference_position) {
                int ok = metal ? dsm_metal_token(metal, ids[next_position], dsm)
                               : dsm_forward_token(forward, ids[next_position], dsm);
                if (!ok) {
                    fprintf(stderr, "check-ref: DSM inference failed at input position %u\n",
                            next_position);
                    goto done;
                }
                ++next_position;
            }
            if (dsm_file && (fwrite(&reference_position, sizeof(reference_position), 1, dsm_file) != 1 ||
                             fwrite(dsm, sizeof(*dsm), model.vocab, dsm_file) != model.vocab)) {
                fprintf(stderr, "check-ref: failed to write DSM dump\n");
                goto done;
            }
        }
        comparison result = compare(reference, dsm, model.vocab);
        kls[row] = result.kl;
        if (result.kl > max_kl) max_kl = result.kl;
        if (result.max_abs > max_abs) max_abs = result.max_abs;
        top1_matches += result.ref_top1 == result.dsm_top1;
        sum_overlap += overlap(result.ref_top5, result.dsm_top5);
        if (first_divergence == UINT32_MAX && result.ref_top1 != result.dsm_top1) {
            first_divergence = reference_position;
            float margin = reference[result.ref_top1] - reference[result.ref_top5[1]];
            printf("first_top1_divergence position=%u ref=%u dsm=%u ref_margin=%.8g\n",
                   reference_position, result.ref_top1, result.dsm_top1, margin);
        }
        printf("position=%u kl=%.8g max_abs_dlogit=%.8g top1=%s top5_overlap=%u/%u\n",
               reference_position, result.kl, result.max_abs,
               result.ref_top1 == result.dsm_top1 ? "match" : "different",
               overlap(result.ref_top5, result.dsm_top5), TOP_K);
    }
    qsort(kls, reference_header.n_positions, sizeof(*kls), compare_kl);
    double median = kls[reference_header.n_positions / 2];
    double gate_kl = fmax(1e-3, 3.0 * floor);
    double top1_percent = 100.0 * top1_matches / reference_header.n_positions;
    double top5_percent = 100.0 * sum_overlap / (TOP_K * reference_header.n_positions);
    bool pass = median <= gate_kl && max_kl <= 1e-2 && top1_percent >= 99.0;
    printf("summary positions=%u kl_median=%.8g kl_max=%.8g top1_match=%.4f%% "
           "top5_overlap=%.4f%% max_abs_dlogit=%.8g gate_kl=%.8g %s\n",
           reference_header.n_positions, median, max_kl, top1_percent, top5_percent,
           max_abs, gate_kl, pass ? "PASS" : "FAIL");
    status = pass ? 0 : 1;
done:
    if (reference_file) fclose(reference_file);
    if (dsm_file) fclose(dsm_file);
    free(kls);
    free(dsm);
    free(reference);
    dsm_forward_free(forward);
    dsm_metal_free(metal);
    dsm_model_close(&model);
    free(ids);
    return status;
}

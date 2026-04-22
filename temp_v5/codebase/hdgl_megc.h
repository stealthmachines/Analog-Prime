/*
 * hdgl_megc.h
 * Mapped Entropic Golden Codec (MEGC) -- C implementation
 * Ported from Python MEGCEncoder/MEGCDecoder/GoldenContext/BreathingEntropyCoder
 *
 * No external dependencies.  Compile with MSVC or gcc/nvcc.
 *
 * Usage:
 *   megc_encoder_t enc;
 *   megc_encoder_init(&enc);
 *   megc_encode_str(&enc, "MEGC V1.0");
 *   // enc.out[i].symbol, enc.out[i].weight
 *
 *   megc_decoder_t dec;
 *   megc_decoder_init(&dec, enc.out, enc.out_len);
 *   char buf[256];
 *   int n = megc_decode_str(&dec, buf, sizeof(buf));
 *
 *   megc_encoder_free(&enc);
 *   megc_decoder_free(&dec);
 *
 *   // DNA codec (ternary -> AGTC alphabet)
 *   int bits[] = {1,0,2,2,1,0};
 *   char dna[64];
 *   megc_encode_dna(bits, 6, dna, sizeof(dna));
 *   int out_bits[64];
 *   int n_bits = megc_decode_dna(dna, out_bits, 64);
 */

#pragma once
#ifndef HDGL_MEGC_H
#define HDGL_MEGC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Constants ---- */
#define MEGC_PHI         1.6180339887498948482
#define MEGC_INV_PHI     0.6180339887498948482
#define MEGC_MAX_SYMBOLS 256    /* ASCII range */
#define MEGC_MAX_TERNARY 16     /* max ternary digits for ord <= 255 (base-3 of 255 = 5 digits; 16 is safe) */
#define MEGC_OUT_CAP     4096   /* initial output capacity for encoder */

/* ---- TernaryNode ---- */
typedef struct megc_node {
    int               value;        /* -1 = internal node; 0..255 = leaf symbol */
    struct megc_node *ch[3];        /* children: 0=low, 1=mid, 2=high */
} megc_node_t;

/* ---- GoldenContext (phi-scaled frequency model) ---- */
typedef struct {
    double count[MEGC_MAX_SYMBOLS]; /* occurrence counts */
    double total[MEGC_MAX_SYMBOLS]; /* PHI-scaled totals */
} megc_ctx_t;

/* ---- BreathingEntropyCoder ---- */
typedef struct {
    double   low;
    double   high;
    double  *output;
    int      out_len;
    int      out_cap;
} megc_bec_t;

/* ---- Encoder output record ---- */
typedef struct {
    unsigned char symbol;
    double        weight;    /* phi-weighted interval (numerator/denominator approx) */
} megc_record_t;

/* ---- MEGCEncoder ---- */
typedef struct {
    uint32_t       freq[MEGC_MAX_SYMBOLS];
    uint32_t       total;
    megc_node_t   *tree_root;
    megc_record_t *out;
    int            out_len;
    int            out_cap;
} megc_encoder_t;

/* ---- MEGCDecoder ---- */
typedef struct {
    const megc_record_t *data;
    int                  data_len;
    megc_node_t         *tree_root;
} megc_decoder_t;

/* ---- API ---- */

/* Encoder lifecycle */
void megc_encoder_init(megc_encoder_t *enc);
void megc_encoder_free(megc_encoder_t *enc);
/* Encode a byte string; appends to enc->out */
int  megc_encode_str(megc_encoder_t *enc, const char *data, int len);

/* Decoder lifecycle */
void megc_decoder_init(megc_decoder_t *dec, const megc_record_t *data, int len);
void megc_decoder_free(megc_decoder_t *dec);
/* Decode back to bytes; returns number of bytes written */
int  megc_decode_str(megc_decoder_t *dec, char *out, int out_cap);

/* GoldenContext helpers (used internally; exposed for testing) */
void   megc_ctx_init(megc_ctx_t *ctx);
void   megc_ctx_update(megc_ctx_t *ctx, unsigned char sym);
double megc_ctx_probability(const megc_ctx_t *ctx, unsigned char sym);

/* BreathingEntropyCoder (BEC) */
void    megc_bec_init(megc_bec_t *bec);
void    megc_bec_free(megc_bec_t *bec);
int     megc_bec_encode_symbol(megc_bec_t *bec, unsigned char sym, megc_ctx_t *ctx);
double *megc_bec_finalize(megc_bec_t *bec, int *out_len);

/* DNA codec (ternary integer input) */
/* data_bits: array of ints in {0,1,2}; dna_out: NUL-terminated AGTC string */
int megc_encode_dna(const int *data_bits, int n_bits, char *dna_out, int dna_cap);
/* Returns number of decoded ints written into bits_out */
int megc_decode_dna(const char *dna, int *bits_out, int bits_cap);

/* Vector/DNA codec (float field input)
 *
 * megc_encode_field_dna:
 *   Encode a float field[0..n) in [0,1] as a DNA strand.
 *   Each sample maps to a ternary symbol:
 *     [0,  1/3) -> 0 -> 'A'  (low entropy / anchor)
 *     [1/3,2/3) -> 1 -> 'G'  (mid entropy)
 *     [2/3, 1]  -> 2 -> 'T'  (high entropy)
 *   Fold-control 'C' is inserted after every fold_trigger consecutive T's.
 *   Returns bytes written (not counting NUL), or -1 if dna_cap too small.
 *
 * megc_decode_field_dna:
 *   Decode a DNA strand back to float samples in [0, 1].
 *   A -> 1/6,  G -> 1/2,  T -> 5/6.  'C' is skipped.
 *   Returns number of floats written, or -1.
 */
int megc_encode_field_dna(const float *field, int n,
                          char *dna_out, int dna_cap,
                          int fold_trigger);
int megc_decode_field_dna(const char *dna,
                          float *field_out, int field_cap);

#ifdef __cplusplus
}
#endif

#endif /* HDGL_MEGC_H */

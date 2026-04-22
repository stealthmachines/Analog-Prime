/*
 * hdgl_megc.c
 * Mapped Entropic Golden Codec (MEGC) -- C implementation
 * Ported from Python MEGCEncoder v1.0.0 / MEGCDecoder / GoldenContext /
 * BreathingEntropyCoder + DNA codec.
 *
 * No external dependencies.
 */

#include "hdgl_megc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Internal: ternary digits of n in base 3
 * Returns number of digits written into trit[] (MSB first).
 * If n==0 returns 1 digit (0).
 * ================================================================ */
static int int_to_ternary(int n, int trit[MEGC_MAX_TERNARY])
{
    if (n == 0) { trit[0] = 0; return 1; }

    int tmp[MEGC_MAX_TERNARY];
    int len = 0;
    while (n > 0 && len < MEGC_MAX_TERNARY) {
        tmp[len++] = n % 3;
        n /= 3;
    }
    /* reverse into trit (MSB first) */
    for (int i = 0; i < len; i++)
        trit[i] = tmp[len - 1 - i];
    return len;
}

/* ================================================================
 * TernaryNode pool allocator (simple bump allocator for trees)
 * ================================================================ */
#define NODE_POOL_SIZE 4096

typedef struct {
    megc_node_t pool[NODE_POOL_SIZE];
    int         used;
} node_pool_t;

static megc_node_t *pool_alloc(node_pool_t *p)
{
    if (p->used >= NODE_POOL_SIZE) return NULL;
    megc_node_t *n = &p->pool[p->used++];
    n->value  = -1;
    n->ch[0]  = n->ch[1] = n->ch[2] = NULL;
    return n;
}

/* Each encoder/decoder owns a pool.  We store the pool pointer in the
 * tree_root field by using a wrapper. */

typedef struct {
    node_pool_t pool;
    megc_node_t *root;
} megc_tree_t;

static megc_tree_t *tree_create(void)
{
    megc_tree_t *t = (megc_tree_t*)calloc(1, sizeof(megc_tree_t));
    if (!t) return NULL;
    t->root = pool_alloc(&t->pool);
    return t;
}

static void tree_insert(megc_tree_t *t, unsigned char sym)
{
    int trit[MEGC_MAX_TERNARY];
    int len = int_to_ternary((int)sym, trit);

    megc_node_t *node = t->root;
    for (int i = 0; i < len; i++) {
        int branch = trit[i];
        if (!node->ch[branch])
            node->ch[branch] = pool_alloc(&t->pool);
        if (!node->ch[branch]) return; /* pool exhausted */
        node = node->ch[branch];
    }
    node->value = (int)sym;
}

/* ================================================================
 * GoldenContext
 * ================================================================ */
void megc_ctx_init(megc_ctx_t *ctx)
{
    for (int i = 0; i < MEGC_MAX_SYMBOLS; i++) {
        ctx->count[i] = 1.0;
        ctx->total[i] = 1.0;
    }
}

void megc_ctx_update(megc_ctx_t *ctx, unsigned char sym)
{
    ctx->count[sym] += 1.0;
    ctx->total[sym] += MEGC_PHI;
}

double megc_ctx_probability(const megc_ctx_t *ctx, unsigned char sym)
{
    return ctx->count[sym] / ctx->total[sym];
}

/* ================================================================
 * BreathingEntropyCoder
 * ================================================================ */
void megc_bec_init(megc_bec_t *bec)
{
    bec->low    = 0.0;
    bec->high   = 1.0;
    bec->out_cap = 256;
    bec->output  = (double*)malloc(bec->out_cap * sizeof(double));
    bec->out_len = 0;
}

void megc_bec_free(megc_bec_t *bec)
{
    free(bec->output);
    bec->output = NULL;
}

static int bec_push(megc_bec_t *bec, double val)
{
    if (bec->out_len >= bec->out_cap) {
        int new_cap = bec->out_cap * 2;
        double *nd = (double*)realloc(bec->output, new_cap * sizeof(double));
        if (!nd) return -1;
        bec->output  = nd;
        bec->out_cap = new_cap;
    }
    bec->output[bec->out_len++] = val;
    return 0;
}

int megc_bec_encode_symbol(megc_bec_t *bec, unsigned char sym, megc_ctx_t *ctx)
{
    double p     = megc_ctx_probability(ctx, sym);
    double range = bec->high - bec->low;

    bec->high = bec->low + range * p;
    bec->low  = bec->low + range * (1.0 - p);
    megc_ctx_update(ctx, sym);

    if (bec->high - bec->low < 1e-6) {
        bec_push(bec, (bec->low + bec->high) * 0.5);
        bec->low  = 0.0;
        bec->high = 1.0;
    }
    return 0;
}

double *megc_bec_finalize(megc_bec_t *bec, int *out_len)
{
    bec_push(bec, (bec->low + bec->high) * 0.5);
    *out_len = bec->out_len;
    return bec->output;
}

/* ================================================================
 * MEGCEncoder
 * ================================================================ */
void megc_encoder_init(megc_encoder_t *enc)
{
    memset(enc->freq, 0, sizeof(enc->freq));
    enc->total    = 0;
    enc->out_cap  = MEGC_OUT_CAP;
    enc->out_len  = 0;
    enc->out      = (megc_record_t*)malloc(enc->out_cap * sizeof(megc_record_t));
    enc->tree_root = (megc_node_t*)tree_create();  /* cast; we treat opaquely */
}

void megc_encoder_free(megc_encoder_t *enc)
{
    free(enc->out);
    enc->out = NULL;
    free(enc->tree_root);  /* frees the megc_tree_t block */
    enc->tree_root = NULL;
}

static int enc_push(megc_encoder_t *enc, unsigned char sym, double w)
{
    if (enc->out_len >= enc->out_cap) {
        int new_cap = enc->out_cap * 2;
        megc_record_t *nd = (megc_record_t*)realloc(enc->out,
                                new_cap * sizeof(megc_record_t));
        if (!nd) return -1;
        enc->out     = nd;
        enc->out_cap = new_cap;
    }
    enc->out[enc->out_len].symbol = sym;
    enc->out[enc->out_len].weight = w;
    enc->out_len++;
    return 0;
}

int megc_encode_str(megc_encoder_t *enc, const char *data, int len)
{
    megc_tree_t *tree = (megc_tree_t*)enc->tree_root;

    for (int i = 0; i < len; i++) {
        unsigned char sym = (unsigned char)data[i];

        /* update_freq */
        enc->freq[sym]++;
        enc->total++;

        /* phi_weight: (freq+1) * INV_PHI */
        double freq   = (double)(enc->freq[sym] + 1);
        double weight = freq * MEGC_INV_PHI;

        /* interval approximation: weight / (total+1) */
        double interval = weight / (double)(enc->total + 1);

        enc_push(enc, sym, interval);
        tree_insert(tree, sym);
    }
    return enc->out_len;
}

/* ================================================================
 * MEGCDecoder
 * ================================================================ */
void megc_decoder_init(megc_decoder_t *dec, const megc_record_t *data, int len)
{
    dec->data      = data;
    dec->data_len  = len;
    dec->tree_root = (megc_node_t*)tree_create();
}

void megc_decoder_free(megc_decoder_t *dec)
{
    free(dec->tree_root);
    dec->tree_root = NULL;
}

int megc_decode_str(megc_decoder_t *dec, char *out, int out_cap)
{
    megc_tree_t *tree = (megc_tree_t*)dec->tree_root;
    int written = 0;

    for (int i = 0; i < dec->data_len && written < out_cap - 1; i++) {
        unsigned char sym = dec->data[i].symbol;
        if (out) out[written] = (char)sym;
        written++;
        tree_insert(tree, sym);
    }
    if (out && written < out_cap) out[written] = '\0';
    return written;
}

/* ================================================================
 * DNA codec
 * A=0 (low/anchor), G=1 (mid), T=2 (high), C=fold-control
 * ================================================================ */
static char trit_to_dna(int t)
{
    if (t == 0) return 'A';
    if (t == 1) return 'G';
    return 'T';
}

static int dna_to_trit(char c)
{
    if (c == 'A') return 0;
    if (c == 'G') return 1;
    if (c == 'T') return 2;
    return -1; /* 'C' = fold control; caller handles */
}

int megc_encode_dna(const int *data_bits, int n_bits, char *dna_out, int dna_cap)
{
    int out_pos    = 0;
    int fold_count = 0;
    const int fold_trigger = 3;

    for (int i = 0; i < n_bits; i++) {
        int entropy = data_bits[i] % 3;

        if (out_pos >= dna_cap - 2) break;
        dna_out[out_pos++] = trit_to_dna(entropy);

        if (entropy == 2) fold_count++;
        if (fold_count >= fold_trigger) {
            if (out_pos < dna_cap - 1)
                dna_out[out_pos++] = 'C'; /* folding signal */
            fold_count = 0;
        }
    }
    if (out_pos < dna_cap) dna_out[out_pos] = '\0';
    return out_pos;
}

int megc_decode_dna(const char *dna, int *bits_out, int bits_cap)
{
    int n = 0;
    for (int i = 0; dna[i] != '\0' && n < bits_cap; i++) {
        if (dna[i] == 'C') continue; /* skip fold-control */
        int t = dna_to_trit(dna[i]);
        if (t >= 0) bits_out[n++] = t;
    }
    return n;
}

/* ================================================================
 * Vector / DNA field codec
 * ================================================================ */

int megc_encode_field_dna(const float *field, int n,
                          char *dna_out, int dna_cap,
                          int fold_trigger)
{
    int   out_pos    = 0;
    int   fold_count = 0;

    if (fold_trigger <= 0) fold_trigger = 3; /* default from Python */

    for (int i = 0; i < n; i++) {
        float v = field[i];
        int trit;
        if      (v < 0.3333333f) trit = 0;
        else if (v < 0.6666667f) trit = 1;
        else                      trit = 2;

        if (out_pos >= dna_cap - 2) return -1;
        dna_out[out_pos++] = trit_to_dna(trit);

        if (trit == 2) fold_count++;
        if (fold_count >= fold_trigger) {
            if (out_pos >= dna_cap - 1) return -1;
            dna_out[out_pos++] = 'C'; /* fold-control signal */
            fold_count = 0;
        }
    }
    if (out_pos < dna_cap) dna_out[out_pos] = '\0';
    return out_pos;
}

int megc_decode_field_dna(const char *dna, float *field_out, int field_cap)
{
    /* centroid values: A=1/6, G=1/2, T=5/6 */
    int n = 0;
    for (int i = 0; dna[i] != '\0' && n < field_cap; i++) {
        char c = dna[i];
        if (c == 'C') continue; /* skip fold-control */
        if      (c == 'A') field_out[n++] = 0.1666667f;
        else if (c == 'G') field_out[n++] = 0.5000000f;
        else if (c == 'T') field_out[n++] = 0.8333333f;
    }
    return n;
}

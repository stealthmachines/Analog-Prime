/*
 * hdgl_fold26.c
 * fold26 Delta+RLE adaptive compression -- stripped of external deps.
 * Ported from fold26_production.c (spare parts).
 *
 * Changes from original:
 *  - SHA-256 checksum replaced with CRC-32 (no OpenSSL dependency)
 *  - zlib / gzip pass removed; Delta+RLE only (covers the wu-wei
 *    STRATEGY_NONACTION / FLOWING_RIVER / REPEATED_WAVES paths)
 *  - Compiles with: cl.exe /O2 hdgl_fold26.c   or  gcc -O2 hdgl_fold26.c
 *
 * API:
 *   fold26_compress(in, in_size, out, out_cap)  -> bytes written, or -1
 *   fold26_decompress(in, in_size, out, out_cap) -> bytes written, or -1
 *
 * Wire format (header 12 bytes):
 *   [0..3]  magic  "F26\x02"
 *   [4..7]  original size  (uint32_t LE)
 *   [8..11] CRC-32 of original data (uint32_t LE)
 *   [12..]  delta-then-RLE compressed payload
 */

#include "hdgl_fold26.h"

#include <stdlib.h>
#include <string.h>

/* ================================================================
 * CRC-32 (IEEE 802.3 polynomial, no lookup table to stay header-only)
 * ================================================================ */
static uint32_t crc32_byte(uint32_t crc, uint8_t b)
{
    crc ^= (uint32_t)b;
    for (int i = 0; i < 8; i++)
        crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    return crc;
}

static uint32_t crc32_buf(const uint8_t *data, size_t size)
{
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < size; i++)
        crc = crc32_byte(crc, data[i]);
    return crc ^ 0xFFFFFFFFU;
}

/* ================================================================
 * Dynamic byte buffer
 * ================================================================ */
typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   cap;
} buf_t;

static int buf_init(buf_t *b, size_t cap)
{
    b->data = (uint8_t*)malloc(cap);
    b->size = 0;
    b->cap  = cap;
    return b->data ? 0 : -1;
}

static void buf_free(buf_t *b)
{
    free(b->data);
    b->data = NULL;
}

static int buf_push(buf_t *b, uint8_t byte)
{
    if (b->size >= b->cap) {
        size_t nc = b->cap * 2;
        uint8_t *nd = (uint8_t*)realloc(b->data, nc);
        if (!nd) return -1;
        b->data = nd;
        b->cap  = nc;
    }
    b->data[b->size++] = byte;
    return 0;
}

static int buf_push_u32le(buf_t *b, uint32_t v)
{
    return buf_push(b, (uint8_t)(v      )) |
           buf_push(b, (uint8_t)(v >>  8)) |
           buf_push(b, (uint8_t)(v >> 16)) |
           buf_push(b, (uint8_t)(v >> 24));
}

/* ================================================================
 * Delta encode / decode  (single-byte delta mod 256)
 * ================================================================ */
static int delta_encode(const uint8_t *src, size_t n, buf_t *out)
{
    if (n == 0) return 0;
    if (buf_push(out, src[0]) < 0) return -1;
    for (size_t i = 1; i < n; i++) {
        uint8_t d = (uint8_t)((int)src[i] - (int)src[i-1]);
        if (buf_push(out, d) < 0) return -1;
    }
    return 0;
}

static int delta_decode(const uint8_t *src, size_t n, buf_t *out)
{
    if (n == 0) return 0;
    if (buf_push(out, src[0]) < 0) return -1;
    for (size_t i = 1; i < n; i++) {
        uint8_t v = (uint8_t)(out->data[out->size - 1] + src[i]);
        if (buf_push(out, v) < 0) return -1;
    }
    return 0;
}

/* ================================================================
 * RLE encode / decode  (escape byte 0xFF)
 * ================================================================ */
#define RLE_ESC      0xFFU
#define RLE_MIN_RUN  3
#define RLE_MAX_RUN  255

static int rle_encode(const uint8_t *src, size_t n, buf_t *out)
{
    size_t i = 0;
    while (i < n) {
        uint8_t v = src[i];
        size_t  run = 1;
        while (i + run < n && src[i + run] == v && run < RLE_MAX_RUN)
            run++;

        if (run >= RLE_MIN_RUN) {
            /* RLE packet: ESC, value, count */
            if (buf_push(out, RLE_ESC) < 0) return -1;
            if (buf_push(out, v)       < 0) return -1;
            if (buf_push(out, (uint8_t)run) < 0) return -1;
        } else {
            for (size_t k = 0; k < run; k++) {
                if (v == RLE_ESC) {
                    /* escaped literal 0xFF: ESC 0xFF 0x01 */
                    if (buf_push(out, RLE_ESC) < 0) return -1;
                    if (buf_push(out, RLE_ESC) < 0) return -1;
                    if (buf_push(out, 0x01)    < 0) return -1;
                } else {
                    if (buf_push(out, v) < 0) return -1;
                }
            }
        }
        i += run;
    }
    return 0;
}

static int rle_decode(const uint8_t *src, size_t n, buf_t *out)
{
    size_t i = 0;
    while (i < n) {
        if (src[i] == RLE_ESC) {
            if (i + 2 >= n) return -1; /* truncated */
            uint8_t v   = src[i + 1];
            uint8_t cnt = src[i + 2];
            for (uint8_t k = 0; k < cnt; k++)
                if (buf_push(out, v) < 0) return -1;
            i += 3;
        } else {
            if (buf_push(out, src[i]) < 0) return -1;
            i++;
        }
    }
    return 0;
}

/* ================================================================
 * Wu-wei strategy selection (simple heuristic; no zlib needed)
 *
 * Strategies mirroring fold26_bridge.c enum:
 *   STRATEGY_NONACTION     -- store raw (delta would expand)
 *   FLOWING_RIVER          -- delta only (low-entropy deltas)
 *   REPEATED_WAVES         -- RLE only  (runs of identical bytes)
 *   GENTLE_STREAM          -- delta + RLE (default)
 *   BALANCED_PATH          -- same as GENTLE_STREAM (future: add more)
 * ================================================================ */
typedef enum {
    STRATEGY_NONACTION = 0,
    FLOWING_RIVER,
    REPEATED_WAVES,
    GENTLE_STREAM,
    BALANCED_PATH
} fold26_strategy_t;

static fold26_strategy_t choose_strategy(const uint8_t *data, size_t n)
{
    if (n < 4) return STRATEGY_NONACTION;

    /* Count byte-to-byte delta zero runs (run indicator) */
    size_t runs = 0;
    for (size_t i = 1; i < n; i++)
        if (data[i] == data[i-1]) runs++;
    double run_ratio = (double)runs / (double)(n - 1);

    /* Count unique deltas (entropy indicator) */
    uint8_t seen[256] = {0};
    size_t  unique_delta = 0;
    for (size_t i = 1; i < n; i++) {
        uint8_t d = (uint8_t)((int)data[i] - (int)data[i-1]);
        if (!seen[d]) { seen[d] = 1; unique_delta++; }
    }
    double delta_entropy = (double)unique_delta / 256.0;

    if (run_ratio > 0.6)   return REPEATED_WAVES;  /* lots of runs */
    if (delta_entropy < 0.15) return FLOWING_RIVER; /* low-entropy deltas */
    return GENTLE_STREAM;                           /* default: delta+RLE */
}

/* ================================================================
 * Public API
 * ================================================================ */

/*
 * fold26_compress -- returns bytes written into out[], or -1 on error.
 * out_cap must be >= FOLD26_HEADER_SIZE + n + some overhead.
 */
int fold26_compress(const uint8_t *in, int in_size, uint8_t *out, int out_cap)
{
    if (!in || in_size <= 0 || !out || out_cap < FOLD26_HEADER_SIZE)
        return -1;

    fold26_strategy_t strategy = choose_strategy(in, (size_t)in_size);
    uint32_t crc = crc32_buf(in, (size_t)in_size);

    /* Build compressed payload */
    buf_t tmp, tmp2, payload;
    if (buf_init(&tmp,     (size_t)in_size * 2) < 0) return -1;
    if (buf_init(&tmp2,    (size_t)in_size * 2) < 0) { buf_free(&tmp); return -1; }
    if (buf_init(&payload, (size_t)in_size * 2) < 0) { buf_free(&tmp); buf_free(&tmp2); return -1; }

    int ok = 0;
    switch (strategy) {
        case STRATEGY_NONACTION:
            for (int i = 0; i < in_size; i++) buf_push(&payload, in[i]);
            break;
        case FLOWING_RIVER:
            ok = delta_encode(in, (size_t)in_size, &payload);
            break;
        case REPEATED_WAVES:
            ok = rle_encode(in, (size_t)in_size, &payload);
            break;
        case GENTLE_STREAM:
        case BALANCED_PATH:
        default:
            ok = delta_encode(in, (size_t)in_size, &tmp);
            if (ok == 0)
                ok = rle_encode(tmp.data, tmp.size, &payload);
            break;
    }

    buf_free(&tmp);
    buf_free(&tmp2);

    if (ok < 0) { buf_free(&payload); return -1; }

    /* If compressed >= original, store raw */
    if (payload.size >= (size_t)in_size && strategy != STRATEGY_NONACTION) {
        buf_free(&payload);
        if (buf_init(&payload, (size_t)in_size) < 0) return -1;
        for (int i = 0; i < in_size; i++) buf_push(&payload, in[i]);
        strategy = STRATEGY_NONACTION;
    }

    int total = FOLD26_HEADER_SIZE + (int)payload.size;
    if (total > out_cap) { buf_free(&payload); return -1; }

    /* Write header */
    out[0] = 'F'; out[1] = '2'; out[2] = '6'; out[3] = '\x02';
    out[4] = (uint8_t)((uint32_t)in_size      );
    out[5] = (uint8_t)((uint32_t)in_size >>  8);
    out[6] = (uint8_t)((uint32_t)in_size >> 16);
    out[7] = (uint8_t)((uint32_t)in_size >> 24);
    out[8]  = (uint8_t)(crc      );
    out[9]  = (uint8_t)(crc >>  8);
    out[10] = (uint8_t)(crc >> 16);
    out[11] = (uint8_t)(crc >> 24);
    out[FOLD26_HEADER_SIZE - 1] = (uint8_t)strategy; /* 1 byte strategy tag */

    memcpy(out + FOLD26_HEADER_SIZE, payload.data, payload.size);
    buf_free(&payload);
    return total;
}

/*
 * fold26_decompress -- returns bytes written into out[], or -1.
 */
int fold26_decompress(const uint8_t *in, int in_size, uint8_t *out, int out_cap)
{
    if (!in || in_size < FOLD26_HEADER_SIZE || !out) return -1;

    /* Verify magic */
    if (in[0]!='F' || in[1]!='2' || in[2]!='6' || in[3]!='\x02') return -1;

    uint32_t orig_size = (uint32_t)in[4]
                       | ((uint32_t)in[5] <<  8)
                       | ((uint32_t)in[6] << 16)
                       | ((uint32_t)in[7] << 24);
    uint32_t stored_crc = (uint32_t)in[8]
                        | ((uint32_t)in[9]  <<  8)
                        | ((uint32_t)in[10] << 16)
                        | ((uint32_t)in[11] << 24);
    fold26_strategy_t strategy = (fold26_strategy_t)in[FOLD26_HEADER_SIZE - 1];

    if ((int)orig_size > out_cap) return -1;

    const uint8_t *payload     = in  + FOLD26_HEADER_SIZE;
    int            payload_len = in_size - FOLD26_HEADER_SIZE;

    buf_t tmp, result;
    if (buf_init(&tmp,    orig_size * 2 + 64) < 0) return -1;
    if (buf_init(&result, orig_size     + 64) < 0) { buf_free(&tmp); return -1; }

    int ok = 0;
    switch (strategy) {
        case STRATEGY_NONACTION:
            for (int i = 0; i < payload_len; i++) buf_push(&result, payload[i]);
            break;
        case FLOWING_RIVER:
            ok = delta_decode(payload, (size_t)payload_len, &result);
            break;
        case REPEATED_WAVES:
            ok = rle_decode(payload, (size_t)payload_len, &result);
            break;
        case GENTLE_STREAM:
        case BALANCED_PATH:
        default:
            ok = rle_decode(payload, (size_t)payload_len, &tmp);
            if (ok == 0)
                ok = delta_decode(tmp.data, tmp.size, &result);
            break;
    }

    buf_free(&tmp);
    if (ok < 0 || result.size != orig_size) { buf_free(&result); return -1; }

    /* CRC check */
    uint32_t actual_crc = crc32_buf(result.data, result.size);
    if (actual_crc != stored_crc) { buf_free(&result); return -1; }

    memcpy(out, result.data, result.size);
    buf_free(&result);
    return (int)orig_size;
}

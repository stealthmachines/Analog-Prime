/*
 * hdgl_onion.c
 * Onion Shell Encoding -- multi-layer fold26 state-block wrapper
 */

#include "hdgl_onion.h"
#include "hdgl_fold26.h"
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * onion_wrap
 * ================================================================ */
int onion_wrap(const onion_layer_t *layers, int n_layers,
               uint8_t *out, int out_cap)
{
    if (!layers || n_layers <= 0 || !out || out_cap <= 0) return -1;

    int pos = 0;

    for (int i = 0; i < n_layers; i++) {
        const onion_layer_t *layer = &layers[i];

        /* Scratch buffer for fold26 output */
        int scratch_cap = layer->size + (layer->size / 4) + FOLD26_HEADER_SIZE + 64;
        uint8_t *scratch = (uint8_t*)malloc((size_t)scratch_cap);
        if (!scratch) return -1;

        int compressed_len = fold26_compress(layer->data, layer->size,
                                             scratch, scratch_cap);
        if (compressed_len < 0) { free(scratch); return -1; }

        int record_size = ONION_LAYER_HEADER + compressed_len;
        if (pos + record_size > out_cap) { free(scratch); return -1; }

        /* Write layer header */
        out[pos]     = layer->id;
        out[pos + 1] = (uint8_t)((uint32_t)compressed_len      );
        out[pos + 2] = (uint8_t)((uint32_t)compressed_len >>  8);
        out[pos + 3] = (uint8_t)((uint32_t)compressed_len >> 16);
        out[pos + 4] = (uint8_t)((uint32_t)compressed_len >> 24);

        /* Copy compressed payload */
        memcpy(out + pos + ONION_LAYER_HEADER, scratch, (size_t)compressed_len);

        free(scratch);
        pos += record_size;
    }

    return pos;
}

/* ================================================================
 * onion_unwrap
 * ================================================================ */
int onion_unwrap(const uint8_t *onion, int onion_size, uint8_t target_id,
                 uint8_t *out, int out_cap)
{
    if (!onion || onion_size < ONION_LAYER_HEADER || !out) return -1;

    int pos = 0;
    while (pos + ONION_LAYER_HEADER <= onion_size) {
        uint8_t  id  = onion[pos];
        uint32_t len = (uint32_t)onion[pos + 1]
                     | ((uint32_t)onion[pos + 2] <<  8)
                     | ((uint32_t)onion[pos + 3] << 16)
                     | ((uint32_t)onion[pos + 4] << 24);

        int payload_start = pos + ONION_LAYER_HEADER;
        if (payload_start + (int)len > onion_size) return -1; /* truncated */

        if (id == target_id) {
            return fold26_decompress(onion + payload_start, (int)len,
                                     out, out_cap);
        }

        pos = payload_start + (int)len;
    }

    return -1; /* layer not found */
}

/* ================================================================
 * onion_layer_count
 * ================================================================ */
int onion_layer_count(const uint8_t *onion, int onion_size)
{
    if (!onion || onion_size < ONION_LAYER_HEADER) return 0;

    int pos   = 0;
    int count = 0;

    while (pos + ONION_LAYER_HEADER <= onion_size) {
        uint32_t len = (uint32_t)onion[pos + 1]
                     | ((uint32_t)onion[pos + 2] <<  8)
                     | ((uint32_t)onion[pos + 3] << 16)
                     | ((uint32_t)onion[pos + 4] << 24);

        int next = pos + ONION_LAYER_HEADER + (int)len;
        if (next > onion_size) break;

        count++;
        pos = next;
    }

    return count;
}

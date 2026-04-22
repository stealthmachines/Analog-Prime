/*
 * hdgl_onion.h
 * Onion Shell Encoding -- multi-layer fold26 state-block wrapper
 *
 * Each "layer" is an independently fold26-compressed chunk tagged with a
 * layer ID.  Layers nest from innermost (MATH) to outermost (BUILD), so
 * a partial decompressor that only knows about the build layer can still
 * extract build info without understanding the inner math encoding.
 *
 * Wire format per layer:
 *   [0]    layer_id  (uint8_t)
 *   [1..4] payload_len after fold26 (uint32_t LE)
 *   [5..]  fold26-compressed payload
 *
 * The full onion block is the concatenation of all layer records.
 * Layers are stored innermost-first (MATH, CODE, BUILD) so sequential
 * readers encounter them in dependency order.
 *
 * No external dependencies.  Requires hdgl_fold26.h.
 */

#pragma once
#ifndef HDGL_ONION_H
#define HDGL_ONION_H

#include <stdint.h>
#include "hdgl_fold26.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layer IDs */
#define ONION_LAYER_MATH   0   /* math constants, formulas, gate equations */
#define ONION_LAYER_CODE   1   /* file/symbol index, build commands */
#define ONION_LAYER_BUILD  2   /* version stamps, test results, timestamps */
#define ONION_LAYER_MAX    3

/* Per-layer header size (id:1 + len:4 = 5 bytes) */
#define ONION_LAYER_HEADER 5

/*
 * onion_layer_t -- one layer's input before wrapping
 */
typedef struct {
    uint8_t        id;
    const uint8_t *data;
    int            size;
} onion_layer_t;

/*
 * onion_wrap
 *   Compress each layer independently with fold26 and concatenate.
 *   layers[]  : array of ONION_LAYER_MAX input layers (innermost first)
 *   n_layers  : number of entries (use ONION_LAYER_MAX)
 *   out       : output buffer
 *   out_cap   : capacity of output buffer
 *   Returns total bytes written, or -1 on error.
 *
 *   Recommended out_cap: sum(layer.size) * 2 + n_layers * (ONION_LAYER_HEADER + FOLD26_HEADER_SIZE)
 */
int onion_wrap(const onion_layer_t *layers, int n_layers,
               uint8_t *out, int out_cap);

/*
 * onion_unwrap
 *   Find and decompress layer `target_id` from an onion block.
 *   Returns bytes written into out[], or -1 if layer not found / error.
 */
int onion_unwrap(const uint8_t *onion, int onion_size, uint8_t target_id,
                 uint8_t *out, int out_cap);

/*
 * onion_layer_count
 *   Returns the number of layer records present in an onion block.
 */
int onion_layer_count(const uint8_t *onion, int onion_size);

#ifdef __cplusplus
}
#endif

#endif /* HDGL_ONION_H */

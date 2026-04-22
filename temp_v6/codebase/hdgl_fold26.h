/*
 * hdgl_fold26.h
 * fold26 Delta+RLE adaptive compression header (no external deps)
 */

#pragma once
#ifndef HDGL_FOLD26_H
#define HDGL_FOLD26_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FOLD26_MAGIC        "F26\x02"
#define FOLD26_HEADER_SIZE  13   /* magic(4) + orig_size(4) + crc32(4) + strategy(1) */

/*
 * fold26_compress
 *   Compress in[0..in_size) into out[0..out_cap).
 *   Returns number of bytes written, or -1 on error / out_cap too small.
 *   Recommended out_cap = in_size + in_size/4 + FOLD26_HEADER_SIZE.
 */
int fold26_compress  (const uint8_t *in, int in_size, uint8_t *out, int out_cap);

/*
 * fold26_decompress
 *   Decompress in[0..in_size) into out[0..out_cap).
 *   Returns number of bytes written (== original size), or -1.
 *   Verifies CRC-32 of reconstructed data; returns -1 on mismatch.
 */
int fold26_decompress(const uint8_t *in, int in_size, uint8_t *out, int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* HDGL_FOLD26_H */

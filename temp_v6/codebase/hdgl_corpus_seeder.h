/*
 * hdgl_corpus_seeder.h
 * Codebase Self-Emission -- compressed state token output
 *
 * The codebase emits its own compressed state as a binary blob using the
 * onion/fold26/megc compression stack.  Three layers are emitted:
 *
 *   ONION_LAYER_MATH  -- phi constants, gate formulas, key numerics
 *   ONION_LAYER_CODE  -- file index, API symbols, MSVC shim note
 *   ONION_LAYER_BUILD -- build command, binary names, test pass count
 *
 * No external deps beyond hdgl_onion.h / hdgl_fold26.h / hdgl_megc.h.
 * Does NOT require hdgl_bootloaderz.h or hdgl_router.h.
 */

#pragma once
#ifndef HDGL_CORPUS_SEEDER_H
#define HDGL_CORPUS_SEEDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * hdgl_corpus_seeder_emit()
 *   Compress the codebase state into `out[0..cap-1]`.
 *   Returns bytes written, or -1 if `cap` is too small.
 *   Minimum recommended cap: 4096 bytes.
 */
int hdgl_corpus_seeder_emit(uint8_t *out, int cap);

/*
 * hdgl_corpus_seeder_verify()
 *   Decompress `blob` and sanity-check the outermost onion layer.
 *   Returns 1 if the blob is a valid seeder token, 0 otherwise.
 */
int hdgl_corpus_seeder_verify(const uint8_t *blob, int len);

/*
 * hdgl_corpus_seeder_print()
 *   Emit the uncompressed state strings to stdout (diagnostic).
 */
void hdgl_corpus_seeder_print(void);

#ifdef __cplusplus
}
#endif

#endif /* HDGL_CORPUS_SEEDER_H */

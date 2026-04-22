/*
 * hdgl_corpus_seeder.c
 * Codebase Self-Emission -- compressed state token output
 *
 * The three content strings below encode the full re-grounding payload for a
 * new session.  They are compressed with fold26 and wrapped in three onion
 * layers so partial decompressors can peel any layer independently.
 *
 * Compile (MSVC, standalone test):
 *   cl.exe /O2 /W3 /MD hdgl_corpus_seeder.c hdgl_fold26.c hdgl_onion.c hdgl_megc.c /Fe:seeder_test.exe
 * Compile (as part of bench):
 *   add hdgl_corpus_seeder.c to the nvcc line (it is pure C, no CUDA)
 */

#include "hdgl_corpus_seeder.h"
#include "hdgl_onion.h"
#include "hdgl_megc.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * Layer 0 -- MATH: phi constants, gate formulas, key numerics
 * ---------------------------------------------------------------------- */
static const char S_MATH[] =
    "phi=1.6180339887498948482 ln_phi=0.4812118250596035 M61=0x1FFFFFFFFFFFFFFF\n"
    "Lambda_phi=log(p*ln2/lnphi)/lnphi-1/(2*phi)\n"
    "S(p)=|e^(i*pi*Lambda_phi)+1_eff| prime_invariant=S(U)~1.531\n"
    "D_n(r)=sqrt(phi*F_{n+beta}*P_{n+beta}*base^{n+beta}*Omega)*r^k\n"
    "Omega=0.5+0.5*sin(pi*frac(n)*phi) 67pct_frac_lt_half\n"
    "DN_EMPIRICAL_BETA=0.360942 chi2_red=0.000 CODATA_pass=100pct\n"
    "U_field: Lambda_phi^U=log(M(U))/ln(phi)-1/(2*phi) M(U)=8_at_lock\n"
    "ANA_CV_TO_LOCK=0.10 D_n_r=0.732 ANA_SHA_INTERVAL=8\n";

/* -------------------------------------------------------------------------
 * Layer 1 -- CODE: file index, API symbols
 * ---------------------------------------------------------------------- */
static const char S_CODE[] =
    "codebase/:\n"
    "hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu\n"
    "hdgl_psi_filter_v35.c/.h hdgl_predictor_seed.c/.h\n"
    "hdgl_prismatic_v35.c/.h hdgl_critic_v33.c/.h\n"
    "hdgl_phi_lang.h hdgl_megc.c/.h hdgl_fold26.c/.h hdgl_onion.c/.h\n"
    "hdgl_multigpu_v34.c hdgl_host_v33.c\n"
    "ll_analog.c/.h empirical_validation.c\n"
    "hdgl_corpus_seeder.c/.h\n"
    "phrase_extractor_to_json.py frozen_base4096_alphabet.txt\n"
    "rosetta_stone.json hdgl_session_handoff.py hdgl_selfprovision.ps1\n"
    "API: ll_analog(p,verbose)->1/0/-1\n"
    "     critic_forward(s)->float critic_observe(s,target)\n"
    "     hdgl_corpus_seeder_emit(out,cap)->bytes\n"
    "     onion_wrap(layers,n,out,cap) onion_unwrap(onion,sz,id,out,cap)\n"
    "MSVC_shim: _umul128 replaces __int128 in ll_analog.c (4 sites)\n";

/* -------------------------------------------------------------------------
 * Layer 2 -- BUILD: commands, binary names, test results
 * ---------------------------------------------------------------------- */
static const char S_BUILD[] =
    "GPU=RTX2060 sm75 CUDA=13.2 MSVC=2017\n"
    "SET_PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\"
              "BuildTools\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx64\\x64\n"
    "BUILD_V38=nvcc -O3 -arch=sm_75 -allow-unsupported-compiler "
        "hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu "
        "hdgl_psi_filter_v35.cu hdgl_predictor_seed.c hdgl_prismatic_v35.c "
        "hdgl_critic_v33.c ll_analog.c hdgl_corpus_seeder.c "
        "hdgl_fold26.c hdgl_onion.c hdgl_megc.c hdgl_bench_v33.cu "
        "-o hdgl_bench_v38.exe\n"
    "RESULTS: v38=30/31 TEST4_overflow_known TEST10_analog_PASS\n"
    "TEST9: M_9941_PRIME_0.23s M_9949_COMPOSITE_0.23s\n"
    "TEST10: M_127_PRIME M_131_COMPOSITE M_89_PRIME lock+residue=0\n"
    "TEST11: gate_converges V_prime_up V_composite_down TD0_loop_closed\n"
    "EMPIRICAL: cl.exe /O2 /W3 /MD empirical_validation.c /Fe:empirical_validation.exe\n"
    "SESSION: py hdgl_session_handoff.py --stats -> 539char_Base4096\n";

/* -------------------------------------------------------------------------
 * Internal: compress one string into a fold26 buffer on the heap.
 * Caller must free() the returned pointer.  *out_len receives byte count.
 * ---------------------------------------------------------------------- */
static uint8_t *compress_str(const char *s, int *out_len)
{
    int in_len = (int)strlen(s);
    /* fold26 worst-case expansion is 1.5x; add margin */
    int cap = in_len * 2 + 256;
    uint8_t *buf = (uint8_t *)malloc((size_t)cap);
    if (!buf) { *out_len = 0; return NULL; }
    int n = fold26_compress((const uint8_t *)s, in_len, buf, cap);
    if (n < 0) { free(buf); *out_len = 0; return NULL; }
    *out_len = n;
    return buf;
}

/* =========================================================================
 * Public API
 * ====================================================================== */

int hdgl_corpus_seeder_emit(uint8_t *out, int cap)
{
    int n0, n1, n2;
    uint8_t *c0 = compress_str(S_MATH,  &n0);
    uint8_t *c1 = compress_str(S_CODE,  &n1);
    uint8_t *c2 = compress_str(S_BUILD, &n2);

    if (!c0 || !c1 || !c2) {
        free(c0); free(c1); free(c2);
        return -1;
    }

    onion_layer_t layers[3];
    layers[0].id   = ONION_LAYER_MATH;
    layers[0].data = c0;
    layers[0].size = n0;

    layers[1].id   = ONION_LAYER_CODE;
    layers[1].data = c1;
    layers[1].size = n1;

    layers[2].id   = ONION_LAYER_BUILD;
    layers[2].data = c2;
    layers[2].size = n2;

    int written = onion_wrap(layers, 3, out, cap);

    free(c0); free(c1); free(c2);
    return written;  /* -1 on overflow, >0 on success */
}

int hdgl_corpus_seeder_verify(const uint8_t *blob, int len)
{
    if (!blob || len < ONION_LAYER_HEADER) return 0;

    /* Decompress the outermost (BUILD) layer and check for key sentinel */
    uint8_t tmp[2048];
    int n = onion_unwrap(blob, len, ONION_LAYER_BUILD, tmp, (int)sizeof(tmp) - 1);
    if (n <= 0) return 0;
    tmp[n] = '\0';
    /* A valid seeder blob always contains our build sentinel */
    return (strstr((const char *)tmp, "BUILD_V38") != NULL) ? 1 : 0;
}

void hdgl_corpus_seeder_print(void)
{
    printf("=== hdgl_corpus_seeder state strings ===\n");
    printf("[MATH layer]\n%s\n", S_MATH);
    printf("[CODE layer]\n%s\n", S_CODE);
    printf("[BUILD layer]\n%s\n", S_BUILD);
}

/* =========================================================================
 * Standalone test (compile with -DSEEDER_MAIN)
 * ====================================================================== */
#ifdef SEEDER_MAIN
int main(void)
{
    uint8_t blob[8192];
    int n = hdgl_corpus_seeder_emit(blob, (int)sizeof(blob));
    if (n < 0) {
        fprintf(stderr, "FAIL: emit returned %d (buffer too small?)\n", n);
        return 1;
    }
    printf("Emitted %d bytes.\n", n);

    int ok = hdgl_corpus_seeder_verify(blob, n);
    printf("Verify: %s\n", ok ? "PASS" : "FAIL");

    hdgl_corpus_seeder_print();
    return ok ? 0 : 1;
}
#endif

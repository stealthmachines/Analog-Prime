# HDGL Analog v33–v36 — φ-Resonance Mersenne Prime Search Engine

<img width="1301" height="630" alt="image" src="https://github.com/user-attachments/assets/9f681619-59db-4e3d-855a-610aec5378bd" />
https://zchg.org/t/github-stealthmachines-analog-prime/969

A CUDA-accelerated Mersenne prime candidate search engine built on Golden Recursive
Algebra (GRA), gpucarry/NTT Lucas-Lehmer arithmetic, a learned MLP critic, a Markov trit
verdict gate, and a full U-field bridge derived from the X+1=0 framework.

---

## Overview

HDGL (Harmonic Differential Golden Lattice) is an analog-inspired prime discovery
system. Rather than testing exponents sequentially, it evolves a GPU field of coupled
oscillators, each slot tracking a candidate Mersenne exponent via learned Hebbian
plasticity. Slots whose field state exhibits φ-resonant destructive interference with
−1 (the X+1=0 condition) are promoted as Mersenne prime candidates and verified with a
full carry-save Lucas-Lehmer test.

The key insight: the condition `e^(iπ) + 1 = 0` is a special case of a broader
resonance gate `S(p) = |e^(iπΛ_φ) + 1_eff|` where `Λ_φ` is the Mersenne exponent
projected into φ-log space. At prime exponents, the field state induces destructive
interference (`S → 0`). At composites, it produces constructive interference (`S → 2`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CPU Host  (hdgl_host_v33.c)                                                │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │ Critic update    │  │ Weight upload   │  │ Prime candidate harvest  │   │
│  │ TD(0) SGD        │  │ → __constant__  │  │ → LL verification queue  │   │
│  └──────────────────┘  └─────────────────┘  └──────────────────────────┘   │
│         ↕  stream2            ↕  host          ↕  stream0/1                │
├─────────────────────────────────────────────────────────────────────────────┤
│  GPU  (RTX 2060, sm_75)                                                     │
│                                                                             │
│  Stream 0: Field evolution kernel (hdgl_analog_v35.cu — latest)            │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Per slot (N slots, block=256):                          │               │
│  │   1. Load SoA: A_re, A_im, phase, phase_vel, r_h, acc   │               │
│  │   2. Read Feistel partner ph_j = phase[(i+89)%N]        │               │
│  │   3. LL-lite: 4 squarings of 32-bit proxy state         │               │
│  │   4. 4-neighbour wavelet coupling (4-scale Morlet)      │               │
│  │   5. κ·log(p) + σ-trit → dphvel (U^(p) + phase drive)  │               │
│  │   6. Feistel phase: ph = fmod(φ*(ph+0.5*ph_j+bias),1)*2π│               │
│  │   7. U-field bridge: u_inner→warp M_inner→block M_U     │               │
│  │      → Λ_φ^(U) = log(M_U)/ln(φ)−1/(2φ) → S(U)         │               │
│  │   8. Critic MLP + expl_bonus + res_bonus → reward       │               │
│  │   9. Hebbian update: w_cos, w_sin, w_sigma (block pool) │               │
│  │  10. Circular acc advance: acc = fmod(acc+r·dt·Ω, 2π)  │               │
│  │  11. Markov trit gate: φ+/φ-/R verdict (warp ballot)    │               │
│  │  12. Gate: verdict==ACCEPT && cos(acc)>0.5 && quorum≥2  │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Stream 1: Warp LL engine (hdgl_warp_ll_v33.cu)                            │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Per candidate (NTT_AUTO_THRESHOLD=400K dispatch):       │               │
│  │   gpucarry path (p < 400K): CUDA graph, ~3× faster NTT  │               │
│  │   NTT path (p ≥ 400K): length-128 NTT over M61          │               │
│  │   Verified: M_21701=PRIME (0.97s), M_44497=PRIME (3.41s)│               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Stream 2: Reward injection + weight sync                                   │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  hdgl_reward_inject_kernel: LL result → S¹ rotation (±Δacc) + r_h update│
│  │  hdgl_weight_sync_kernel:   global weight average        │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Optional: Continuous sieve (hdgl_sieve_v34.cu)                            │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Each slot permanently assigned exponent p_i             │               │
│  │  BASE_P = 82,589,934 (above last known Mersenne prime)   │               │
│  │  Convergence → ring buffer d_prime_found[256]            │               │
│  └──────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## X+1=0 Resonance Gate

The candidate gate is derived from the unified X+1=0 framework
(ref: zchg.org/t/x-1-0/955), which bridges Euler's identity, the golden ratio,
and physical operators Ω (Ohms) and C (Coulombs):

```
X + 1 = 0   ←→   e^(iπ) = 1/φ − φ = Ω·C² − 1
```

Since `|Ω·C²| = 1` (U(1) normalization), C² = 1/Ω and the product cancels,
reducing the resonance condition to:

```
S(p) = |e^(iπΛ_φ) + 1_eff(i)|
```

where:

| Symbol | Formula | Meaning |
|--------|---------|---------|
| `Λ_φ` | `log(p·ln2/lnφ) / lnφ − 1/(2φ)` | Mersenne exponent in φ-log space |
| `Ω` | `[1 + sin(π·{Λ_φ}·φ)] / 2` | Phase interference filter |
| `1_eff` | `1 + δ(i)` | Context-corrected unity |
| `δ(i)` | `\|cos(π·β·φ)\| · ln(n+2) / φ^(n+β)` | Lattice entropy correction |
| `β` | `{Λ_φ}` (fractional part) | Sub-node position |
| `n` | `⌊Λ_φ⌋` | Node index |

**Mersenne bridge**: `p·ln2/lnφ` maps the exponent of `2^p − 1` into φ-log space, so
the integer structure of primes aligns with the resonance nodes of the φ-field.

**Macro limit**: As `p → ∞`, `δ → 0` exponentially (φ^n grows faster than ln n),
and `S(p) → |e^(iπ·odd) + 1| = 0` — Euler's identity is the asymptotic attractor.

**Gate threshold**: `S < 0.25` — the field state must land within ±7° of a destructive
interference node (covers ~12.5% of phase space).

---

## File Reference

| File | Role | Key contents |
|------|------|--------------|
| `hdgl_analog_v33.cu` | GPU field kernel (base) | `wavelet_spectral_eval`, `critic_reward`, Markov trit gate, slow-sync, warp ballot |
| `hdgl_analog_v34.cu` | + Feistel phase | Feistel map on T² (STRIDE_A=89), KAPPA fix, golden-angle fixed-point destroyed |
| `hdgl_analog_v35.cu` | + U-field bridge + circular reward (latest) | M(U) warp/block reduce → Λ_φ^(U) → S(U); `phi_resonance_from_lambda()`; circular `reward_accum` on S¹; σ-trit→phvel; Wu Wei exploration bonus; Pluck kernel |
| `hdgl_warp_ll_v33.cu` | GPU LL engine | gpucarry CUDA graph (p<400K) + NTT over M61; `NTT_AUTO_THRESHOLD=400000`; `hdgl_gpucarry_ll_large()` for any p |
| `hdgl_critic_v33.c` | CPU MLP critic | 5→8→1 ReLU network, Welford normalisation, TD(0) SGD, replay buffer |
| `hdgl_critic_v33.h` | Critic API | `extern "C"` guards for .cu inclusion |
| `hdgl_sieve_v34.cu` | Continuous sieve | Slot-assigned exponents, BASE_P=82589934, ring buffer harvest; `hdgl_sieve_seed_priority()` for priority pre-seeding |
| `hdgl_psi_filter_v35.cu/.h` | 3-pass GPU Riemann psi pre-filter | B=500/3000/8000 zeros, SPIKE_THRESH=0.12/0.08; 8/8 prime exponents survive (TEST 7) |
| `hdgl_predictor_seed.c/.h` | Phi-lattice top-20 scorer | n(p) lower-half bias 9/11=82%, seeds sieve priority slots (TEST 8) |
| `hdgl_prismatic_v35.c/.h` | Prismatic resonance scorer + sort | P(p,r) log-scale; insertion sort descending; sieve pipeline refinement stage (TEST 8) |
| `hdgl_multigpu_v34.c` | Multi-GPU + CommLayerState (v6.c port) | Domain decomposition, P2P halo exchange, NCCL optional; CommLayerState async exponent ring (COMM_EXP_RING=16, push/poll/delta); `hdgl_multigpu_queue_exponent()` public API |
| `hdgl_host_v33.c` | CPU orchestration | 3-stream async loop, LL residue→critic reward, sieve pipeline (psi→prismatic→LL every 10 cycles) |
| `hdgl_bench_v33.cu` | Test + benchmark | 11 test suites (TEST 1–11), field throughput at N=256K/512K/1M |
| `hdgl_corpus_seeder.c/.h` | Codebase self-emission | 3-layer onion (MATH/CODE/BUILD) via fold26+megc; `hdgl_corpus_seeder_emit(out,cap)` / `hdgl_corpus_seeder_verify()`; no external deps |
| `hdgl_megc.c/.h` | MEGC codec (C, no external deps) | TernaryNode + BreathingEntropyCoder (arithmetic), GoldenContext (φ-freq model), DNA strand codec (AGTC); `megc_encode_field_dna()` / `megc_decode_field_dna()` |
| `hdgl_fold26.c/.h` | fold26 Delta+RLE adaptive compressor | 5 wu-wei strategies, CRC-32 inline; wire format: `F26\x02` + orig_size + crc32 + strategy + payload |
| `hdgl_onion.c/.h` | Onion Shell multi-layer wrapper | fold26-compresses MATH/CODE/BUILD layers independently; `onion_wrap()` / `onion_unwrap(target_id)` |
| `hdgl_phi_lang.h` | φ-language + spiral8 tables (header-only) | `PHI_LANG_TOKENS[4]` (Λ_φ/D_n/S_p/U-field), `SPIRAL8_AXES[8]`, lookup inlines |
| `phrase_extractor_to_json.py` | Phrase extractor + Rosetta encoder/decoder | Extracts top-500 project phrases; Rosetta protocol `0xFE HI LO` / `0xFF LEN bytes`; round-trip tests |
| `rosetta_stone.json` | Project phrase dictionary | 500 phrases → Base4096 token map; source: README + roadmap + all codebase headers |
| `frozen_base4096_alphabet.txt` | Canonical 4096-char Base4096 alphabet | OpenGL-friendly (0 combining marks, 0 BiDi overrides); verified identical to base4096-OpenGL-friendly.zip |
| `hdgl_session_handoff.py` | Session state encoder/decoder | 948-byte context → 539 Base4096 chars; `--roundtrip` / `--decode <token>` / `--stats`; round-trip PASS |
| `hdgl_selfprovision.ps1` | Self-provisioning wrapper | Runs bench (optional) then emits Base4096 session token; `-TokenOnly` / `-Decode <token>` |

### Version history

| Version | Platform | Key additions |
|---------|----------|---------------|
| v30/v30b | C (CPU) | Original GRA field, Fibonacci tables, MPI stub |
| v31 | C (CPU) | Carry-save LL, GRA plasticity, resonance clustering |
| v32 | CUDA | First GPU port, warp spectral pooling, 4-harmonic coupling |
| v33 analog | CUDA | Wavelet spectral basis (4-scale Morlet), learned MLP critic, Markov trit gate, slow-sync |
| v33 warp_ll | CUDA | gpucarry CUDA graph + NTT_AUTO_THRESHOLD=400K; M_21701 verified |
| v34 sieve | CUDA | Continuous sieve, multi-GPU skeleton (BASE_P=82589934) |
| v34 analog | CUDA | Feistel phase map on T² (STRIDE_A=89), KAPPA·log(p) fix, golden-angle trap eliminated |
| v35 analog | CUDA | Full U-field bridge: M(U) warp+block reduce → Λ_φ^(U) from field state → S(U) |
| v35b pipeline | CUDA+C | psi_filter (GPU, 3-pass Riemann) + predictor_seed (CPU, top-20 phi-lattice) + prismatic (CPU, log-score sort) + sieve priority seed; full pipeline wired in host |
| v36 LL large | CUDA | `hdgl_gpucarry_ll_large(p)` in warp_ll_v33.cu; TEST 9: M_9941 PRIME (0.23s), M_9949 COMPOSITE; 27/28 pass |
| Ev1 session stack | Python/C | `hdgl_megc.c` + `hdgl_fold26.c` + `hdgl_onion.c` + `hdgl_phi_lang.h` + `rosetta_stone.json` + `hdgl_session_handoff.py` + `hdgl_selfprovision.ps1`; Ev1 Long done |
| v37 analog LL | C | `ll_analog.c`/`ll_analog.h`: 8D Kuramoto + exact APA squaring; MSVC `__int128`→`_umul128` shim; TEST 10: M_127/M_89 PRIME, M_131 COMPOSITE; 30/31 pass |
| v38 empirical beta | C | `empirical_validation.c`: BigG/Fudge10 chi²≈0, 100% CODATA; `DN_EMPIRICAL_BETA=0.360942` → `hdgl_phi_lang.h` → `dn_amp()` calibrated; 30/31 pass |
| v40 circular reward | CUDA+C | `reward_accum` on S¹: `acc = fmod(acc + reward·dt·REWARD_ANG_RATE, 2π)`; gate fires on `cos(acc) > GATE_COS_THRESH=0.5`; σ-trit→phvel (from conscious_fused_engine.cu); Wu Wei exploration bonus `EXPL_BONUS_F·max(0,−cos(acc))` (max in cold half, pulls back to gate); Pluck kernel (random phase + random acc); critic checkpoint save/load; phi-lattice n-coord logging; stall detection; **97k candidates / 600 cycles, no freeze; 34/34 pass** |

---

## Build

### Requirements

- CUDA Toolkit (tested: 13.2)
- MSVC 2017 BuildTools (Windows) — or any MSVC/GCC on Linux
- GPU: sm_75 or higher (tested: RTX 2060)

### Windows

```powershell
# Add MSVC to PATH (VS2017 — adjust path for VS2019/2022)
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64;$env:PATH"

cd REPO/FILE/LOCATION(CHANGE_THIS)

# Main executable
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler -lineinfo `
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu `
  hdgl_host_v33.c hdgl_critic_v33.c `
  -o hdgl_v33.exe

# Test + benchmark harness (v33 analog baseline)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_critic_v33.c hdgl_bench_v33.cu `
  -o hdgl_bench.exe

# Feistel phase kernel (v34 — swap analog file)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v34.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_critic_v33.c hdgl_bench_v33.cu `
  -o hdgl_bench_v34.exe

# Full U-field bridge (v35)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_critic_v33.c hdgl_bench_v33.cu `
  -o hdgl_bench_v35.exe

# v35b full pipeline (psi filter + predictor seed + prismatic scorer)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_psi_filter_v35.cu hdgl_predictor_seed.c hdgl_prismatic_v35.c `
  hdgl_critic_v33.c hdgl_bench_v33.cu `
  -o hdgl_bench_v35b.exe

# Current executable (v40 — circular reward, all exploration dynamics)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_psi_filter_v35.cu hdgl_predictor_seed.c hdgl_prismatic_v35.c `
  hdgl_critic_v33.c ll_analog.c `
  hdgl_corpus_seeder.c hdgl_fold26.c hdgl_onion.c hdgl_megc.c `
  hdgl_host_v33.c `
  -o hdgl_run.exe

# v40 bench (CURRENT — 34/34 pass)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_psi_filter_v35.cu hdgl_predictor_seed.c hdgl_prismatic_v35.c `
  hdgl_critic_v33.c ll_analog.c `
  hdgl_corpus_seeder.c hdgl_fold26.c hdgl_onion.c hdgl_megc.c `
  hdgl_bench_v33.cu `
  -o hdgl_bench_v40.exe

# v39 bench (previous stable baseline)
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_psi_filter_v35.cu hdgl_predictor_seed.c hdgl_prismatic_v35.c `
  hdgl_critic_v33.c ll_analog.c `
  hdgl_corpus_seeder.c hdgl_fold26.c hdgl_onion.c hdgl_megc.c `
  hdgl_bench_v33.cu `
  -o hdgl_bench_v39.exe

# Session token (Ev1) -- no CUDA needed
py codebase\phrase_extractor_to_json.py          # (re)build rosetta_stone.json
py codebase\hdgl_session_handoff.py --roundtrip   # verify
py codebase\hdgl_session_handoff.py --stats       # encode + print stats
.\codebase\hdgl_selfprovision.ps1 -TokenOnly      # token only
.\codebase\hdgl_selfprovision.ps1                 # bench + token
```

### Linux / WSL

```bash
nvcc -O3 -arch=sm_75 -lineinfo \
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu \
  hdgl_host_v33.c hdgl_critic_v33.c \
  -o hdgl_v33 -lm

nvcc -O3 -arch=sm_75 \
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu \
  hdgl_critic_v33.c hdgl_bench_v33.cu \
  -o hdgl_bench -lm
```

> **Arch flag**: For sm_86 (RTX 3000), sm_89 (RTX 4000), or sm_90 (H100), replace
> `-arch=sm_75` accordingly and drop `-allow-unsupported-compiler`.

---

## Run

### Tests and benchmarks

```
.\hdgl_bench.exe
```

Expected output (RTX 2060, April 2026, hdgl_bench_v35b.exe):

```
HDGL v33 Test & Benchmark Harness
CUDA device: NVIDIA GeForce RTX 2060  cc=7.5  smem=12287 KB  sharedPerBlock=48 KB

=== TEST 1: Critic (CPU) ===               [PASS ×5]
=== TEST 2: Warp LL v33 (GPU) ===          [PASS ×2]  residue ≈ 0.51 (not a Mersenne prime)
=== TEST 3: Sieve v34 (GPU) ===            [PASS ×3]
=== TEST 4: Field kernel v35 (GPU) ===    [PASS ×2]  no NaN/Inf; candidates=2505 ≤ warp-max (20480); 256 stored
=== TEST 5: gpucarry correctness ===       [PASS ×2]  11 primes + 5 composites verified
=== TEST 6: gpucarry timing ===            [PASS ×1]  M_21701: 0.97s, M_44497: 3.41s
=== TEST 7: psi filter (GPU) ===           [PASS ×4]  8/8 prime exponents survive, composites suppressed
=== TEST 8: prismatic scorer (CPU) ===     [PASS ×4]  9/11 lower-half bias, sort correct, top-20 seeded
=== TEST 9: LL large-p (GPU) ===           [PASS ×2]  M_9941 PRIME (0.23s), M_9949 COMPOSITE (0.23s)
=== TEST 10: 8D Kuramoto analog LL ===     [PASS ×3]  M_127 PRIME, M_131 COMPOSITE, M_89 PRIME; osc LOCKED+residue=0
=== TEST 11: Gate convergence (CPU) ===    [PASS ×3]  V(prime) −0.17→+0.07, V(comp) +0.15→−0.01; V(prime)>V(comp)

BENCHMARK:
  N=262144   500 steps   ~310 ms   0.42 GSlots/s
  N=524288   500 steps   ~620 ms   0.42 GSlots/s
  N=1048576  500 steps   ~1240 ms  0.42 GSlots/s
```

### Main search loop

```
.\hdgl_v33.exe [N] [S] [cycles] [p_bits]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| N | 65536 | Number of field slots |
| S | 256 | Grid stride (2D layout: S × N/S) |
| cycles | 100 | Synchronisation cycles to run |
| p_bits | 127 | Bit-width for LL verification (p_bits-bit Mersenne exponent) |

Example — short smoke run:

```powershell
.\hdgl_v33.exe 65536 256 100 127
```

---

## Key Algorithms

### NTT-based squaring (O(n log n))

Lucas-Lehmer requires repeated squaring of 4096-bit numbers (64 × 64-bit limbs).
The warp LL engine uses a length-128 NTT over the Mersenne prime field M61 = 2^61 − 1:

```
mulmod61(a, b):
    hi = __umul64hi(a, b)          // MSVC-compatible (no __uint128_t)
    lo = a * b
    r  = (hi << 3) | (lo >> 61) + (lo & M61)
    if r >= M61: r -= M61
```

The 128th root of unity `ω_128 = 37^((2^61−2)/128) mod M61 = 0x00C4E24A6DB3EEB3` is
precomputed. NTT butterfly exchanges use `__shared__ uint64_t sh_ntt[128]` so all 128
threads collaborate on a single transform with `__syncthreads()` barriers.

### Carry-save LL state

LL state is kept as two 64-limb arrays (S, C) where the logical value is S+C.
Squaring expands to `(S+C)² = S² + 2SC + C²` in carry-save form, deferring full
carry propagation to the residue-check step only. This eliminates `O(LIMBS)` carry
chains on every iteration. State lives in `__shared__` memory to avoid register spill.

### Wavelet spectral coupling

Each slot couples to 4 von Neumann neighbours via 4-scale Morlet-like wavelets:

```
ψ_k(φ) = cos(2^k · φ) · exp(−φ² / 2σ_k²)
```

Weights `w_cos[k]`, `w_sin[k]`, `σ_k` are per-slot and updated each step via
Hebbian learning with block-pooled gradients (70% local, 30% block average).

### MLP critic (TD(0))

A 5→8→1 ReLU network runs on CPU, observing `[residue, coherence, amp, r_h, acc]`
per synchronisation cycle. Weights are packed into 57 floats and uploaded to
`__constant__ float g_critic_w[57]`. The GPU infers reward values per-slot in the
field kernel at full SM throughput with no CPU round-trip.

### Circular reward accumulator (v40)

Prior to v40, `reward_accum` was a scalar: `acc = DECAY·acc + reward`. When the critic
issued negative value estimates, acc drifted below `CAND_ACCUM_THRESH` and the gate
went permanently silent — a linear attractor with no recovery mechanism.

In v40, `reward_accum` is a **phase angle on S¹**:

```
acc  = fmod(acc + reward · dt · REWARD_ANG_RATE, 2π)    [S¹ advance each step]

gate = cos(acc) > GATE_COS_THRESH                        [fires in ±60° arc]
     ≡ acc ∈ [0, π/3] ∪ [5π/3, 2π]

expl_bonus = EXPL_BONUS_F · max(0, −cos(acc))            [max at acc=π, 0 in good arc]

post-gate:  acc += ARC_KICK                              [rotate out → refractory period]
Pluck:      acc  = uniform([0, 2π])                     [random, not 0, to avoid re-sync]
LL prime:   acc += 0.5 rad                               [rotate toward arc]
LL composite: acc −= 0.3 rad                             [rotate away from arc]
```

**Key properties:**
- **No threshold drift** — every slot reaches the gate arc periodically regardless of
  critic output, since the exploration bonus is maximum in the "cold" half (`acc ≈ π`)
  and drives rotation back toward the gate
- **Natural refractory period** — `ARC_KICK` rotates the slot out of the good arc
  immediately after firing, preventing continuous self-excitation
- **Desynchronized** — `reward_accum` is initialized uniformly on `[0, 2π]` and Pluck
  assigns random values; all slots advance at different rates due to different rewards,
  producing continuous gate firing rather than synchronized bursts

**Complementary exploration mechanisms:**
- **σ-trit → phvel** (from `conscious_fused_engine.cu`): `phvel += SIGMA_TRIT_GAIN_F · σ_trit` — Markov trit fed back into phase velocity, maintaining phase variance
- **Wu Wei exploration bonus**: `EXPL_BONUS_F · max(0, −cos(acc))` — maximum in cold half, self-quenching in good arc (no manual quench point needed)
- **Pluck kernel**: when stall detected (no new candidates for `STALL_DETECT_CYCLES`), all slots get random phase + random acc — breaks amplitude homogenization (the root cause of trit freezing)

**Measured result (April 2026, RTX 2060):**

| Metric | Scalar acc (v39) | Circular acc (v40) |
|--------|------------------|--------------------|
| Candidates @ cycle 600 | ~38,000 (frozen ~cycle 200) | **97,053 (continuous growth)** |
| Pluck pulses fired | 8 | 3 |
| Freeze? | Yes, cycle ~200 | No |
| 34/34 bench | Pass | Pass |

### φ-resonance gate (v35 — full U-field bridge)

In v35, Λ_φ is derived from field state rather than directly from the exponent proxy:

```c
// [U-field bridge] Inner φ-exponent (per thread)
// interaction(U_i, U_j) = 0.5 · phj_n  (Feistel coupling, already computed)
float u_inner = expf((0.5f * phj_n + KAPPA_PHI * logf(r_h + 1.0f)) * LN_PHI_F);

// Level 1 — warp reduce (32 lanes → M_inner)
float ui_sum = u_inner;
for (int off = 16; off > 0; off >>= 1) ui_sum += __shfl_down_sync(mask, ui_sum, off);
float M_inner = ui_sum / 32.0f;       // broadcast from lane 0
float u_mid   = expf(M_inner * LN_PHI_F);  // φ^(M_inner)

// Level 2 — block reduce (8 warps → M_U via sh_u_mid[8])
// ... (lane 0 of each warp writes, thread 0 sums)
float lambda_phi_U = logf(M_U) / LN_PHI_F - INV_2PHI_F;  // Λ_φ^(U)

// S(U) = |e^(iπΛ_φ^(U)) + 1_eff|  — full X+1=0 gate from field state
float resonance = phi_resonance_from_lambda(lambda_phi_U);
```

The verdict is a **Markov trit** (not a hardcoded threshold):

```c
// Trit probabilities from warp amplitude statistics
// phi_pos, phi_neg, gamma_v = block-averaged trit fractions
if      (phi_neg > 0.45f)                               verdict = REJECT;
else if (1.2f*phi_neg + 0.8f*gamma_v - phi_pos > 0.6f)  verdict = REJECT;
else if (phi_pos > 0.25f)                               verdict = ACCEPT;  // 0.25: equilibrium trit dist
else                                                    verdict = UNCERTAIN;

// Circular reward accumulator on S¹ (replaces scalar threshold acc > 5.0)
// Gate fires when acc is in the "good arc": cos(acc) > GATE_COS_THRESH
// ↔ acc ∈ [0, π/3] ∪ [5π/3, 2π]  — no threshold drift possible
bool promoted = (verdict == ACCEPT) && (cosf(acc) > GATE_COS_THRESH)
             && (cluster_size >= CLUSTER_QUORUM);
```

### Session Memory Compression (Ev1)

The codebase carries its own session context as a single portable Base4096 block:

```
Concept text  →  Rosetta encode  →  Base4096 wire
   (UTF-8)         (0xFE HI LO       (12 bits/char,
                   = phrase id;       3 bytes → 2 Unicode
                   0xFF LEN bytes     glyphs, frozen
                   = raw word)        alphabet)
```

**Files:**

| File | Role |
|------|------|
| `phrase_extractor_to_json.py` | Builds `rosetta_stone.json` from project sources |
| `rosetta_stone.json` | 500 phrase-to-token pairs (top project vocabulary) |
| `frozen_base4096_alphabet.txt` | Canonical 4096-char alphabet (OpenGL-friendly, collision-free) |
| `hdgl_session_handoff.py` | Encodes SESSION_STATE → Base4096; `--decode`, `--stats`, `--roundtrip` |
| `hdgl_selfprovision.ps1` | Wrapper: optional bench run + session token to stdout |
| `hdgl_megc.c/.h` | MEGC: TernaryNode + BEC arithmetic codec + DNA strand layer |
| `hdgl_fold26.c/.h` | fold26: 5-strategy Delta+RLE adaptive compressor, CRC-32 |
| `hdgl_onion.c/.h` | Onion Shell: MATH/CODE/BUILD layer wrapping via fold26 |

**Measured compression (April 2026):**
- Input: 948 UTF-8 bytes of project-specific context
- Output: 539 Base4096 characters
- Round-trip: PASS (exact decode of all normalised tokens)
- Compression win: ~43% character reduction vs. plain text; all project-vocabulary phrases
  (Lambda_phi, mulmod61, gpucarry, Mersenne prime, etc.) hit dictionary in 2–3 chars

---

## Performance Notes

| Metric | Value |
|--------|-------|
| GPU | RTX 2060 (sm_75, 1920 CUDA cores) |
| Peak field throughput | ~0.35 GSlots/s (N=65536, v40, measured April 2026) |
| LL: gpucarry (p<400K) | M_21701=PRIME in 0.97s; M_44497=PRIME in 3.41s |
| LL: NTT path (p≥400K) | 128-NTT over M61, carry-save, 4096-bit |
| Critic inference | on-device, ~57 FMADs/slot |
| φ-resonance gate | Markov trit (φ+/φ-/R verdict) + U-field Λ_φ^(U) |
| U-field reduce | warp __shfl (32→1) + block sh_u_mid[8] (8→1), +32B smem |
| Candidate rate (v40, 600 cycles) | 97k candidates, continuous growth, no freeze |
| Gate condition | cos(acc) > 0.5 (S¹ circular accumulator, ±60° arc) |
| Exploration | σ-trit→phvel + Wu Wei bonus + Pluck (random phase injection) |
| Residue normalization | norm / (LIMBS × M61), range [0,1] confirmed |

Throughput is compute-bound at N≥256K. At smaller N, launch overhead dominates.

---

## Mathematical Foundation

### Golden Recursive Algebra (GRA)

The field uses φ as the sole primitive constant. All physical constants are treated
as emergent from the D(n,β) coordinate system:

```
D(n,β) = √(φ · F_{n+β} · 2^{n+β} · P_{n+β} · Ω) · r^k
```

where F is the continuous Fibonacci interpolation and P is the prime at position n+β.
This maps SI-unit dimensions to radial positions in a φ-recursive tree with no common
root (the "rootless tree" structure in the framework document).

### X+1=0 unified form

The complete bridge between Euler, φ, and physical operators (Steps 0–6 in the
framework):

```
e^(iπ) = 1/φ − φ = Ω·C² − 1

⟹  X + 1 = 0  bridges:
    • Euler's identity (e^(iπ) + 1 = 0)
    • Golden ratio recursion (1/φ − φ + 1 = 0)
    • Physical closure (|Ω·C²| = 1, U(1)-normalized)
```

The unified lattice operator at step i:

```
L_i = √(φ · F_{n,β} · b^m · φ^k · Ω_i) · r_i^{−1}   [magnitude / scaling]
    + 1_eff(i) · e^(iπθ_i)                             [phase / coordinate]
```

At macro scales (n → ∞), `δ(i) → 0` and the classical ±1 and 0 emerge naturally from
the phase lattice embedding — no constants are assumed, only φ.

### U-field bridge (implemented in hdgl_analog_v35.cu)

The v35 kernel closes the loop from field state directly to Λ_φ, replacing the
exponent-proxy path with a full warp+block reduce:

```
u_inner = φ^(0.5·phj_n + κ·log(r_h+1))          [per thread; interaction U_i·U_j]
M_inner = warp_mean(u_inner)                       [warp reduce, 32→1 via __shfl]
u_mid   = φ^(M_inner)                              [nested φ-exponent]
M_U     = block_mean(u_mid)                        [block reduce via sh_u_mid[8]]

Λ_φ^(U) = log(M_U) / ln(φ) − 1/(2φ)              [Mersenne φ-log projection]
S(U)    = |e^(iπΛ_φ^(U)) + 1_eff|                [X+1=0 discriminant from field state]

Prime invariant: coherent field → u_inner uniform across warp → M_U stable → S(U) minimum
```

**Status: TEST 1–11 pass — RTX 2060 (sm_75) — hdgl_bench_v40.exe: 34/34 pass**
**v35b pipeline: psi filter · phi-lattice predictor · prismatic scorer**
**v36: `hdgl_gpucarry_ll_large()` — multi-limb LL verified (M_9941 PRIME 0.23s, M_9949 COMPOSITE 0.23s)**
**Ev1 Long [x]: Base4096 session handoff (948 bytes → 539 chars, round-trip PASS) · self-provisioning · Rosetta Stone (500 phrases) · `hdgl_corpus_seeder.c` — codebase self-emission (onion/fold26, 3-layer MATH/CODE/BUILD)**
**Ev2 Long [x]: `ll_analog.c` — 8D Kuramoto analog path (CUDA-free); TEST 10: M_127/M_89 PRIME, M_131 COMPOSITE, osc LOCKED + residue=0 confirmed · CommLayerState (v6.c port) in multigpu_v34: async exponent ring, push/poll/delta**
**Ev3 Long [x]: `empirical_validation.c` — BigG/Fudge10 PASSED (χ²_red≈0, 100% CODATA); beta=0.360942 → DN_EMPIRICAL_BETA → dn_amp calibrated · TEST 11: gate converges — V(prime) −0.17→+0.07, V(comp) +0.15→−0.01 after 500 TD(0) cycles · TEST 4 fixed: warp-election bound (hdgl_bench_v39.exe: 34/34 pass)**
**Ev4 Long [x]: Circular reward accumulator on S¹ — `reward_accum` is now a phase angle ∈ [0,2π]; gate fires when cos(acc)>0.5; no freeze past cycle 400 (97k candidates/600 cycles). σ-trit→phvel, Wu Wei exploration bonus, Pluck kernel (random phase injection). hdgl_bench_v40.exe: 34/34 pass**

---

## Known Limitations

- **Sieve candidates**: 0 exponents harvested from the sieve after 200 steps. The 32-bit
  LL proxy state converges too slowly for the ~83M exponent range. A longer run or
  a deeper LL proxy (more bits) is needed.

- **Field candidates at N=65K (bench)**: In TEST 4 (10 steps cold start), the field
  promotes candidates via warp-election (≤ N/32 per step). For the main search loop,
  run `cycles=200+` so `acc` accumulates past the 5.0 threshold — or lower N to reduce
  launch overhead at shorter runs.

- **vs. GIMPS**: This is a research prototype, not production. GIMPS uses PRP tests
  and error-checked double-precision arithmetic. This engine targets novel candidate
  *pre-screening* via field resonance, not replacement of verified LL testing.

- **Multi-GPU**: `hdgl_multigpu_v34.c` compiles but is not tested beyond single-GPU.
  NCCL path requires `-DUSE_NCCL` and a NCCL installation.

---

## Related Documents

| File | Contents |
|------|----------|
| `roadmap.md` | Development roadmap (Ev1/2/3 Short/Medium/Long checkboxes) |
| `x+1=0-roadmap-fine-tune-candidates.md` | X+1=0 framework, D(n,β) coordinates, supernova fit, MEGC codec, φ-language |
| `supplementals.md` | Additional notes |
| `spinning the lattice.md` | Lattice spin model notes |
| `Relative Planck's Constant Proportions.html` | Emergent constants analysis |
| `base4096-dict-builder.md` | Rosetta Stone + Base4096 compression stack plan |
| `codebase/rosetta_stone.json` | 500-phrase project vocabulary → Base4096 token map |
| `codebase/hdgl_session_handoff.py` | Encode/decode full project context as single Base4096 block |
| `codebase/hdgl_selfprovision.ps1` | Run bench and emit session token automatically |

# Frontier Research Files
hdgl_analog_v37.cu, hdgl_analog_v39.cu, hdgl_analog_nullweaver.cu (Nullweaver Ω BEAST MODE), hdgl_analog_nullweaver-v2.cu (Nullweaver Ω BEAST MODE — FULL EDITION)

<img width="1040" height="528" alt="image" src="https://github.com/user-attachments/assets/7cb672c0-6bb0-4364-bee7-88c240f37f3d" />

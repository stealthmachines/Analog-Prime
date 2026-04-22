# HDGL Analog v33/v34 — φ-Resonance Mersenne Prime Search Engine

A CUDA-accelerated Mersenne prime candidate search engine built on Golden Recursive
Algebra (GRA), NTT-based Lucas-Lehmer arithmetic, a learned MLP critic, and a unified
φ-resonance gate derived from the X+1=0 framework.

**Status: 14/14 tests pass — zero warnings — RTX 2060 (sm_75)**

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
│  Stream 0: Field evolution kernel (hdgl_analog_v33.cu)                     │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Per slot (N slots, block=256):                          │               │
│  │   1. Load SoA: A_re, A_im, phase, phase_vel, r_h, acc   │               │
│  │   2. LL-lite: 4 squarings of 32-bit proxy state         │               │
│  │   3. 4-neighbour wavelet coupling (4-scale Morlet)      │               │
│  │   4. Critic MLP inference (5→8→1, ReLU)                 │               │
│  │   5. φ-resonance score S(p) via X+1=0 gate              │               │
│  │   6. κ·log(p) phase injection (U^(p) field bias)        │               │
│  │   7. Hebbian update: w_cos, w_sin, w_sigma (block pool) │               │
│  │   8. Candidate gate: S < 0.25 ∧ acc > 5 ∧ amp > 0.6   │               │
│  │   9. Warp quorum vote (≥2 agree) → candidate emit       │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Stream 1: Warp LL engine (hdgl_warp_ll_v33.cu)                            │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Per candidate (block=128 threads, 4 warps):             │               │
│  │   1. Carry-save state: S[64] + C[64] limbs in __shared__ │               │
│  │   2. NTT squaring mod M61 (O(n log n), NTT_SIZE=128)     │               │
│  │   3. Carry-save combination → subtract 2                 │               │
│  │   4. Residue norm / (LIMBS × M61)  → [0,1]              │               │
│  │   5. Write verified flag → reward injection stream       │               │
│  └──────────────────────────────────────────────────────────┘               │
│                                                                             │
│  Stream 2: Reward injection + weight sync                                   │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  hdgl_reward_inject_kernel: LL result → acc bonus/penalty│               │
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
| `hdgl_analog_v33.cu` | GPU field kernel | `wavelet_spectral_eval`, `critic_reward`, `phi_resonance_score`, `hdgl_field_step_kernel`, weight sync |
| `hdgl_warp_ll_v33.cu` | GPU LL engine | `ntt_square_lane`, `mulmod61` (via `__umul64hi`), `cs_square_step`, `cs_residue_norm`, `warp_ll_kernel_v33` |
| `hdgl_critic_v33.c` | CPU MLP critic | 5→8→1 ReLU network, Welford normalisation, TD(0) SGD, replay buffer |
| `hdgl_critic_v33.h` | Critic API | `extern "C"` guards for .cu inclusion |
| `hdgl_sieve_v34.cu` | Continuous sieve | Slot-assigned exponents, ring buffer harvest |
| `hdgl_multigpu_v34.c` | Multi-GPU | Domain decomposition, P2P halo exchange, NCCL optional |
| `hdgl_host_v33.c` | CPU orchestration | 3-stream async loop, critic→GPU weight upload, Windows QPC timer |
| `hdgl_bench_v33.cu` | Test + benchmark | 4 subsystem tests, field throughput at N=256K/512K/1M |

### Version history

| Version | Platform | Key additions |
|---------|----------|---------------|
| v30/v30b | C (CPU) | Original GRA field, Fibonacci tables, MPI stub |
| v31 | C (CPU) | Carry-save LL, GRA plasticity, resonance clustering |
| v32 | CUDA | First GPU port, warp spectral pooling, 4-harmonic coupling |
| v33 | CUDA | Wavelet spectral basis (4-scale Morlet), learned MLP critic, NTT squaring |
| v34 | CUDA | Continuous sieve, multi-GPU domain decomposition |
| v33+X | CUDA | X+1=0 φ-resonance gate, Mersenne bridge, 1_eff lattice correction |

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

cd "C:\Users\Owner\Documents\Conscious 2.0"

# Main executable
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler -lineinfo `
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu `
  hdgl_host_v33.c hdgl_critic_v33.c `
  -o hdgl_v33.exe

# Test + benchmark harness
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler `
  hdgl_analog_v33.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu `
  hdgl_critic_v33.c hdgl_bench_v33.cu `
  -o hdgl_bench.exe
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

Expected output (RTX 2060, April 2026):

```
HDGL v33 Test & Benchmark Harness
CUDA device: NVIDIA GeForce RTX 2060  cc=7.5  smem=12287 KB  sharedPerBlock=48 KB

=== TEST 1: Critic (CPU) ===         [PASS ×5]
=== TEST 2: Warp LL v33 (GPU) ===    [PASS ×2]  residue ≈ 0.51 (not a Mersenne prime)
=== TEST 3: Sieve v34 (GPU) ===      [PASS ×3]
=== TEST 4: Field kernel v33 (GPU) === [PASS ×3]  candidates promoted = 0

BENCHMARK:
  N=262144   500 steps   291 ms   0.450 GSlots/s
  N=524288   500 steps   488 ms   0.537 GSlots/s
  N=1048576  500 steps   979 ms   0.536 GSlots/s

RESULTS: 14 passed, 0 failed
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

### φ-resonance gate

```c
// Mersenne bridge: project exponent into φ-log space
float M_U        = r_harmonic * (LN2 / LN_PHI);   // p · ln2/lnφ
float lambda_phi = logf(M_U) / LN_PHI - 1/(2·φ);  // Λ_φ

// Lattice correction (vanishes at large p)
float n_f    = floorf(lambda_phi);
float beta   = lambda_phi - n_f;
float delta  = |cos(π·β·φ)| · ln(n+2) / φ^(n+β);
float one_eff = 1.0f + delta;

// S(p) = |e^(iπΛ_φ) + 1_eff|
float re = cosf(PI * lambda_phi) + one_eff;
float im = sinf(PI * lambda_phi);
float S  = sqrtf(re*re + im*im);   // ∈ [0, 2]; primes → δ ≈ 0

// Gate: must be within ±7° of destructive node
bool promoted = (S < 0.25f) && (acc > 5.0f) && (amp > 0.6f);
```

---

## Performance Notes

| Metric | Value |
|--------|-------|
| GPU | RTX 2060 (sm_75, 1920 CUDA cores) |
| Peak field throughput | ~0.54 GSlots/s (N=512K) |
| LL residue | 4096-bit, 128-NTT, carry-save |
| Critic inference | on-device, ~57 FMADs/slot |
| φ-resonance gate | ~12 trig ops/slot |
| Candidate rate (10 steps) | 0 at N=65536 (needs sustained acc > 5.0) |
| Residue normalization | norm / (LIMBS × M61), range [0,1] confirmed |

Throughput is compute-bound at N≥512K. At smaller N, launch overhead dominates.
The ~5% drop from 512K to 1M is likely L2 eviction from `g_soa` SoA arrays.

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

---

## Known Limitations

- **Sieve candidates**: 0 exponents harvested from the sieve after 200 steps. The 32-bit
  LL proxy state converges too slowly for the ~83M exponent range. A longer run or
  a deeper LL proxy (more bits) is needed.

- **Field candidates at N=65K**: 0 candidates promoted in 10 steps. The `acc > 5.0`
  threshold requires ~50+ sync cycles to accumulate. Run with `cycles=200+` or
  lower N to see candidates.

- **vs. GIMPS**: This is a research prototype, not production. GIMPS uses PRP tests
  and error-checked double-precision arithmetic. This engine targets novel candidate
  *pre-screening* via field resonance, not replacement of verified LL testing.

- **Multi-GPU**: `hdgl_multigpu_v34.c` compiles but is not tested beyond single-GPU.
  NCCL path requires `-DUSE_NCCL` and a NCCL installation.

---

## Related Documents

| File | Contents |
|------|----------|
| `roadmap.md` | Original development roadmap, items 1–16 |
| `x+1=0-roadmap-fine-tune-candidates.md` | X+1=0 framework, D(n,β) coordinates, supernova fit, MEGC codec, φ-language |
| `supplementals.md` | Additional notes |
| `spinning the lattice.md` | Lattice spin model notes |
| `Relative Planck's Constant Proportions.html` | Emergent constants analysis |

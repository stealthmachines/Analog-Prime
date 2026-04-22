# conscious — Quantum Prime & Phi-Lattice Cryptographic Platform
*by [zchg.org](https://zchg.org)*

> A self-correcting spectral Markov dynamical sieve, Mersenne prime engine,
> and cryptographically complete proprietary OS/kernel platform — running on
> GPU warps, CPU phi-lattice math, and an interactive Windows TUI with no
> external crypto dependencies.

**GPU target:** NVIDIA RTX 2060 12 GB (CUDA sm_75) · **OS:** Windows 10/11 · **Compiler:** clang 14+

---

## Table of Contents

1. [What It Is](#what-it-is)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [prime\_ui.exe — Interactive TUI](#prime_uiexe--interactive-tui)
   - [Prime Math Modules (1–5)](#prime-math-modules-15)
   - [OS & Lattice Modules (6–9, A)](#os--lattice-modules-69-a)
   - [Crypto Platform Modules (B–S)](#crypto-platform-modules-bs)
6. [Crypto Platform Deep Dive](#crypto-platform-deep-dive)
   - [PhiKernel lk\_ API](#phikernel-lk_-api)
   - [phi\_fold Hash Family](#phi_fold-hash-family)
   - [phi\_stream AEAD Cipher](#phi_stream-aead-cipher)
   - [Nonlinear S-box (NLSB)](#nonlinear-s-box-nlsb)
   - [Multi-Source Entropy](#multi-source-entropy)
   - [Wu-Wei Codec](#wu-wei-codec)
   - [PhiSign](#phisign)
   - [Observer / MITM View](#observer--mitm-view)
7. [Lucas-Lehmer Verifier (ll\_mpi.exe)](#lucas-lehmer-verifier-ll_mpiexe)
8. [GPU Engine (conscious.exe)](#gpu-engine-consciousexe)
9. [Slot4096 Alpine OS](#slot4096-alpine-os)
10. [Additional Tracks](#additional-tracks)
11. [Benchmarks](#benchmarks)
12. [Mathematical Foundation](#mathematical-foundation)
13. [Security Architecture](#security-architecture)
14. [Use Cases](#use-cases)
15. [Requirements](#requirements)
16. [Build Reference](#build-reference)

---

## What It Is

**conscious** is a Windows-native research platform combining three things that
do not normally share a codebase:

1. **A Mersenne prime research engine** — exact-integer Lucas-Lehmer verification
   with warp-parallel GPU squaring, NTT path, and a CUDA-free 8D Kuramoto analog path.

2. **A cryptographically complete proprietary OS/kernel** — all cryptographic
   primitives (hash, AEAD, signing, key derivation, capability tokens, PCR chains)
   implemented entirely via phi-lattice mathematics. No SHA, no AES, no XOR in
   any live crypto path. No bcrypt. No external crypto library of any kind.

3. **A lattice-derived Alpine Linux container** — an entire OS whose hostname,
   packages, timezone, user UID, shell limits, and ulimits are deterministically
   derived from a single 4096-slot phi-irrational resonance lattice. No config
   files. The OS is a mathematical consequence of the lattice state.

All three are accessible from a single self-contained `prime_ui.exe` — a Windows
TUI with 23 interactive modules, zero DLL dependencies beyond MSVCRT + kernel32.

---

## Architecture Overview

```
prime_ui.exe  (Windows TUI — 23 modules, single binary, zero external deps)
      │
      ├── Prime Math (1–5)         sieve · Miller-Rabin · Mersenne · zeta zeros · benchmark
      │
      ├── OS / Lattice (6–9, A)    Alpine install · lattice shell · kernel build · scheduler
      │
      └── Crypto Platform (B–S)    the phi-native cryptographic kernel
            │
            ├── [B]  Lattice Benchmark      phi-native perf profile
            ├── [C]  Crystal-Native Init    quartz crystal → lattice → OS
            ├── [D]  Phi-Native Engine      AVX2+FMA3 phi-resonance compute peak
            ├── [E]  Crypto Layer           SHA-256/HMAC/HKDF lattice CSPRNG
            ├── [F]  Full Crypto Stack      AES-256-GCM · X25519 · Noise_XX (lattice-keyed)
            ├── [G]  Wu-Wei Codec           fold26 lattice-adaptive compression
            ├── [H]  PhiSign               Ed25519 twisted Edwards sign/verify
            ├── [I]  Lattice Kernel         lk_read · lk_advance · lk_commit
            ├── [J]  Kernel Benchmark       throughput + correctness for all lk_ primitives
            ├── [K]  Lattice Gateway        wu-wei + lk — cryptographic I/O choke point
            ├── [L]  PhiHash               phi_fold_hash32/64 — no SHA, no XOR
            ├── [M]  PhiStream             additive stream cipher — no AES, no XOR
            ├── [N]  PhiVault              lattice-native key-value vault
            ├── [O]  PhiChain              phi_fold PCR chain — no SHA
            ├── [P]  PhiCap               phi capability tokens
            ├── [R]  PhiKernel             fully phi-native kernel demo (no SHA/AES/BCrypt/XOR)
            └── [S]  Observer/MITM View    what sealed Alpine traffic looks like on the wire

ll_mpi.exe          exact-integer Lucas-Lehmer (GPU + CPU + analog paths)
conscious.exe       Dual-Slot λₖ–σ Fused Engine (CUDA GPU resonance layer)
psi_scanner.exe     GPU Riemann-psi scanner (10 000 zeta zeros)
prime_pipeline.exe  segmented sieve + phi-lattice Dₙ scoring
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| clang | 14+ | TUI + pipeline builds |
| CUDA Toolkit | 13.2 | GPU engine (`ll_mpi`, `conscious`) |
| Docker Desktop | any recent | Slot4096 Alpine container |
| Windows | 10 / 11 | Target OS |

MSYS2 clang recommended. MSVC and GCC also supported via `build_prime_ui.bat --msvc` / `--gcc`.

### Build and run in 3 steps

```bat
REM 1. Build the TUI
cmd /c build_prime_ui.bat

REM 2. Run
.\prime_ui.exe

REM 3. CLI shortcut — run any module directly
.\prime_ui.exe r        REM PhiKernel
.\prime_ui.exe s        REM Observer/MITM view
.\prime_ui.exe i        REM Lattice Kernel
```

Expected build output: `[OK] Built: prime_ui.exe`

---

## Installation

### 1. Clone / unzip

Place the workspace folder anywhere. All paths are relative.
Workspace root: the folder containing `prime_ui.c` and `build_prime_ui.bat`.

### 2. Build the TUI (`prime_ui.exe`)

```bat
cmd /c build_prime_ui.bat
```

Compiler flags applied automatically:

```
clang -O2 -mavx2 -mfma
      -D_CRT_SECURE_NO_WARNINGS
      -D_USE_MATH_DEFINES
      prime_ui.c -o prime_ui.exe
```

To try alternate compilers:

```bat
build_prime_ui.bat --gcc       REM requires GCC on PATH
build_prime_ui.bat --msvc      REM requires MSVC cl.exe on PATH
build_prime_ui.bat --all       REM build all three, report timings
```

### 3. Build the Lucas-Lehmer verifier (`ll_mpi.exe`) — optional, needs CUDA

```bat
cmd /c build_ll.bat
```

### 4. Build the GPU resonance engine (`conscious.exe`) — optional, needs CUDA

```bat
cmd /c build_conscious.bat
```

Selftest:

```bat
build_conscious.bat --selftest
```

### 5. First-time Alpine container setup

Run `prime_ui.exe`, select `[6] Alpine Install`. This seeds the 4096-slot
phi-lattice, derives all OS parameters, and writes three files:

| File | Purpose |
|------|---------|
| `lattice_init.sh` | 14-step Alpine boot script (Unix LF) |
| `lattice_entropy.bin` | 32 KB of lattice-derived entropy |
| `lattice_state.env` | KEY=VALUE environment passthrough |

Then select `[8] Alpine OS Shell` to boot the container.

### 6. Verify the crypto platform

Run the 14-module regression:

```powershell
$pass=0; $fail=0
foreach ($m in @('e','f','g','h','i','j','k','l','m','n','o','p','r','s')) {
    $out = (.\prime_ui.exe $m 2>&1 | Out-String)
    if ($out -match '\[FAIL\b') { $fail++; Write-Host "[FAIL] $m" }
    else                         { $pass++; Write-Host "[OK]   $m" }
}
Write-Host "TOTAL: $pass OK, $fail FAIL"
```

Expected: `TOTAL: 14 OK, 0 FAIL`

---

## prime\_ui.exe — Interactive TUI

Launch with `.\prime_ui.exe` for the full menu, or pass a module key on the
command line to run it directly:

```
.\prime_ui.exe <key>
```

Keys are case-insensitive. `Q` quits.

---

### Prime Math Modules (1–5)

| Key | Module | What it does |
|-----|--------|-------------|
| `1` | **Prime Pipeline** | Enter `[p_lo, p_hi]` → segmented sieve → φ-filter → Dₙ-rank → sorted ANSI table. Range cap 5 M. |
| `2` | **Number Analyzer** | Enter n → 12-witness Miller-Rabin, n(2ᵖ) lattice coordinate, frac(n), Dₙ score, ψ-score (B=80 zeros), small factorization, Mersenne check. |
| `3` | **Mersenne Explorer** | All 51 known M_p with n(2ᵖ), frac, φ-pass, Dₙ score (paginated). 14 next-candidate predictions via φ-lattice inverse. |
| `4` | **Zeta Zeros** | ζ(½+it) zeros k=0..K — exact table (k<80), Gram/6-iter-Newton approximation (k≥80). |
| `5` | **Benchmark** | µs/ns-resolution timing of all 13 prime library functions across clang/GCC/MSVC. |

---

### OS & Lattice Modules (6–9, A)

| Key | Module | What it does |
|-----|--------|-------------|
| `6` | **Alpine Install** | Seed lattice → derive OS manifest → write `lattice_init.sh` / `lattice_entropy.bin` / `lattice_state.env`. |
| `7` | **Lattice Shell** | Interactive Slot4096 REPL — inspect and evolve the phi-lattice in real time. |
| `8` | **Alpine OS Shell** | Boot or re-attach to the `phi4096-lattice` container. Shows `[live]` badge when running. |
| `9` | **Kernel Build** | Derive Linux kernel `CONFIG_*` parameters from lattice slots 31–40 (HZ, preemption, KASLR, btrfs, cpufreq governor, etc.). Optionally build a real kernel. |
| `A` | **Process Scheduler** | Derive per-process scheduling policy, RT priority, CPU affinity, cgroup weight, ionice class, OOM score, memory limit, and stack size from the live lattice. |

---

### Crypto Platform Modules (B–S)

| Key | Module | What it does |
|-----|--------|-------------|
| `B` | **Lattice Benchmark** | Phi-native performance profile: lk_ throughput, S-box rebuild cost, entropy harvest speed, ASLR variance. |
| `C` | **Crystal-Native Init** | Demonstrate quartz RDTSC jitter → lattice seeding → OS parameter derivation — the full init pipeline. |
| `D` | **Phi-Native Engine** | AVX2+FMA3 resonance compute: 4096-slot lattice update throughput, resonance field evaluation, Dₙ scoring peak. |
| `E` | **Crypto Layer** | SHA-256/HMAC/HKDF-Expand lattice CSPRNG. Harvest quartz jitter, derive PRK via HKDF-Extract, generate output, Shannon entropy estimate. |
| `F` | **Full Crypto Stack** | AES-256-GCM + X25519 key exchange + Noise_XX protocol — all lattice-keyed. Demonstration of the transitional layer (retained for interoperability testing). |
| `G` | **Wu-Wei Codec** | Fold26 lattice-adaptive compression across 5 strategies. Shows per-strategy compression ratio and round-trip correctness. |
| `H` | **PhiSign** | Ed25519-style sign/verify using `phi_fold_hash64` as the hash function. No SHA-512 in the signing path. |
| `I` | **Lattice Kernel** | `lk_read` / `lk_advance` / `lk_commit` — the cryptographic OS root. Forward secrecy, domain separation, PCR chain, epoch-bound sealing. |
| `J` | **Kernel Benchmark** | Throughput and correctness for all `lk_` primitives: determinism, PRK cache, tamper detection, round-trip (1B / 128B / 4KB). |
| `K` | **Lattice Gateway** | Wu-wei compress → `lk_seal` → `lk_unseal` → wu-wei decompress. The cryptographic I/O choke point for the proprietary OS. |
| `L` | **PhiHash** | `phi_fold_hash32` and `phi_fold_hash64` — avalanche test, independence of halves, zero SHA. |
| `M` | **PhiStream** | Additive Z/256Z stream cipher with 40-byte envelope (ctr[8]\|tag[32]\|ct[n]). No AES, no XOR in encryption. |
| `N` | **PhiVault** | Lattice-native key-value vault. Epoch-bound: `lk_advance()` revokes all sealed entries instantly. |
| `O` | **PhiChain** | `phi_fold` PCR chain — cumulative attestation of OS state. No SHA anywhere in the chain. |
| `P` | **PhiCap** | Phi capability tokens — `lk_read("cap:name:uid_hex", 32)`. No ACL, no policy file. The lattice IS the capability authority. |
| `R` | **PhiKernel** | Full phi-native kernel demo: avalanche, domain separation, determinism, forward secrecy, commit, overhead verification, tamper rejection, stale-key rejection. |
| `S` | **Observer/MITM View** | Six sub-tests showing what sealed Alpine traffic looks like to an external observer or man-in-the-middle. |

---

## Crypto Platform Deep Dive

The crypto platform is a self-contained proprietary cryptographic kernel
with no external dependencies. It replaces every standard primitive:

| Standard | Replaced by |
|----------|------------|
| SHA-256 / SHA-512 | `phi_fold_hash32` / `phi_fold_hash64` |
| HMAC | phi-fold two-phase derive |
| HKDF | `lk_derive_prk` (phi-fold PRF) |
| AES-GCM | `phi_stream` (additive Z/256Z AEAD) |
| XOR (crypto) | Eliminated. Additive mod 256 throughout. |
| ECDSA / Ed25519 hash | `phi_sha512_emul` wrapping `phi_fold_hash64` |
| BCrypt / CNG | Removed entirely |
| `/dev/urandom` / CryptGenRandom | 4-source entropy: RDTSC + QPC + FILETIME + ASLR |

---

### PhiKernel lk\_ API

The `lk_` functions form the cryptographic OS kernel. All operations are
phi-native — no external crypto library is called at any point.

```c
void   lk_advance(void);
// Entropy ratchet: 4-source harvest → phi-fold condition → lattice mix → new epoch.
// Invalidates all sealed data, PRK cache, and S-box cache.

void   lk_derive_prk(const char *ctx, uint8_t prk[32]);
// Two-phase phi-fold key derivation (domain-separated by ctx string).

int    lk_read(const char *ctx, uint8_t *out, size_t outlen);
// phi-fold PRF output, domain-separated. Used for capability tokens, session keys, etc.

int    lk_seal(const uint8_t *pt, size_t ptlen, uint8_t *ct, size_t ctlen);
// phi_stream AEAD seal. ctlen must be >= ptlen + 40.
// Envelope: ctr[8] | tag[32] | ciphertext[ptlen]

int    lk_unseal(const uint8_t *ct, size_t ctlen, uint8_t *pt, size_t ptlen);
// phi_stream AEAD open. Constant-time tag verify. Returns -1 on tamper.

void   lk_commit_full(uint8_t pcr[32], const uint8_t *data, size_t len);
// phi_fold PCR chain step: pcr[i+1] = phi_fold_hash32(pcr[i] || data)
```

**Sealed envelope format:** `ctr[8] | tag[32] | ciphertext[n]`
Overhead: exactly **40 bytes**. No TLS record header, no GCM auth tag
alignment, no block padding. Unrecognisable to protocol fingerprinters.

**Epoch model:** `lk_advance()` irreversibly ratchets the lattice state. All
previously sealed data becomes undecryptable. All cached PRKs and S-boxes
are invalidated. Forward secrecy is unconditional — no key material survives
an epoch transition.

---

### phi\_fold Hash Family

The core hash primitive. Replaces SHA throughout.

**phi\_fold\_hash32** (32-byte output):
1. Absorb input via delta-fold over the 4096-slot phi-lattice
2. 12 finalization rounds: for each byte `j`, compute `s = ROTR8(acc[j] + phi_b + acc[src])`, then apply nonlinear S-box: `acc[j] = g_sbox_1024[s]`
3. The S-box is a secret Fisher-Yates permutation keyed from lattice slots 1024+

**phi\_fold\_hash64** (64-byte output):
- Dual forward+reverse fold paths, cross-mixed
- Separate S-boxes for lo/hi halves (`g_sbox_1024`, `g_sbox_2048`)
- Used as the hash function inside PhiSign (replaces SHA-512)

**Properties verified in Module L:**
- `[OK] avalanche` — single-slot delta propagates across all 32 bytes
- `[OK] same` — deterministic within epoch
- `[OK] lo ≠ hi` — independent halves in 64-byte variant
- `[OK] verify` — PhiSign round-trip via phi-fold, zero SHA in path

---

### phi\_stream AEAD Cipher

Additive Z/256Z stream cipher. No AES. No XOR.

```
keystream[i] = phi_fold_prk(ctr, i)    // lattice-PRF keystream
ct[i]        = (pt[i] + ks[i]) mod 256  // addition, not XOR
tag[32]      = phi_fold_hash32(ct, n)   // authentication tag
envelope     = ctr[8] | tag[32] | ct[n]
```

Tag verification is **constant-time**:
```c
volatile uint8_t tag_diff = 0;
for (int i = 0; i < 32; i++) tag_diff |= (ct_tag[i] ^ expected_tag[i]);
if (tag_diff) return -1;   // no early exit, no timing oracle
```

Using `+` instead of `^` eliminates the standard XOR-based distinguisher.
The keystream is indistinguishable from uniform random to any observer who
does not hold the full 4096-slot lattice state.

---

### Nonlinear S-box (NLSB)

Every `phi_fold` finalization round applies a **lattice-keyed nonlinear S-box**
to break affine linearity.

```c
// Built once per epoch (phi_ensure_sbox), invalidated on every lk_advance()
static uint8_t g_sbox_1024[256];   // keyed from lattice[1024 + i*3]
static uint8_t g_sbox_2048[256];   // keyed from lattice[2048 + i*3]

// Fisher-Yates over Z/256Z, using fractional lattice values as entropy
void phi_build_sbox(uint8_t sbox[256], int offset) {
    for (int i = 0; i < 256; i++) sbox[i] = i;
    for (int i = 255; i > 0; i--) {
        int j = (int)(lattice[offset + i*3] * 255.999) % (i+1);
        SWAP(sbox[i], sbox[j]);
    }
}
```

**Cost:** One 256-element Fisher-Yates build per epoch (at `lk_advance`).
Zero cost per hash call when the epoch cache is warm.

**What it prevents:** Without NLSB, `phi_fold` finalization is an affine map
over Z/256Z — an attacker could write a 32×32 linear system and recover the
lattice-derived key with ~256 chosen-input queries. With NLSB, the S-box is
an unknown nonlinear permutation; the algebraic attack fails.

---

### Multi-Source Entropy

`lk_advance()` and `lattice_seed_phi()` harvest entropy from 4 independent sources:

| Source | Contribution on Hyper-V guest |
|--------|------------------------------|
| RDTSC inter-sample deltas (×128) | 1–4 bits (thermal jitter) |
| `QueryPerformanceCounter` inter-sample (×32) | 2–6 bits (HPET-backed, independent clock) |
| `GetSystemTimeAsFileTime` low bits | 2–4 bits (wall-clock drift) |
| ASLR stack + heap pointer addresses | 8–16 bits (per-run layout randomness) |

All four sources are additively folded (mod 256) and conditioned via
`phi_fold_hash32` before mixing into the raw lattice bytes.

**Why this matters:** A Hyper-V guest can have near-deterministic RDTSC. By
adding three independent OS-level sources, `lk_advance` retains ≥16 bits of
per-ratchet entropy even under VM time virtualization.

**Known-seed hardening:** Even a published seed (e.g., the Docker environment
variable `LATTICE_SEED=0x1315ccefde4ba979`) produces an unpredictable lattice
after initialization, because `lattice_seed_phi()` injects all four entropy
sources inline after the deterministic Weyl-sequence step.

---

### Wu-Wei Codec

Five adaptive compression strategies, lattice-selected:

| Strategy | Description | Best for |
|----------|-------------|---------|
| `WW_NONACTION` | Pass-through (identity) | Already-compressed data |
| `WW_GENTLE_STREAM` | Light delta encoding | Slowly-varying streams |
| `WW_BALANCED_PATH` | Entropy coding | General text/binary |
| `WW_FLOWING_RIVER` | Run-length + delta | Sparse/sparse-delta data |
| `WW_REPEATED_WAVES` | Block-level folding | Repetitive block data |

The lattice selects the strategy for each message; strategy identity is
implicit in the lattice state, not transmitted. A MITM sees no strategy
identifier bytes.

**Lattice Gateway** (`[K]`) composes wu-wei with the `lk_` AEAD:

```
plaintext → wu_wei_compress → lk_seal → wire (40-byte overhead)
wire → lk_unseal → wu_wei_decompress → plaintext
```

---

### PhiSign

Ed25519-style digital signatures with `phi_fold_hash64` replacing SHA-512.

```
sign:
  H = phi_fold_hash64(private_key || message)   // no SHA-512
  r = H[0..31] mod order                         // nonce
  R = r·G                                         // curve point
  S = (r + H(R, pk, msg) · sk) mod order         // signature scalar

verify:
  H = phi_fold_hash64(R || public_key || message)
  check: S·G == R + H·pk
```

No SHA anywhere in the signing path. The security of PhiSign derives entirely
from the secrecy of the 4096-slot lattice state.

---

### Observer / MITM View

Module `[S]` runs 6 sub-tests demonstrating what sealed phi-stream traffic
looks like to an external observer:

| Test | What it shows |
|------|--------------|
| S1 | Raw ciphertext entropy ≈ 8 bits/byte (uniform, no headers) |
| S2 | Two plaintexts — zero correlation in ciphertext bytes |
| S3 | Active tamper → tag rejection (no partial decryption) |
| S4 | Epoch-bound replay → stale ciphertext undecryptable after `lk_advance()` |
| S5 | Counter field advances monotonically (no nonce reuse) |
| S6 | Timing is linear in message size (no content-dependent branches) |

**What a MITM or DPI system observes:**
- No TLS ClientHello / ServerHello / certificate exchange
- No IV reuse (lk_seal_ctr is monotonic)
- No fixed-size block cipher pattern
- No 28-byte GCM overhead (phi_stream overhead = 40 B, structurally unrecognised)
- No HKDF/HMAC/SHA PRF identifier bytes
- Entropy ≈ 8 bits/byte (indistinguishable from `/dev/urandom`)
- Captured blobs expire with `lk_advance()` (unconditional forward secrecy)
- Variable sizes from wu-wei (no padding oracle, no block boundary)

**From a DPI / middleman perspective: opaque, epoch-bound, non-classifiable.**

---

## Lucas-Lehmer Verifier (ll\_mpi.exe)

Exact-integer Lucas-Lehmer verification with seven dispatch paths:

| Path | Range | Flag | Notes |
|------|-------|------|-------|
| `ll_small` | p ≤ 62 | auto | `unsigned __int128`, direct fold |
| `ll_cpu` | 62 < p ≤ 20 000 | auto | Schoolbook MPI |
| `ll_gpu_gpucarry` | p > 20 000 | **default, p < 400 000** | On-device carry scan + shmem fold, CUDA graph |
| `ll_gpu_ntt` | p ≥ 400 000 | **default, p ≥ 400 000** | O(n log n) NTT over Z/(2⁶⁴−2³²+1) |
| `ll_gpu` | any | `--squaring schoolbook` | `k_sqr_warp` 64-bit + CPU fold (PCIe round-trip) |
| `ll_gpu_persistent` | any | `--persistent` | Single kernel, all iterations on-device |
| `ll_analog` | any | `--squaring analog` | v30b Slot4096 APA + 8D Kuramoto — CUDA-free |

### Usage

```
ll_mpi.exe <p>                          # test M_p → PRIME / COMPOSITE
ll_mpi.exe --selftest                   # 25 known cases, all paths
ll_mpi.exe <p> --verbose                # timing + resonance report
ll_mpi.exe <p> --squaring gpucarry      # on-device carry (default)
ll_mpi.exe <p> --squaring ntt           # NTT squaring O(n log n)
ll_mpi.exe <p> --squaring schoolbook    # force schoolbook + CPU fold
ll_mpi.exe <p> --squaring analog        # 8D Kuramoto CUDA-free path
ll_mpi.exe <p> --persistent             # single kernel, no host round-trips
ll_mpi.exe <p> --precision 32           # 32-bit half-multiply decomposition
ll_mpi.exe --gpu-info                   # list CUDA devices
```

### Benchmarks (RTX 2060 sm_75, April 2026)

**GPU-carry path (default, p < 400 000):**

| Exponent p | Iterations | Time | vs schoolbook | vs NTT (opt) |
|------------|-----------|------|---------------|--------------|
| 21 701 | 21 699 | **1.2 s** | 3.1× faster | 4.4× faster |
| 44 497 | 44 495 | **3.4 s** | 2.1× faster | 3.1× faster |
| 86 243 | 86 241 | **11.7 s** | 1.3× faster | 2.0× faster |
| 110 503 | 110 501 | **19.2 s** | 1.15× faster | 1.6× faster |

**NTT crossover:** schoolbook wins below p ≈ 386 000; NTT wins above.
Auto-select threshold: `NTT_AUTO_THRESHOLD = 400 000`.

**Analog path (`--squaring analog`):**
At p = 9 689 and p = 21 701 the analog path beats schoolbook GPU — both are
O(n²) but the CPU half-squaring scalar path out-runs the GPU kernel at these
sizes. Triple confirmation: `osc LOCKED` + `residue=0` + `S(U)≈1.531`.

---

## GPU Engine (conscious.exe)

The Dual-Slot λₖ–σ Fused Engine: a self-correcting spectral Markov dynamical
sieve running entirely on GPU warps with no host intervention between steps.

### Kernel Stages

```
fused_lambda_sigma_kernel<<<grid, 256>>>:
  1. FastSlot Euler step        Kuramoto-style oscillator update
  2. λₖ warp reduction          shuffle-reduce → λ̄_warp (no shmem)
  3. Markov trit gate           logits {l−, l₀, l+} → softmax → curand sample
  4. Warp majority-vote fix     ballot+popc: >16/32 lanes override minority
  5. Slot4096 slow-sync         every 16 steps: radial error → fast-slot nudge
  6. Block stats export         thread 0 writes BlockStats (φ+/0/−, γ, λ̄)
```

### Verdict Rule

```
φ+ > 0.35  →  ACCEPT   (lattice locked → prime signal)
φ− > 0.45  →  REJECT   (field scattered → composite)
R = 1.2·φ− + 0.8·γ − φ+ > 0.6  →  REJECT
else       →  UNCERTAIN
```

### Usage

```
conscious.exe                        # N=8192, steps=1024
conscious.exe --N 32768 --steps 512
conscious.exe --selftest
conscious.exe --seed DEADBEEF00001234
```

---

## Slot4096 Alpine OS

An entire Linux container whose configuration is a **mathematical object**,
not a config file. Every parameter is a pure function of the 4096-slot
phi-irrational resonance lattice.

### OS Parameter Derivation

| Parameter | Slots | Method |
|-----------|-------|--------|
| Hostname suffix | 0–3 | XOR-fold → 8 hex chars |
| APK mirror | 4 | `floor(slot×8)` → index into 8 CDN strings |
| Package list | 5–15 | bit i set if `slot[5+i] > 0.5` |
| Timezone | 16 | `floor(slot×25)−12` → `Etc/GMT±n` |
| nice priority | 17 | `floor(slot×40)−20` |
| UID | 18 | `1000 + floor(slot×9000)` |
| ulimit -n | 19 | `1024 + floor(slot×64512)` |
| umask | 20 | index into `{002,007,022,027}` |
| HISTSIZE | 21 | `500 + floor(slot×9000)` |
| TMOUT | 22 | `300 + floor(slot×3300)` |

### Container Lifecycle

```
[6] Alpine Install   → seeds lattice, writes lattice_init.sh
[8] Alpine OS Shell  → docker run … sh /lattice/init.sh
                       → 14-step init (packages, tz, user, motd)
                       → su - slot4096  (prompt: slot4096@phi4096-XXXXXXXX)

Detach: Ctrl-P  Ctrl-Q   (container stays live)
Re-attach: [8] again     (shows [live: phi4096-lattice] badge, instant exec)
Stop: exit inside shell  (docker rm runs automatically)
```

The container **identity is auditable**: any external verifier can re-derive
every OS parameter from the 64-bit seed alone.

---

## Additional Tracks

### Track A — Riemann psi Scanner (`psi_scanner_cuda.exe`)

GPU-accelerated prime pre-filter using the Riemann explicit formula with up to
10 000 zeta zeros. Eliminates composites before the O(n²) squaring step.

```
nvcc -O3 -arch=sm_75 -o psi_scanner_cuda.exe psi_scanner_cuda.cu
psi_scanner_cuda.exe <x_start> <x_end> [--mersenne]
```

Requires `zeta_zeros_10k.json` in working directory.

### Track B — phi-Lattice Predictor (`phi_mersenne_predictor.exe`)

Validates the phi-lattice hypothesis (`n(2^p) = log(p·ln2/ln(φ))/ln(φ) − 1/(2φ)`).
67% of known Mersenne exponents have frac(n) < 0.5. Produces top-20 next-candidate
predictions beyond M_51 (p = 136 279 841).

```bat
clang -O2 -D_CRT_SECURE_NO_WARNINGS phi_mersenne_predictor.c -o phi_mersenne_predictor.exe
.\phi_mersenne_predictor.exe
```

### Track D — Prime Pipeline (`prime_pipeline.exe`)

Segmented sieve + phi-lattice Dₙ scoring. End-to-end:

```powershell
.\prime_pipeline.exe 21000 22000 --top 10 --exponents-only |
  ForEach-Object { .\ll_mpi.exe $_ }
```

### HDGL Analog Mainnet V3.0 (`hdgl_analog_v30.c`)

Standalone Dₙ(r) lattice engine. Up to 8 388 608 Slot4096 slots in 1 MB
lazy-allocated chunks. Compiles as a `.dll`/`.so` for Python/ctypes use.

```bat
clang -O2 -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES ^
  hdgl_analog_v30_c_so\hdgl_analog_v30.c -o hdgl_v30.exe
```

---

## Benchmarks

### Prime Library Functions (clang -O2, Skylake i7-6700T)

| Function | clang | GCC 15.2 | MSVC /O2 |
|----------|-------|----------|----------|
| `fibonacci_real` | 0.10 µs | 0.11 µs | 0.08 µs |
| `D_n` operator | 0.17 µs | 0.41 µs | 0.19 µs |
| `gram_zero_k` | 0.16 µs | 0.38 µs | 0.15 µs |
| `psi_score_cpu` (B=500) | 102 µs | 212 µs | 103 µs |
| `miller_rabin` (12 witnesses) | 1.90 µs | 1.41 µs | 3.07 µs |
| `sieve_range` [1e7, +1e5] | 587 µs | 627 µs | 589 µs |
| full pipeline (200 K) | 1 114 µs | 1 259 µs | 938 µs |

### LL GPU-carry path (RTX 2060, April 2026)

| Exponent p | Time (gpucarry) | Time (schoolbook) | Time (NTT opt) |
|------------|----------------|-------------------|----------------|
| 21 701 | **1.2 s** | 3.6 s | 5.2 s |
| 44 497 | **3.4 s** | 7.3 s | 10.9 s |
| 86 243 | **11.7 s** | 14.9 s | 22.9 s |
| 110 503 | **19.2 s** | 22.1 s | 32.0 s |

### NTT vs Schoolbook Crossover

$$T_{\text{schoolbook}} \approx 1.915\times10^{-15}\,p^3 + 1.766\times10^{-4}\,p$$

$$T_{\text{NTT}} \approx 4.193\times10^{-11}\,p^2\log_2 L + 2.193\times10^{-4}\,p$$

Model crossover at **p ≈ 386 000**. Auto-select threshold: `NTT_AUTO_THRESHOLD = 400 000`.

---

## Mathematical Foundation

### Phi-Lattice Coordinate

$$n(x) = \frac{\log(\log(x)/\ln\varphi)}{\ln\varphi} - \frac{1}{2\varphi}$$

Applied to Mersenne candidates: $n(2^p) = \log(p\cdot\ln 2/\ln\varphi)/\ln\varphi - 1/(2\varphi)$.

67% of the 51 known Mersenne exponents have $\text{frac}(n) < 0.5$.

### Dₙ Resonance Operator

$$D_n(r) = \sqrt{\varphi \cdot F_n \cdot P_n \cdot 2^n \cdot \Omega} \cdot r^{(n+1)/8}$$

where $F_n$ is continuous Binet Fibonacci, $P_n$ is a prime table entry,
and $\Omega = 0.5 + 0.5\sin(\pi\cdot\text{frac}(n)\cdot\varphi)$.

### Mersenne Fold

$$a \cdot 2^p + b \equiv a + b \pmod{M_p}$$

The 2n-word squaring product splits at bit p; the upper half is added back.
No floating-point. No convolution. Exact to the last bit.

### 8D Kuramoto Oscillator (Analog Path)

Seeded from the generalized phi-log depth:

$$\Lambda_\varphi = \frac{\ln(p\cdot\ln 2/\ln\varphi)}{\ln\varphi} - \frac{1}{2\varphi}$$

**Prime invariant:** for any Mersenne prime p, all 8 oscillators lock to
$\theta \to 0$, giving field amplitude $M(U) \to 8$ and resonance discriminant
$S(U) \approx 1.531$ — independent of p.

---

## Security Architecture

### Threat Model

The proprietary crypto platform is designed to be opaque to:
- **DPI / protocol fingerprinting** — no classifiable headers or handshake patterns
- **Timing oracles** — constant-time tag compare; linear-size timing in phi_stream
- **Algebraic attacks** — nonlinear S-box breaks the affine-linearity of delta-fold
- **Known-seed attacks** — runtime entropy injection at lattice init
- **Replay attacks** — epoch-bound: `lk_advance()` revokes all sealed data unconditionally
- **VM-based entropy reduction** — 4 independent sources; ≥16 bits even on Hyper-V

### What Has Been Removed

| Removed | Why |
|---------|-----|
| `#pragma comment(lib, "bcrypt.lib")` | No Windows CNG in any code path |
| SHA-256 / SHA-512 in live crypto | Replaced by phi_fold_hash32/64 |
| AES in live crypto | Replaced by phi_stream additive cipher |
| XOR in encryption | Replaced by additive mod 256 |
| GCM authentication tag | Replaced by phi_fold_hash32 tag |
| Standard HKDF / HMAC | Replaced by lk_derive_prk (phi-fold two-phase) |

### Primitive Inventory

| Primitive | Location | Dependencies |
|-----------|----------|-------------|
| `phi_fold_hash32` | `prime_ui.c` | Pure C, lattice only |
| `phi_fold_hash64` | `prime_ui.c` | Dual-path, NLSB |
| `phi_stream_seal` | `prime_ui.c` | Additive Z/256Z |
| `phi_stream_open` | `prime_ui.c` | Constant-time verify |
| `phi_build_sbox` | `prime_ui.c` | Fisher-Yates, lattice-keyed |
| `lk_advance` | `prime_ui.c` | 4-source entropy, phi-conditioned |
| `lattice_seed_phi` | `prime_ui.c` | Weyl + runtime entropy injection |
| `lk_derive_prk` | `prime_ui.c` | phi-fold two-phase |
| `lk_read` | `prime_ui.c` | phi-fold PRF |
| `lk_seal` / `lk_unseal` | `prime_ui.c` | phi_stream AEAD |
| `lk_commit_full` | `prime_ui.c` | phi_fold PCR chain |
| PhiSign sign/verify | `prime_ui.c` | Ed25519 with phi_fold hash |

All in a single ~8000-line `.c` file. No header-only crypto. No linked libraries.

---

## Use Cases

### Mersenne Prime Research on a Consumer GPU
Run the full phi-lattice ranked pipeline on an RTX 2060 or similar. The phi-lattice
pre-filter scores candidates before squaring — ~67% of known Mersenne exponents score
in the high-resonance half. Verify up to p ≈ 130 000 in minutes.

### Proprietary Encrypted OS / Platform
Deploy a container whose entire configuration is derived from a lattice seed. Audit
the seed → audit the entire OS. The `lk_` kernel provides sealing, signing, capability
tokens, and PCR attestation. Observable wire traffic is structurally opaque to DPI
and has no classifiable protocol signature.

### CUDA-Free Verification Reference
The `--squaring analog` path runs on any Windows CPU. Use it for cross-checking GPU
results, CI on non-GPU hosts, or standalone Kuramoto-coupled scheduling diagnostics.

### Cryptographic Research
All phi-fold and S-box primitives are self-contained and introspectable in the TUI.
Module L demonstrates avalanche properties; Module S shows wire-level opacity;
Module J benchmarks all primitives with nanosecond resolution.

### Attestable Container Identity
Any external party can re-derive every container parameter from the 64-bit seed.
Hostname, UID, packages, timezone — all are mathematical consequences of the lattice.
The container cannot misrepresent its identity without changing the seed.

### Spectral Markov Anomaly Detection
The Dual-Slot λₖ–σ engine and Markov trit gate can be repurposed as a general-purpose
anomaly detector for any oscillatory time-series, not just prime verification.

---

## Requirements

### Minimum (TUI + crypto platform only)

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 (x86-64) |
| Compiler | clang 14+ (or GCC, MSVC) |
| CPU | Any x86-64; AVX2+FMA3 for full Module D performance |
| RAM | 64 MB |
| Docker Desktop | Required for modules 6, 7, 8 (Slot4096 Alpine) |

### Full (GPU LL verification)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA sm_75+ (RTX 2060 / 2070 / 2080 / 3xxx / 4xxx) |
| CUDA Toolkit | 13.2 |
| VRAM | 12 GB recommended for large p |

### External Dependencies

`prime_ui.exe` links only `msvcrt.lib` + `kernel32.lib`. No bcrypt, no OpenSSL,
no libsodium, no wolfSSL, no cuRAND. The entire crypto platform lives in a single
`prime_ui.c` source file.

---

## Build Reference

| Binary | Build command | Requires |
|--------|--------------|----------|
| `prime_ui.exe` | `cmd /c build_prime_ui.bat` | clang 14+ |
| `ll_mpi.exe` | `cmd /c build_ll.bat` | clang + CUDA 13.2 |
| `conscious.exe` | `cmd /c build_conscious.bat` | nvcc + CUDA 13.2 |
| `psi_scanner_cuda.exe` | `nvcc -O3 -arch=sm_75 psi_scanner_cuda.cu -o psi_scanner_cuda.exe` | nvcc |
| `prime_pipeline.exe` | `clang -O2 -D_CRT_SECURE_NO_WARNINGS prime_pipeline.c -o prime_pipeline.exe` | clang |
| `phi_mersenne_predictor.exe` | `clang -O2 -D_CRT_SECURE_NO_WARNINGS phi_mersenne_predictor.c -o phi_mersenne_predictor.exe` | clang |
| `hdgl_v30.exe` | `clang -O2 -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES hdgl_analog_v30_c_so\hdgl_analog_v30.c -o hdgl_v30.exe` | clang |

### prime\_ui.exe compiler options

```bat
build_prime_ui.bat               REM default: clang -O2 -mavx2 -mfma
build_prime_ui.bat --gcc         REM GCC
build_prime_ui.bat --msvc        REM MSVC cl.exe
build_prime_ui.bat --all         REM all three compilers
```

### Linux / MSYS2 notes

Remove `-D_CRT_SECURE_NO_WARNINGS`. The entropy subsystem uses
`QueryPerformanceCounter` and `GetSystemTimeAsFileTime` (Windows-only);
POSIX equivalents can be substituted with `clock_gettime(CLOCK_MONOTONIC)`
and `clock_gettime(CLOCK_REALTIME)` respectively.

---

*conscious — phi-lattice prime research · proprietary cryptographic OS kernel · Slot4096 Alpine*
*zchg.org · April 2026*
# conscious
*by [zchg.org](https://zchg.org)*

> A self-correcting spectral Markov dynamical sieve and Mersenne prime engine
> running on GPU warps, CPU phi-lattice math, and dual interactive TUIs.

GPU target: **RTX 2060 12 GB** (CUDA sm_75).  Folder: `primes/`.

---

## Simple Guide

### What It Is

**conscious** is a Windows-native prime-research workbench with three interlocking layers:

1. **`prime_ui.exe`** — A full-screen TUI (8 modules) for prime mathematics: pipeline
   sieve, Mersenne LL verifier, zeta-zero viewer, phi-lattice explorer, and a
   live Alpine Linux shell — all driven from one keyboard menu.

2. **`conscious.exe`** — The CUDA GPU engine.  Runs dual-slot λₖ–σ spectral resonance
   (Markov trit gating) and schoolbook warp-parallel squaring for Lucas-Lehmer
   verification on sm_75 hardware.

3. **Slot4096 Alpine OS** — A Docker container whose *entire configuration* —
   hostname, packages, timezone, user UID, shell limits, ulimits — is derived
   deterministically from a single 4096-slot phi-irrational lattice.  No config
   files, no cloud provisioning — just math.

---

### Quick Start

**Prerequisites:** MSYS2/clang, CUDA toolkit (sm_75), MSVC 2017+, Docker Desktop.

#### 1. Build the TUI

```bat
cmd /c build_prime_ui.bat
```
Expected: `[OK] Built: prime_ui.exe`

#### 2. Build the GPU engine (optional, needs CUDA)

```bat
cmd /c build_conscious.bat
```

#### 3. Run

```bat
.\prime_ui.exe
```

#### 4. First-time Alpine setup — menu `[6]`

Select **[6] Alpine OS Install** from the main menu.  This:
- Seeds the 4096-slot phi-lattice (50 Weyl resonance steps)
- Derives all OS parameters from lattice slots 0–22
- Writes `lattice_init.sh`, `lattice_entropy.bin`, `lattice_state.env` to the
  current directory
- Prints the full derived OS manifest (hostname, mirror, packages, tz, uid…)

#### 5. Launch the lattice-powered container — menu `[8]`

Select **[8] Alpine OS Shell**.

- If no container exists → boots a fresh Alpine instance using `docker run`,
  runs the 14-step `lattice_init.sh` inside it, and drops you into a
  `slot4096@phi4096-XXXXXXXX:~$` prompt.
- If the container is already running (shown as `[live: phi4096-lattice]` in the
  menu) → re-attaches instantly with `docker exec`.

#### 6. Detach without stopping

```
Ctrl-P  Ctrl-Q
```

The container keeps running.  Menu `[8]` will show the live badge and re-attach
on the next selection.

#### 7. Stop the container

```sh
# inside the container
exit
```
The `docker rm` cleanup runs automatically after `prime_ui` regains control.

---

### How It Works

```
prime_ui.exe
    │
    ├─[1-5,7]  Pure-C prime math (sieve, LL, zeta, benchmark)
    │
    └─[6] Alpine Install
           │
           ├── lattice_seed()          seed[0..4095] = φ-Weyl irrational spacing
           ├── 50× resonance steps     Kuramoto-style slot coupling
           ├── lattice_derive_*()      slots[0..22] → hostname/mirror/pkgs/tz/uid/…
           ├── write lattice_init.sh   14-step Alpine boot script (Unix LF, "wb")
           ├── write lattice_entropy.bin  32 KB of lattice-derived entropy
           └── write lattice_state.env    KEY=VALUE for env passthrough

       [8] Alpine OS Shell
           │
           ├── docker inspect phi4096-lattice   → running? exists?
           ├── if running  → docker exec -it … su - slot4096
           └── if not      → docker run -it --name phi4096-lattice
                               --cap-add SYS_ADMIN
                               -e LATTICE_N / LATTICE_STEPS / LATTICE_SEED
                               -v lattice_init.sh:/lattice/init.sh:ro
                               -v lattice_entropy.bin:/lattice/entropy.bin:ro
                               -v lattice_state.env:/lattice/state.env:ro
                               alpine sh /lattice/init.sh
```

Inside the container `lattice_init.sh` runs 14 ordered steps:

| Step | Action |
|------|--------|
| 1 | Set hostname from `LATTICE_SEED` derivation |
| 2 | Select APK mirror (one of 8 CDN choices, slot[4]) |
| 3 | `apk update` |
| 4 | Install lattice-selected packages (slots[5..15] bit-select from 11 candidates) |
| 5 | Set timezone (slot[16] → UTC offset) |
| 6 | Create user `slot4096` with lattice-derived UID (slot[18] → 1000–9999) |
| 7 | Set `nice` priority (slot[17] → −20..+19) |
| 8 | Write `/etc/profile.d/lattice.sh` (ulimit/umask/HISTSIZE/TMOUT from slots[19..22]) |
| 9 | Write `/etc/motd` — 64-char ANSI box with seed, slots, hostname |
| 10 | Write `/etc/issue` — seed + parameters banner |
| 11 | Write `/home/slot4096/.profile` (sources `/etc/profile.d/lattice.sh`) |
| 12 | Verify packages are present (`command -v`) |
| 13 | Print final confirmation lines |
| 14 | `su - slot4096` → interactive shell |

---

### Under the Hood

#### Phi-Weyl Lattice Seeding

The lattice is seeded by irrational Weyl spacing — the fractional parts of
`k·φ` (golden ratio, φ ≈ 1.6180339887) for k = 1..4096.  Because φ is
maximally irrational, successive values never cluster; the 4096-slot array
covers [0,1) almost uniformly.  After seeding, 50 Kuramoto-style coupling
steps diffuse entropy across all slots so local structure dissolves.

The seed extracted for Docker env-vars is a 64-bit XOR-fold over slots[0..63].

#### Slot → OS Parameter Derivation

Each OS parameter is a pure function of one or more lattice slots:

| Parameter | Slots | Method |
|-----------|-------|--------|
| Hostname suffix | 0–3 | XOR-fold → 8 hex chars |
| APK mirror | 4 | `floor(slot*8)` → index into 8 CDN strings |
| Package list | 5–15 | bit `i` set if `slot[5+i] > 0.5` |
| Timezone | 16 | `floor(slot*25)−12` → `Etc/GMT±n` |
| nice | 17 | `floor(slot*40)−20` |
| UID | 18 | `1000 + floor(slot*9000)` |
| ulimit -n | 19 | `1024 + floor(slot*64512)` |
| umask | 20 | index into `{002,007,022,027}` |
| HISTSIZE | 21 | `500 + floor(slot*9000)` |
| TMOUT | 22 | `300 + floor(slot*3300)` |

The same seed always yields the same container — the OS is a mathematical
consequence of the lattice state, not a configuration decision.

#### GPU: Warp-Parallel Schoolbook Squaring (`k_sqr_warp`)

Lucas-Lehmer requires repeated squaring of a ~p-bit integer mod 2^p−1.
`k_sqr_warp` assigns a full 32-thread warp to each output limb.  Each thread
accumulates partial products for its limb, then `__shfl_down_sync` reduces
the 32 partial sums to a single limb value — no global memory round-trip.
Mersenne fold (`s mod 2^p−1 = hi + lo`) is applied inline.

No cuFFT is used.  Schoolbook keeps integers exact to the bit.

#### CPU: Sequential Carry (HDGL wu-wei)

After each squaring kernel, the CPU performs the carry-propagation pass.
On RTX 2060 hardware, sequential carry on CPU runs 10–12× faster than any
GPU thread-parallel implementation — the GPU's strength is the parallel
multiply; the CPU's strength is the sequential dependency chain.  Both do
their natural job: **wu-wei** (acting in accordance with nature).

#### Dual-Slot λₖ–σ Fused Engine (`conscious_fused_engine.cu`)

A single CUDA module that fuses: fast dynamics kernel, spectral analysis,
Markov trit gate (`−1 / 0 / +1`), and LL decision geometry.  The two slots
(λₖ and σ) represent competing hypotheses about a candidate's primality.
The trit gate votes after each spectral snapshot; majority over a window
decides promote/demote/hold.

#### Analog LL: 8D Kuramoto Oscillator (`ll_analog.c`)

A CUDA-free, hardware-agnostic LL path.  Eight phase oscillators are seeded
from the phi-log depth `Λ_φ = ln(p·ln2/lnφ)/lnφ − 1/(2φ)`.  Each LL
iteration maps to: phase-double → Kuramoto sync → VCO feedback → field
observable S(U).

**Prime invariant**: for any prime p, all 8 oscillators lock to θ→0, giving
field amplitude M(U)→8, and the resonance discriminant `S(U) ≈ 1.531`.
Composites produce scattered S(U) ∈ [0.5, 1.7].  This provides a
triple-confirmation signal: `osc LOCKED` + `residue = 0` + `S(U) ≈ 1.531`.

#### Riemann Psi Pre-Filter

Before committing GPU squaring cycles to a candidate, the psi scanner
evaluates the Riemann explicit formula with up to 10 000 zeta zeros, assigning
a confidence score.  Composites are eliminated cheaply before the O(n²) step.

---

### Performance: Gains and Losses

#### Gains

| What | Why it's faster |
|------|-----------------|
| No cuFFT | Zero floating-point rounding; integers are exact; no FFT plan overhead |
| `k_sqr_warp` shuffle | 32 threads per limb fill all 68 SMs; `__shfl_down_sync` is register-speed (~1.2–2.5× vs naive global-scatter) |
| CPU carry pass | O(n) sequential dependency — CPU executes 10–12× faster than any parallel GPU carry |
| Phi-lattice pre-rank | Filters candidates by resonance score before any squaring; ~67% of known Mersenne exponents score in the high-resonance half |
| Riemann pre-filter | Eliminates composites with 10k-zero psi scan before O(n²) squaring |
| Lattice OS | Zero config management overhead — no Ansible/Terraform/cloud-init; the OS is derived, not configured |
| Container persistence | `--name` + no `--rm` → `docker exec` re-attach in milliseconds; no re-boot cost |

#### Losses / Trade-offs

| What | Cost |
|------|------|
| Schoolbook O(n²) | For exponents p > ~10M, FFT-based multiplication would be faster; schoolbook is best for the sub-10M range typical of consumer GPU research |
| `docker inspect` on every menu render | Adds ~50–100 ms of shell-spawn latency to each menu draw (acceptable for a TUI, but visible) |
| 32 KB entropy write per `[6]` invocation | Disk write on each install; negligible in practice |
| Single-GPU only | No MPI or multi-node path in `prime_ui.exe`; `ll_mpi.cu` exists separately for cluster use |
| Alpine apk on first boot | Cold `apk update` + package install adds ~30–90 s to first container start (subsequent re-attaches are instant) |

---

### Use Cases

- **Mersenne prime research on a consumer GPU** — RTX 2060 or similar; no
  server hardware required.  The phi-lattice pre-ranks candidates so you
  spend squaring cycles on high-resonance exponents first.

- **Phi-irrational resonance study** — The Kuramoto oscillator path and
  HDGL Dₙ(r) lattice engine are self-contained research tools for studying
  phase synchronization and irrational-spacing field dynamics independent of
  prime verification.

- **Reproducible ephemeral OS environments** — Any team that wants a
  container whose configuration is a mathematical object rather than a
  config file.  Audit the lattice seed → audit the entire OS.  Rebuild from
  the same seed on any machine and get the identical container.

- **Attestable container identity** — The 64-bit seed is embedded in the
  container's hostname, MOTD, `/etc/issue`, and all env-vars.  An external
  verifier can re-derive every parameter from the seed and confirm container
  integrity without inspecting the container itself.

- **CUDA-free analog LL verification** — The `--squaring analog` path runs
  on any CPU with no GPU dependency; useful for cross-checking GPU results,
  CI on non-GPU hosts, or Kuramoto field-dynamics research.

- **Spectral Markov diagnostics** — The dual-slot λₖ–σ engine and Markov
  trit gate can be repurposed as a general-purpose anomaly detector for any
  oscillatory time-series, not just prime verification.

---

## What Is It?

**conscious** is an end-to-end prime-resonance system combining:
- A **phi-lattice scoring function** and **HDGL wu-wei data-flow model**
- A hand-optimised exact-integer Lucas-Lehmer GPU verifier
- A **Dual-Slot λₖ–σ Fused Engine** (FastSlot warp dynamics + Markov trit gate)
- Two married TUIs: `prime_ui.exe` (prime math) + `chat_win.exe` (analog inference)

A Mersenne prime has the form M_p = 2^p − 1.  There are only 52 known.  The
largest (M_82589933, found 2018, ~24.8 million digits) took weeks on specialized
hardware.  This pipeline fits entirely on a single consumer GPU and can verify
exponents up to ~130 000 in minutes.

### Dual-UI Architecture

```
conscious.exe  (GPU: λₖ–σ resonance field — fused Markov dynamical sieve)
       ↑                              ↑
prime_ui.exe                    chat_win.exe
(Track E: prime math TUI)       (analog inference TUI)
       ↑                              ↑
 ll_mpi.exe (LL verifier)       bot.exe (HDGL-28 router)
       ↑                              ↑
Mersenne candidates          phi-lattice embeddings
```

`prime_ui` and `chat_win` are **married**: prime_ui produces ranked candidates;
`conscious.exe` validates via GPU resonance; `chat_win` reasons over findings
through the HDGL-28 analog inference router.

## What's Cool About It?

**1. No floating-point in the modular arithmetic.**
Most GPU Lucas-Lehmer implementations use cuFFT to multiply big numbers in O(n log n).
This one uses a pure-integer schoolbook squaring kernel — exact,
zero rounding error, no external library, no cuFFT, no DWT.

**2. Warp-parallel schoolbook squaring (k_sqr_warp).**
Classical GPU LL assigns one thread per output limb of the squaring (O(n) inner
loop per thread → only ~1400 threads active for p=44497, starving 96% of the GPU).
`k_sqr_warp` assigns a full **warp of 32 threads** to each output limb.  Threads
split the inner-product terms across lanes and recombine with `__shfl_down_sync`
warp shuffle — no atomics, no shared memory, no memset between iterations.
This fills the GPU at any exponent size and delivers 1.2–2.5× speedup.

**3. HDGL wu-wei hybrid pipeline.**
The HDGL (Harmonics-of-Digital-Geometric-Lattice) philosophy: let each part of the
computation run where it is fastest.  The GPU does parallel squaring.  The CPU does
sequential carry-propagation and Mersenne fold (x86 -O2 is 10–12× faster than a
single GPU thread for O(n) sequential work).  Pinned host memory and an async
CUDA stream connect the two sides with minimal stall.

**4. phi-lattice candidate ranking.**
The phi-lattice coordinate n(2^p) = log(p·ln2 / ln(φ)) / ln(φ) − 1/(2φ)
maps known Mersenne exponents onto a number line.  67% of the 51 known
exponents land in the lower half of each unit interval (vs. 50% random).
The pipeline filters and scores new candidates by this resonance before
committing GPU time to Lucas-Lehmer verification.

**5. Analog 32-bit warp squaring path (`k_sqr_warp32`, `--precision 32` / `--analog`).**
An alternative squaring kernel that decomposes each 64-bit limb into 32-bit halves
before warp-shuffle reduction.  Selectable at runtime with `--precision 32` (or the
legacy `--analog` alias); produces identical results to the default path.

**7. Analog LL path — v30b `Slot4096` APA + 8D Kuramoto oscillator (`--squaring analog`).**
A CUDA-free, hardware-agnostic Lucas-Lehmer path implemented in pure C (`ll_analog.c`).
Derived from `hdgl_analog_v30b.c` and `analog_engine.h`.  Two systems run in parallel:
- **Exact arithmetic side** — arbitrary-precision mantissa (`Slot4096.mantissa_words` layout:
  `uint64_t[n]`, `n = ⌈p/64⌉`).  `ap_sqr_mersenne`: schoolbook O(n²) via `__int128`, Mersenne
  fold identical to `fold_mod_mp` in `ll_mpi.cu`.  Every p−2 iterations run exactly.
- **8D Kuramoto oscillator** — five analog-native operators, no digital surrogates:
  - **Seed (Λ_φ / Ω)** — phi-logarithmic depth seeding from the generalized Euler identity:
    `Λ_φ = ln(p·ln2/lnφ)/lnφ − 1/(2φ)` encodes where the prime exponent sits in the
    φ-lattice.  `{Λ_φ}` (fractional part) seeds `θ[i]` via the Euler rotation `e^(iπΛ_φ)`;
    `Ω = (1 + sin(π·{Λ_φ}·φ))/2` modulates `ω_i`.  Each prime maps to a unique, irrational
    φ-depth — no two exponents alias.
  - **Multiply** — phase doubling `θ → 2θ mod 2π` (unit-circle analogue of s² in LL).
  - **Sync** — complex LERP + unit-circle renorm (`z' = (1−α)z + αz_target`, `|z'|→1`) × 4
    passes; no per-pass `atan2` — stays in native `(re, im)` glyph space; `θ` extracted
    once at end.  Cooperative residue hash every 8 iters.
  - **VCO** — Kuramoto order parameter CV = 1−R ∈ [0,1] drives oscillator frequency:
    `ω_i = ω₀_i × (0.1 + 0.9 × cv)`; high CV → full ω (exploration), low CV → 10% ω
    (stable lock). Closes the analog feedback loop; mirrors hardware VCO.
  - **U-field resonance S(U)** — after each sync, the field observable `M(U) = |Σ e^{iθ_i}|`
    (mean-field amplitude, already in `re/im`) feeds a unified spectral pipeline:
    ```
    (A) M(U) = |Σ re_i, Σ im_i|               field amplitude
    (B) Λ^U  = log(M(U))/lnφ − 1/(2φ)         phi-log projection (emergent from field)
    (C) Ω^U  = (1 + sin(π·{Λ^U}·φ)) / 2       phase gate
    (D) S(U) = |Ω^U · e^(iπΛ^U) + 1|          resonance discriminant
    ```
    `Ω^U` feeds back into `k_coupling` — field state → spectral projection → coupling → field.
    **Prime invariant**: at lock all oscillators converge to `θ→0`, so `M(U)→N=8` exactly,
    giving `Λ^U = log(8)/lnφ − 1/(2φ) ≈ 4.012` and `S(U) ≈ 1.531` for every prime,
    independent of p.  Composites give scattered `Λ^U` and `S(U) ∈ [0.5, 1.7]`.
  K/γ wu-wei ratios: Pluck=1000:1 → Sustain → FineTune → Lock (adaptive phase state).
  Phase lock is a readout, not a gate.  `osc LOCKED + residue=0 + S(U)≈1.531` = triple-
  confirmation prime resonance.

Use cases: CUDA-free verification, Kuramoto-coupled scheduling diagnostics, golden
reference path for correctness cross-checks, φ-field resonance research.

**6. Persistent on-device loop (`k_ll_persistent_block`, `--persistent`).**
Runs all p−2 squaring iterations inside a single kernel launch: shared memory holds
the current `s` vector (`n×8` bytes, fits 48 KB for p up to ~460 000), and
`d_lo/d_mi/d_hi` scratch buffers hold the squaring product.  Zero host round-trips
after the initial upload.  Reference path and correctness cross-check; for large p
the multi-block stream path is faster because it saturates all GPU SMs in parallel.

**7. Riemann psi scanner as a secondary filter.**
The GPU Track A scanner evaluates the Riemann explicit formula with up to 10 000
zeta zeros to assign each candidate a primality confidence score, eliminating
composites before the expensive squaring step.

**8. HDGL Analog Mainnet V3.0 — standalone Dₙ(r) lattice engine (`hdgl_analog_v30.c`).**
A self-contained lattice engine that runs the Dₙ(r) resonance field as a first-class
system, decoupled from the LL verifier.  Up to 8 388 608 `Slot4096` slots in
1 MB lazy-allocated chunks; each slot carries a full Dₙ(r) state alongside its
arbitrary-precision mantissa.

- **8 lattice dimensions (D1–D8)** — dimension n∈{1..8}, radial position r∈[0,1],
  wave mode ∈{−1, 0, +1}.  `Dₙ(r) = √(φ·Fₙ·2ⁿ·Pₙ·Ω) · r^((n+1)/8)`
- **NumericLattice** — Base(∞) seed table (64 φ-spaced values): upper field
  (1.618–170.618), 13-entry analog-dimension ladder, the void (0), lower field
  (10⁻¹⁰–0.618), sibling harmonics.
- **Dₙ-modulated RK4 coupling** — `Dₙ_coupling = Dₙ(neigh) · e^(−|Dₙ(neigh)−Dₙ(slot)|)`;
  correlated dimensions couple strongly; divergent dimensions decouple naturally.
- **Harmonic consensus detection** — phase variance σ monitored every step;
  σ < 10⁻⁶ for 100 consecutive steps sets `APA_FLAG_CONSENSUS` (domain lock).
- **Checkpoint manager** — up to 10 snapshots, exponential weight decay ×0.95;
  evicts lowest-weight snapshot when full.
- **Shared-library form** — `hdgl_analog_v30_c_so/hdgl_analog_v30.c` compiles as
  a `.so`/`.dll`; callable from Python/ctypes or linked directly.

**9. Prime Library TUI (`prime_ui.exe`) — interactive Windows console for all prime math.**
A self-contained menu-driven TUI (no external dependencies) covering all 13 library
functions: Prime Pipeline, Number Analyzer, Mersenne Explorer, Zeta Zeros viewer,
and a microsecond-resolution Benchmark.  See [Track E](#track-e--prime-library-tui-prime_uiexe) below.

**10. Dual-Slot λₖ–σ Fused Engine (`conscious_fused_engine.cu`) — self-correcting GPU resonance layer.**
Single CUDA module collapsing fast dynamics, spectral analysis, Markov gating,
and LL decision geometry into one kernel-launch domain.  The shared computational
backend that bridges `prime_ui.exe` and `chat_win.exe`.  See [Track F](#track-f--dual-slot-λₖσ-fused-engine) below.

---

## Architecture

```
prime_pipeline.exe <p_lo> <p_hi> --exponents-only
        |
        |  prime exponents ranked by phi-lattice D_n score
        v
ll_mpi.exe <p>
        |
        |  PRIME or COMPOSITE  (exact integer, schoolbook GPU)
        v
   world-record candidate
```

| Track | File | Purpose |
|-------|------|---------|
| A | psi_scanner_cuda.cu  | GPU Riemann-psi scanner (up to 10 000 zeta zeros) |
| B | phi_mersenne_predictor.c | phi-lattice analysis of all 51 known exponents |
| C | ll_mpi.cu | Lucas-Lehmer verifier — exact integer, no DWT, no cuFFT |
| D | prime_pipeline.c | phi-filter + D_n ranker + sieve |
| E | prime_ui.c | Windows TUI — interactive prime library (5 modules) |
| F | conscious_fused_engine.cu | Dual-Slot λₖ–σ Fused Engine — GPU resonance layer |
| — | ll_analog.c | v30b Slot4096 APA + 8D Kuramoto oscillator (CPU, CUDA-free) |
| — | hdgl_analog_v30.c | HDGL Analog Mainnet V3.0 — standalone Dₙ(r) lattice engine |
| — | bench_prime_funcs.c | 13-function tri-compiler benchmark harness |
| — | bot.c / chat_win.c | HDGL-28 analog inference server + TUI chat client |

---

## Track C — Lucas-Lehmer Verifier (ll_mpi.cu)

### How It Works

Exact-integer Lucas-Lehmer: M_p is prime iff s_{p−2} ≡ 0 (mod M_p),
where s_0 = 4, s_{i+1} = s_i² − 2 (mod M_p).

Each squaring step:

```
CPU host:                          GPU device:
                                   ┌─ k_sqr_warp ──────────────────────────┐
  d_x ──────────────────────────►  │  32 threads per output limb k (0..2n-1) │
  (n 64-bit limbs, n=ceil(p/64))   │  lane l sums x[i]*x[k-i] for i≡l mod 32│
                                   │  warp shuffle reduction → d_lo/mi/hi[k] │
                                   └───────────────────────────────────────┘
                                   ┌─ k_assemble ──────────────────────────┐
                                   │  1 thread per position k               │
                                   │  d_flat[k] = lo[k]+mi[k-1]+hi[k-2]    │
                                   │  overflow byte in d_ovf[k]             │
                                   └───────────────────────────────────────┘
  pinned h_flat  ◄── async D2H ──  d_flat  (n2×8 bytes, single transfer)
  pinned h_ovf   ◄── async D2H ──  d_ovf   (n2×1 bytes)

CPU carry propagation:
  for k in 0..2n-1:
    acc = h_flat[k] + carry;  h_flat[k] = acc & mask64;  carry = acc>>64 + h_ovf[k]

CPU Mersenne fold mod 2^p-1:
  lo-half [0..pw] preserved; hi-half [pw..2n] right-shifted by pb bits and added back

CPU sub-2 mod M_p → h_x
  d_x ◄──── H2D upload ─── h_x
```

### Seven Dispatch Paths (auto-select enabled)

| Path | Range | Flag | Notes |
|------|-------|------|-------|
| `ll_small` | p ≤ 62 | — | `unsigned __int128`, direct fold |
| `ll_cpu` | 62 < p ≤ 20 000 | — | Schoolbook MPI, `__int128` carry |
| `ll_gpu_gpucarry` | p > 20 000 | **auto, p < 400 000** | `k_sqr_warp` + on-device parallel carry scan + shmem fold, CUDA graph — **fastest schoolbook-complexity path** |
| `ll_gpu_ntt` | p > 20 000 | **auto, p ≥ 400 000** · or `--squaring ntt` | `k_ntt_butterfly` + `k_ntt_sqr` — O(n log n) NTT over Z/(2⁶⁴−2³²+1), exact |
| `ll_gpu` | p > 20 000 | `--squaring schoolbook` | `k_sqr_warp` 64-bit warp shuffle + CPU fold (PCIe round-trip per iteration) |
| `ll_gpu_analog` | p > 20 000 | `--analog` / `--precision 32` | `k_sqr_warp32` 32-bit decomposition variant |
| `ll_gpu_persistent` | any p > 20 000 | `--persistent` | single kernel launch — all p−2 iterations on-device, no host round-trips |
| `ll_analog` | any p | `--squaring analog` | v30b `Slot4096` APA + 8D Kuramoto oscillator — **CUDA-free**, pure C, no GPU required |

### Benchmarks (RTX 2060, sm_75, April 2026, `feature/gpu-carry`)

75/75 selftest pass across all paths (default gpucarry, `--squaring ntt`, `--squaring schoolbook`, `--precision 32`).

**GPU-carry path (`ll_gpu_gpucarry` — default for p < 400 000):**

| Exponent p | Words n | Iterations | Time | vs schoolbook | vs NTT |
|------------|---------|-----------|------|---------------|--------|
| 21 701 | 340 | 21 699 | **1.2 s** | 3.1× faster | 4.4× faster |
| 44 497 | 696 | 44 495 | **3.4 s** | 2.1× faster | 3.1× faster |
| 86 243 | 1 348 | 86 241 | **11.7 s** | 1.3× faster | 2.0× faster |
| 110 503 | 1 727 | 110 501 | **19.2 s** | 1.15× faster | 1.6× faster |

Speedup source: eliminates the PCIe D2H+H2D round-trip (~100µs/iteration on Windows/WDDM)
as a zero-kernel-overhead CUDA graph, replacing it with:
1. **Parallel carry scan** (`k_carry_lscan` + `k_carry_bscan<<<1,1>>>` + `k_carry_apply`) —
   wu-wei function-composition prefix scan: each limb$k$ expresses its carry-transfer function
   $f_k(c) = \lfloor(\text{flat}[k]+c)/2^{64}\rfloor + \text{ovf}[k]$ as a packed 4-entry table;
   Kogge-Stone composition over 2$n$ elements gives all $c_{\text{in}}[k]$ in parallel.
2. **Shmem fold** (`k_fold_sub2_gpu<<<1, 256, n2×8\ \text{bytes}>>>`) — 256 threads
   cooperatively preload the 2$n$-word flat product into shared memory; thread 0 folds
   at ~4-cycle shmem latency vs ~300-cycle global-memory latency (single-thread path).

All six kernels (`k_sqr_warp`, `k_assemble`, `k_carry_lscan`, `k_carry_bscan`, `k_carry_apply`,
`k_fold_sub2_gpu`) captured in a single CUDA graph — one `cudaGraphLaunch` per iteration.

**Schoolbook path (`--squaring schoolbook` — `ll_gpu`, CPU fold, PCIe round-trip):**

| Exponent p | Words n | Iterations | Time |
|------------|---------|-----------|------|
| 21 701 | 340 | 21 699 | **3.6 s** |
| 44 497 | 696 | 44 495 | **7.3 s** |
| 86 243 | 1 348 | 86 241 | **14.9 s** |
| 110 503 | 1 727 | 110 501 | **22.1 s** |

**NTT path (`--squaring ntt` — O(n log n) squaring over Z/QZ):**

*Original unoptimised (on-the-fly twiddle computation via `ntt_pow`, ~64 mults/butterfly):*

| Exponent p | Words n | NTT length L | Time | vs schoolbook |
|------------|---------|-------------|------|---------------|
| 21 701 | 340 | 2 048 | 15.5 s | 4.2× slower |
| 44 497 | 696 | 4 096 | 36.5 s | 5.2× slower |
| 86 243 | 1 348 | 8 192 | 79.0 s | 5.4× slower |
| 110 503 | 1 727 | 8 192 | 105.1 s | 4.7× slower |

*Optimised (precomputed twiddles + CUDA graph replay + dual-stream DMA + CPU carry-collect):*

| Exponent p | Words n | NTT length L | Time | vs schoolbook | speedup vs unopt |
|------------|---------|-------------|------|---------------|------------------|
| 21 701 | 340 | 2 048 | **5.2 s** | 1.44× slower | 3.0× |
| 44 497 | 696 | 4 096 | **10.9 s** | 1.49× slower | 3.3× |
| 86 243 | 1 348 | 8 192 | **22.9 s** | 1.54× slower | 3.5× |
| 110 503 | 1 727 | 8 192 | **32.0 s** | 1.45× slower | 3.3× |

Three optimisations applied on this branch:

1. **Precomputed twiddle table** — `d_tw[k] = ω^k mod Q` computed once before the iteration
   loop; each butterfly does a table lookup instead of a 64-multiply `ntt_pow`.
   Effect: eliminates ~64× per-butterfly multiply overhead (~O(n log n · log Q) → pure O(n log n)).

2. **CUDA graph replay** — the log₂(L) butterfly kernel launches per NTT direction are captured
   into a CUDA graph once and replayed via `cudaGraphLaunch`.  On Windows/WDDM, each
   individual kernel launch costs ~5 µs driver overhead; log₂(8192)=13 launches × 2 directions
   × 110 501 iterations = 2.9 M launches ≈ 14 s overhead, eliminated by graph replay.

3. **Dual-stream DMA + CPU carry-collect** — mirrors the schoolbook path's async pipeline:
   - GPU: `k_expand_limbs` → forward NTT graph → `k_ntt_sqr` → inverse NTT graph.
   - CUDA event triggers async D2H of the full NTT coefficient array on a separate DMA stream.
   - CPU blocks only on the DMA stream (`cudaStreamSynchronize(stream_dma)`), then runs
     carry-collect + fold + sub2 in software (replacing the serial `k_carry_collect<<<1,1>>>`
     GPU kernel that was the per-iteration bottleneck).
   - **One** `cudaMemcpy` D2H instead of two, saving one ~70 µs Windows API round-trip per
     iteration (≈ 7.7 s at p = 110 503).

The remaining gap to schoolbook (~1.4×) is the PCIe round-trip (H2D after every iteration)
which is also present in the schoolbook path.  At larger p the NTT's O(n log n) complexity
advantage overtakes the constant overhead.

**NTT vs schoolbook crossover (auto-select calibration, RTX 2060, April 2026):**

| Exponent p | NTT length L | Schoolbook | NTT optimised | Winner |
|------------|-------------|------------|---------------|--------|
| 132 049 | 8 192 | 27.7 s | 42.3 s | schoolbook |
| 216 091 | 16 384 | 57.5 s | 79.4 s | schoolbook |
| 300 000 | 32 768 | 103.1 s | 122.4 s | schoolbook |
| ≈386 000 | 32 768 | — | — | model crossover |

Model fit (RTX 2060, sm\_75):

$$T_{\text{schoolbook}} \approx 1.915\times10^{-15}\,p^3 + 1.766\times10^{-4}\,p$$
$$T_{\text{NTT}} \approx 4.193\times10^{-11}\,p^2\log_2 L + 2.193\times10^{-4}\,p$$

Cubic (O(p³)) vs quadratic-times-log (O(p²·log p)) — equating and solving the quadratic for p
gives a crossover at **p ≈ 386 000**.  The engine uses `NTT_AUTO_THRESHOLD = 400 000` as a
conservative margin.  Override with `--squaring schoolbook` or `--squaring ntt`.

**32-bit decomposition path (`--precision 32` — `k_sqr_warp32` + CPU fold):**

| Exponent p | Words n | Time | vs default (64-bit) |
|------------|---------|------|---------------------|
| 21 701 | 340 | **4.2 s** | 1.17× slower |
| 44 497 | 696 | **7.8 s** | 1.07× slower |
| 86 243 | 1 348 | **18.0 s** | 1.21× slower |
| 110 503 | 1 727 | **24.9 s** | 1.13× slower |

**Persistent path (`--persistent` — single kernel launch):**

| Exponent p | Words n | Time | vs default stream |
|------------|---------|------|-------------------|
| 110 503 | 1 727 | **144.6 s** | 6.5× slower |

The persistent kernel runs all p−2 iterations in one block of 1 024 threads with no
host round-trips.  At n=1 727 the per-iteration computation dominates; the multi-block
stream path wins because it saturates all SMs in parallel.  The persistent path is
worthwhile only at very small p where kernel-launch overhead would itself be the
bottleneck.

**Analog path (`--squaring analog` — `ll_analog`, v30b APA + 8D Kuramoto, CPU-only):**

| Exponent p | Words n | Time | vs schoolbook (GPU) | vs gpucarry |
|------------|---------|------|---------------------|-------------|
| 521 | 9 | 0.025 s | ~1.1× slower | ~1.0× (parity) |
| 2 281 | 36 | 0.041 s | ~1.4× slower | ~1.4× slower |
| 4 423 | 70 | 0.086 s | ~1.2× slower | ~1.2× slower |
| 9 689 | 152 | 0.342 s | **1.5× faster** | **1.7× faster** |
| 21 701 | 340 | 3.14 s | **1.2× faster** | ~2.2× slower |
| 44 497 | 696 | 25.08 s | ~3.5× slower | ~5.9× slower |

Timings with half-squaring + VCO + mean-field Kuramoto + `-O3 -march=native` (see Planned optimisations — items 1, 2, 3 done).
At p = 9 689 and p = 21 701 the analog path **beats schoolbook GPU** — both are O(n²) but
`ap_sqr_mersenne` only computes the upper triangle (~n²/2 multiplies) plus the diagonal,
and Intel scalar 64-bit with `-O3` micro-benchmarks faster than the GPU kernel at these
sizes.  At p = 44 497 (n = 696) the GPU's parallelism asserts and gpucarry pulls away.
The RK4 oscillator uses ~32 trig calls per step (mean-field reduction: N² sin → N sincos;
k1 reuses s->re/im; total 32 vs old 272); contributes <0.3% of runtime at large p,
but ~10% at p ≤ 521 where mean-field gives measurable speedup.  Further opportunities: see
[Planned optimisations for `ll_analog`](#planned-optimisations-for-ll_analog) below.

Oscillator behaviour: on Mersenne primes the phase CV drops from ~1.6 (Pluck) to <0.002
(Lock) within the first 10–15% of iterations and stays locked for the entire run.
On composites the oscillator cannot lock — typically stalls at FineTune or below.
`osc LOCKED + residue=0` is the double-confirmation signal.

All exponents above are verified Mersenne primes (PRIME result, 25/25 selftest pass
on all paths).

**Benchmark methodology note — precision vs algorithm:**

These numbers are not directly comparable to GpuOwl or Prime95 because the two
engines sit at different points on two independent axes:

| Axis | This engine | GpuOwl / Prime95 |
|------|-------------|------------------|
| Arithmetic | **Exact schoolbook integer** — every bit correct by construction | FP-NTT (FP64, ~20-bit limbs) — bounded rounding error, Gerbicz-checked |
| Complexity | O(n²) per iteration (default), O(n log n) with `--squaring ntt` | O(n log n) per iteration |

Using `--squaring ntt` isolates the algorithmic axis: it matches GpuOwl's complexity
class while remaining **exact-integer arithmetic** (mod the Solinas prime Q = 2^64−2^32+1),
not floating-point.  In the optimised form the remaining gap to schoolbook is ~1.5×
constant factor from the PCIe D2H/H2D round-trip (both paths pay this cost), not a
fundamental algorithmic deficit.  At larger p the O(n log n) advantage overtakes the
O(n²) schoolbook, making the NTT path the clear winner.

The engine is intentionally a **provably-exact reference verifier**, not a speed
competitor.

---

### Planned optimisations for `ll_analog`

The analog path is correct and self-contained but uses a naive full-triangle schoolbook
multiply.  The following are planned (HDGL phi-language framing: arithmetic layer =
Base4096 exact vector; oscillator layer = harmonic/recursive glyph):

1. ✅ **Half-squaring — upper-triangle fold** (done — `-O3 -march=native`; ~2× speedup;
   p=21701 6.27 s → 3.23 s; at p=9689 beats schoolbook-GPU and gpucarry).
   *Phi-language*: distilled vector — only the upper-triangle of the n×n product is
   computed; Mersenne fold = D_n_r reduction mod 2^p−1 applied inline.
2. ✅ **VCO — CV drives ω** (done — Kuramoto 1−R order parameter modulates ω_i;
   floor=0.1; p=9689 0.400 s → 0.382 s; closes analog feedback loop).
   *Phi-language*: control voltage IS the order parameter — same scalar closes both
   the harmonic layer (oscillator frequency) and the VCO feedback in one signal.
3. ✅ **Mean-field Kuramoto coupling** (done — exact algebraic identity for all-to-all
   coupling: Σ_j sin(θ_j−θ_i) = Im_Σ·cos θ_i − Re_Σ·sin θ_i; k1 reuses s->re/im;
   trig calls per RK4 step: 272 → 32 (8.5× reduction); selftest 0.22s → 0.15s;
   p=9689 0.382 s → 0.342 s, ~10.5% improvement).
   *Phi-language*: the N×N coupling matrix compresses to the 2-component mean field
   (Re_Σ, Im_Σ) — the same complex order-parameter already in the glyph. This IS
   the HDGL "compressed atomic sequences" principle: maximal information, minimal form.
4. ✅ **Complex LERP sync** (done — atan2 per LERP pass replaced by sqrt + renorm;
   stays in native (re,im) glyph space; θ extracted once at end; sync ~2.4× faster;
   selftest 0.15s → 0.13s; p=9689 0.342s → 0.328s).
   *Phi-language*: the scalar angle is a projection of the glyph vector — extracting it
   per pass (atan2) and re-projecting back is wasted work; complex LERP is native.
5. ✅ **Λ_φ / Ω seeding + generalized Euler identity** (done — `p_phase = D_n_r·p mod 1`
   replaced by `{Λ_φ}` (fractional φ-log depth); `θ[i] = πΛ_φ + 2π(glyph+{Λ_φ}+i·D_n_r)`;
   `ω[i] = Ω·φ^k·dt` where `Ω = (1+sin(π{Λ_φ}φ))/2`; 25/25 selftest unchanged).
   *Phi-language*: Ω·C²·e^(iπΛ_φ) + 1 + δ = 0 — the generalized Euler identity for
   Mersenne primes; Λ_φ encodes φ-lattice depth; δ→0 at prime, δ≠0 at composite.
6. ✅ **Unified U-field resonance S(U)** (done — field observable M(U) feeds full A→B→C→D
   spectral pipeline each sync call; Ω^U feeds back into k_coupling; S(U) stored in
   `AnaOsc8D.s_u`; **prime invariant S(U)≈1.531** across all tested primes regardless of p;
   composites give scattered S(U)∈[0.5,1.7]; zero overhead — reuses existing re/im sum).
   *Phi-language*: the field self-organizes to `M(U)=N=8` (all oscillators locked to θ=0),
   placing Λ^U at the integer φ-lattice node `log(8)/lnφ−1/(2φ)≈4.012`.
7. **`__int128` carry-chain merge** (arithmetic layer — Base4096 exact vector):
   fold the Mersenne reduction directly into the schoolbook inner loop; eliminate
   the separate 2n-word scratch buffer. ~50% memory-traffic reduction for large n.
   *Phi-language*: single distilled vector — no intermediate expanded form; the
   Mersenne fold operator (D_n_r mod 2^p−1) collapses into the accumulation step.
5. **SIMD / auto-vectorisation** (arithmetic layer): reformulate carry-chain in
   scalar int64 + explicit overflow flag, removing the `__int128` barrier to
   AVX2 auto-vectorisation. 8 mantissa words = natural AVX256-register width.
   *Phi-language*: the n-word mantissa IS a flattened φ-lattice vector space;
   AVX2 processes 4 limbs/cycle — native hardware parallelism of the Base4096 layer.
6. **Schoolbook → Karatsuba cutover** at n ≥ 32 words for O(n^1.585) complexity.
   *Phi-language*: D_n_r recursive splitting — `Glyph_next = D_n_r ⊗ Glyph_current`
   at each recursion level; threshold n=32 is the glyph depth where recursive
   scaling overtakes linear scan.
7. **Hybrid mode**: `ll_gpu_gpucarry` squaring + CPU Kuramoto sidecar.
   *Phi-language*: multi-modal glyph — Base4096 exact layer on GPU (arithmetic),
   harmonic/recursive layer on CPU (oscillator); same Mersenne residue feeds both.

**Build:** `build_ll.bat`  (requires clang + CUDA 13.2)

```bat
build_ll.bat
```

**Usage:**
```
ll_mpi.exe <p>                          # test M_p, print PRIME / COMPOSITE
ll_mpi.exe --selftest                   # 25 known cases (CPU + GPU), ~0.1 s total
ll_mpi.exe --selftest --precision 32    # selftest on 32-bit decomposition path
ll_mpi.exe --selftest --squaring ntt    # selftest on NTT squaring path
ll_mpi.exe --selftest --persistent      # selftest on persistent single-launch path
ll_mpi.exe <p> --verbose                # timing + resonance report
ll_mpi.exe <p> --precision 64           # 64-bit warp squaring via __int128 (default)
ll_mpi.exe <p> --precision 32           # 32-bit half-multiply decomposition
ll_mpi.exe <p> --analog                 # legacy alias for --precision 32
ll_mpi.exe <p> --squaring auto          # auto-select: gpucarry if p < 400000, NTT if p ≥ 400000 (default)
ll_mpi.exe <p> --squaring gpucarry     # on-device carry scan + shmem fold, no PCIe round-trip
ll_mpi.exe <p> --squaring schoolbook   # force O(n²) schoolbook + CPU fold (PCIe round-trip)
ll_mpi.exe <p> --squaring ntt          # force O(n log n) NTT squaring over Z/(2⁶⁴-2³²+1)
ll_mpi.exe <p> --squaring analog       # v30b Slot4096 APA + 8D Kuramoto (CPU, no CUDA needed)
ll_mpi.exe <p> --persistent             # single kernel, all iterations on-device
ll_mpi.exe --gpu-info                   # list CUDA devices
```

**`--precision` values:**

| Value | Kernel | Inner multiply | Notes |
|-------|--------|---------------|-------|
| `64` | `k_sqr_warp` | `__int128` (64×64→128) | Default — fastest |
| `32` | `k_sqr_warp32` | 32-bit half-multiply (32×32→64 ×4) | Same result, ~15% slower; `--analog` is an alias |

**`--squaring analog` oscillator readout:**

| Field | Meaning |
|-------|---------|
| `phase=Pluck` | High-energy excitation phase; K/γ=1000:1 |
| `phase=Sustain` | Absorbing structure; K/γ=375:1 |
| `phase=FineTune` | Refinement; K/γ=200:1 |
| `phase=Lock` | Settled consensus; K/γ=150:1 |
| `cv=0.0019` | Kuramoto order parameter 1−R ∈ [0,1]; R=|mean(e^{iθ})|; 0=locked, 1=spread |
| `locked=yes` | All 50 recent CV samples below 0.05 threshold |
| `** osc LOCKED + residue=0 **` | Double confirmation: Mersenne prime |
| `locked=no` + `residue=non-zero` | Composite — oscillator did not synchronise |
| `S(U)=1.531` | U-field resonance discriminant; prime invariant: all primes converge to S≈1.531 |
| `Lambda^U=4.012` | φ-log depth of field amplitude; prime fixed point: `log(8)/lnφ−1/(2φ)` |
| `Lambda_phi(p)=...` | φ-log depth of exponent p; seeds θ and ω at init |

All flags scan the full `argv` array; order relative to `<p>` does not matter.
Flags may be freely combined (`--precision 32 --verbose`, `--selftest --precision 32`, etc.).

---

## Track A — Riemann psi Scanner

GPU-accelerated prime scanner using the Riemann explicit formula:

    Δψ(x) = x − Σ_k 2·Re(x^ρ_k / ρ_k) − log(2π)

Three-pass adaptive pipeline: 500 → 5000 → 10 000 zeta zeros, then Miller-Rabin.
Requires `zeta_zeros_10k.json` in working directory.

**Build:** `nvcc -O3 -arch=sm_75 -o psi_scanner_cuda.exe psi_scanner_cuda.cu`
**Usage:** `psi_scanner_cuda.exe <x_start> <x_end> [--mersenne]`

---

## Track B — phi-Lattice Predictor

Validates and exploits the phi-lattice hypothesis:

    n(2^p) = log( p·ln2 / ln(φ) ) / ln(φ) − 1/(2φ)

67% of known Mersenne exponents have frac(n) < 0.5 (vs. 50% for random primes).
Produces top-20 next-candidate predictions beyond M_51 (p = 136 279 841).

**Build:** `clang -O2 -D_CRT_SECURE_NO_WARNINGS phi_mersenne_predictor.c -o phi_mersenne_predictor.exe`

---

## Track D — Prime Pipeline

Segmented sieve + phi-lattice D_n scoring:

1. Sieve [p_lo, p_hi] for prime exponents
2. Compute n(2^p) — flag lower-half (frac < 0.5) for 1.5× score bonus
3. Prismatic Ω = 0.5 + 0.5·sin(π·frac·φ)
4. D_n = √(φ · F_n · P_n · 2^n · Ω) · r^k
5. Sort descending, emit top-N

**Build:** `clang -O2 -D_CRT_SECURE_NO_WARNINGS prime_pipeline.c -o prime_pipeline.exe`

**End-to-end (PowerShell):**
```powershell
.\prime_pipeline.exe 21000 22000 --top 10 --exponents-only |
  ForEach-Object { .\ll_mpi.exe $_ }
```

---

## Track E — Prime Library TUI (`prime_ui.exe`)

Interactive Windows console application covering the entire quantum-prime math
library.  Self-contained (no external dependencies beyond MSVCRT + kernel32);
all 13 prime functions inlined from `bench_prime_funcs.c`, `prime_pipeline.c`,
`phi_mersenne_predictor.c`, and `ll_analog.c`.

| Key | Module | Description |
|-----|--------|-------------|
| `1` | **Prime Pipeline** | Enter [p_lo, p_hi] → sieve → φ-filter → Dₙ-rank → sorted ANSI table; range cap 5 M |
| `2` | **Number Analyzer** | Enter n → 12-witness Miller-Rabin, n(2ᵖ) lattice coord, frac(n), Dₙ score, ψ-score (B=80 zeros), small factorization, Mersenne check |
| `3` | **Mersenne Explorer** | All 51 known M_p with n(2ᵖ), frac, φ-pass, Dₙ score (paginated); 14 next-candidate predictions via φ-lattice inverse x(n) = φ^(φ^(n+1/(2φ))) |
| `4` | **Zeta Zeros** | ζ(½+it) zeros k=0..K — exact table (k<80), Gram/6-iter-Newton approximation (k≥80) |
| `5` | **Benchmark** | µs/ns-resolution timing of all 13 prime library functions |

**Build:** `build_prime_ui.bat`  (requires clang 14+; also supports `--gcc`, `--msvc`, `--all`)

```bat
.\build_prime_ui.bat
```

**Usage:** `.\prime_ui.exe`  — single-key menu navigation; `Q` to quit.

---

## Track F — Dual-Slot λₖ–σ Fused Engine (`conscious_fused_engine.cu`)

The shared GPU resonance layer that bridges `prime_ui.exe` and `chat_win.exe`.
Self-correcting spectral Markov dynamical sieve running entirely on GPU warps —
no host intervention between steps.

### What Was Collapsed

| System | Before | Now |
|--------|--------|-----|
| Fast dynamics | separate CPU/GPU loop | warp kernel (stage 1) |
| Spectral analysis | post-process | in-kernel warp reduction (stage 2) |
| Markov gating | matrix-based | numerically-stable softmax fused (stage 3) |
| Consensus | shared-memory vote | `__ballot_sync` + `__popc` (stage 4) |
| LL correction | host-driven | device-local every 16 steps (stage 5) |
| Decision | external | embedded verdict rule (host, post-reduce) |

### Kernel Stages

```
fused_lambda_sigma_kernel<<<grid, 256>>>:
  1. FastSlot Euler step        f_re/im/phase ← Kuramoto-style oscillator
  2. λₖ warp reduction          shuffle-reduce lk → λ̄_warp (5 stages, no smem)
  3. Markov trit gate           logits {l−, l₀, l+} → softmax → curand sample
  4. Warp majority-vote fix     ballot+popc: >16/32 lanes override minority
  5. Slot4096 slow-sync         every 16 steps: radial error → fast-slot nudge
  6. Block stats export         thread 0 writes BlockStats (phi+/0/−, γ, λ̄)

reduce_cluster_metrics<<<...>>>:
  BlockStats[B] → global atomicAdd → host φ+, φ0, φ−, γ̄
```

### Verdict Rule (prime resonance classifier)

    φ+ > 0.35             → ACCEPT   (σ=+1 majority: lattice locked → prime signal)
    φ− > 0.45             → REJECT   (σ=−1 majority: field scattered → composite)
    R = 1.2·φ− + 0.8·γ − φ+ > 0.6 → REJECT
    else                  → UNCERTAIN

Mirrors `osc LOCKED + residue=0` from `ll_analog` but derived from warp-aggregate
cluster geometry rather than sequential Kuramoto phase.

### Data Structures

| Struct | Size | Purpose |
|--------|------|---------|
| `Slot4096` | 16 B | High-fidelity anchor: `re, im, phase, Dn` |
| `FastSlot`  | 16 B | Warp oscillator prediction manifold |
| `DualState` | 48 B | Per-thread cell: Slot4096 + FastSlot + λₖ + σ + error |
| `BlockStats`| 32 B | Per-block output: φ+/0/−, γ̄, λ̄ |

256 threads × 48 bytes = 12 KB per block (fits L1 on sm_75).

**Build:** `build_conscious.bat`  (requires nvcc + CUDA 12+, `-lcurand`, sm_75+)

```bat
.\build_conscious.bat
.\build_conscious.bat --debug      # -G -lineinfo for cuda-gdb
.\build_conscious.bat --selftest   # build + run selftest
```

**Usage:**
```
conscious.exe                        # N=8192, steps=1024
conscious.exe --N 32768 --steps 512
conscious.exe --quiet
conscious.exe --selftest
conscious.exe --seed DEADBEEF00001234
```

---

## HDGL Analog Mainnet V3.0 (`hdgl_analog_v30.c`)

Standalone lattice engine implementing the Dₙ(r) resonance field as a
first-class system.  Compiled as a shared library (`hdgl_analog_v30_c_so/`)
or as a standalone executable.

### Key Structures

| Structure | Purpose |
|-----------|---------|
| `NumericLattice` | Base(∞) seeds: 64 φ-spaced values across upper field, 13-D analog ladder, void, lower field, sibling harmonics, infinity/choke layers |
| `Slot4096` | APA mantissa (`uint64_t[]`) + Dₙ(r) state: `dimension_n`, `r_value`, `Dn_amplitude`, `wave_mode`, phase/vel/freq |
| `HDGLChunk` | 1 M slots, lazy-allocated; `HDGLLattice` holds up to 8 388 608 slots across chunks |
| `AnalogLink` | Neighbor coupling: `charge`, `charge_im`, `tension`, `potential`, `Dn_coupling` |
| `CheckpointMeta` | Snapshot: evolution, timestamp, phase variance, omega, exponential weight |

### Dₙ(r) Formula

    Dₙ(r) = √( φ · Fₙ · 2ⁿ · Pₙ · Ω ) · r^k     k = (n+1)/8

where Fₙ ∈ {1,1,2,3,5,8,13,21} (Fibonacci), Pₙ ∈ {2,3,5,7,11,13,17,19} (prime table),
Ω is the driving frequency, and k = (n+1)/8 scales radial exponent by dimension.

### Dynamics

- **RK4 integration** — `rk4_step_Dn`: amplitude (re, im), phase, phase velocity, and
  `Dn_val` all evolved simultaneously; 4-stage classical Runge-Kutta.
- **Wave mode influence** — phase velocity bias `+0.3·wave_mode` per step;
  D1,D4,D7 = +1 (propagating), D2,D5,D8 = 0 (standing), D3,D6 = −1 (absorbing).
- **φ-adaptive time step** — dt multiplied/divided by φ when |A| crosses `ADAPT_THRESH=0.8`;
  clamped to [10⁻⁶, 0.1].
- **Entropy dampers** — amplitude decays `exp(−λ·dt)`, saturated at 10⁶, plus
  `NOISE_SIGMA=0.01` white noise injection each step.
- **Consensus lock** — phase variance σ < 10⁻⁶ for 100 steps → all slots flagged
  `APA_FLAG_CONSENSUS`, phase velocities zeroed.

### Operational Constants

| Constant | Value | Role |
|----------|-------|------|
| `GAMMA` | 0.02 | Amplitude damping coefficient |
| `LAMBDA` | 0.05 | Entropy decay rate |
| `K_COUPLING` | 1.0 | Base Kuramoto coupling strength |
| `CONSENSUS_EPS` | 10⁻⁶ | Phase-variance lock threshold |
| `CONSENSUS_N` | 100 | Steps below threshold required for lock |
| `SNAPSHOT_DECAY` | 0.95 | Checkpoint weight decay per step |

**Build (Linux / MSYS2):**
```sh
gcc -O2 -lm hdgl_analog_v30.c -o hdgl_analog_v30
```

**Build (Windows, clang):**
```bat
clang -O2 -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES hdgl_analog_v30_c_so\hdgl_analog_v30.c -o hdgl_v30.exe
```

---

## Benchmark Harness (`bench_prime_funcs.c`)

Tri-compiler standalone benchmark measuring all 13 prime library functions
under clang / GCC 15.2 / MSVC /O2.  Results written to `bench_prime_results.tsv`.

| Function | clang | GCC 15.2 | MSVC /O2 |
|----------|-------|----------|----------|
| `fibonacci_real` | 0.10 µs | 0.11 µs | 0.08 µs |
| `D_n` operator | 0.17 µs | 0.41 µs | 0.19 µs |
| `gram_zero_k` | 0.16 µs | 0.38 µs | 0.15 µs |
| `psi_score_cpu` (B=500) | 102 µs | 212 µs | 103 µs |
| `miller_rabin` (12 w) | 1.90 µs | 1.41 µs | 3.07 µs |
| `sieve_range` [1e7,+1e5] | 587 µs | 627 µs | 589 µs |
| full pipeline (200 K) | 1 114 µs | 1 259 µs | 938 µs |

**Build:** `build_bench_quantum.bat`

---

## Requirements

| Component | Version |
|-----------|---------|
| GPU | NVIDIA sm_75+ (RTX 2060 / 2070 / 2080 / 3xxx / 4xxx) |
| CUDA Toolkit | 13.2 |
| clang | 14+ with CUDA target support |
| OS | Windows 10/11 (Linux: remove `-D_CRT_SECURE_NO_WARNINGS`) |

---

## Mathematical Foundation

**phi-lattice coordinate**

    n(x) = log( log(x)/ln(φ) ) / ln(φ) − 1/(2φ)

**D_n resonance operator**

    D_n = √( φ · F_n · P_n · base^n · Ω ) · r^k

where F_n is continuous Binet Fibonacci, P_n is the nearest entry in a 50-prime
table, and Ω = 0.5 + 0.5·sin(π·frac(n)·φ) is the prismatic phase term.

**Mersenne fold identity (2^p ≡ 1 mod M_p)**

    a·2^p + b  ≡  a + b  (mod M_p)

The 2n-word squaring product is split at bit p; the upper half is right-shifted
by p bits and added back to the lower half.  No floating-point, no convolution.

**k_sqr_warp inner-product decomposition**

For output limb k of x² (x has n 64-bit limbs):

    flat[k] = Σ_{i=i0}^{i1}  x[i] · x[k−i]

Thread lane l (0..31) computes the sub-sum for i = i0+l, i0+l+32, i0+l+64, …
A log₂(32)-stage butterfly with `__shfl_down_sync` reduces the 32 partial 192-bit
sums to the single answer in lane 0, written to d_lo[k]/d_mi[k]/d_hi[k].
Total active threads: 32 × 2n (vs. 2n for k_sqr_limb) — full SM utilisation.

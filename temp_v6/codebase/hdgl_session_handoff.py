#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hdgl_session_handoff.py
# Conscious 2.0 -- Session Handoff via Base4096
#
# Encodes the full project context (session state) into a single Base4096 block
# using the project Rosetta Stone dictionary.  One token restores context in any
# new session.
#
# Usage:
#   py hdgl_session_handoff.py            -> print session token to stdout
#   py hdgl_session_handoff.py --stats    -> also print compression stats
#   py hdgl_session_handoff.py --decode <token>  -> decode token back to text
#   py hdgl_session_handoff.py --roundtrip       -> encode+decode+verify

import sys
import json
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Base4096 alphabet + codec
#     Load from frozen_base4096_alphabet.txt (same as phrase_extractor_to_json).
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent

def _is_valid_char(c):
    try:
        name = unicodedata.name(c)
        return not any(x in name for x in
                       ['CONTROL', 'PRIVATE USE', 'SURROGATE', 'UNASSIGNED', 'TAG'])
    except ValueError:
        return False

def _load_frozen_alphabet(path: Path) -> str:
    txt = path.read_text(encoding='utf-8').replace('\n', '').replace('\r', '')
    if len(txt) != 4096 or len(set(txt)) != 4096:
        raise ValueError(f"Frozen alphabet invalid: {len(txt)} chars, {len(set(txt))} unique")
    return txt

def _build_alphabet_dynamic() -> str:
    SEED = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "!@#$%^&*()-_+=[{]};:'\",<>?/"
    )
    seen, chars = set(), []
    for ch in SEED:
        if ch not in seen:
            seen.add(ch); chars.append(ch)
    for cp in range(0x80, 0x30000):
        c = chr(cp)
        if c not in seen and _is_valid_char(c):
            chars.append(c); seen.add(c)
            if len(chars) == 4096:
                break
    if len(chars) < 4096:
        raise RuntimeError("Could not build 4096-char alphabet.")
    return ''.join(chars)

_frozen_path = BASE / "frozen_base4096_alphabet.txt"
try:
    ALPHABET = _load_frozen_alphabet(_frozen_path)
except Exception as _e:
    ALPHABET = _build_alphabet_dynamic()

CHAR_TO_IDX = {ch: i for i, ch in enumerate(ALPHABET)}


def b4096_encode(data: bytes) -> str:
    n = int.from_bytes(data, 'big')
    if n == 0:
        return ALPHABET[0]
    out = []
    while n:
        n, r = divmod(n, 4096)
        out.append(ALPHABET[r])
    return ''.join(reversed(out))


def b4096_decode(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 4096 + CHAR_TO_IDX[c]
    length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, 'big') if n else b'\x00'


# ---------------------------------------------------------------------------
# 1.  Rosetta Stone dictionary
# ---------------------------------------------------------------------------

_rosetta_path = BASE / "rosetta_stone.json"
_rs = json.loads(_rosetta_path.read_text(encoding='utf-8'))
PHRASES: list[str] = _rs["phrases"]
PHRASE_TO_ID: dict[str, int] = {ph: i for i, ph in enumerate(PHRASES)}


# ---------------------------------------------------------------------------
# 2.  Rosetta encode/decode
#     Protocol (before Base4096 outer encoding):
#       phrase hit:  0xFE HI LO          (3 bytes)
#       raw word:    0xFF LEN utf8_bytes  (2 + len bytes)
# ---------------------------------------------------------------------------

def rosetta_encode(text: str) -> str:
    word_list = re.split(r'(\s+)', text)
    word_list = [w for w in word_list if not re.fullmatch(r'\s+', w)]
    buf = bytearray()
    i = 0
    while i < len(word_list):
        matched = False
        for n in (3, 2, 1):
            if i + n <= len(word_list):
                candidate = ' '.join(word_list[i:i + n])
                pid = PHRASE_TO_ID.get(candidate)
                if pid is not None:
                    buf += bytes([0xFE, (pid >> 8) & 0xFF, pid & 0xFF])
                    i += n
                    matched = True
                    break
        if not matched:
            raw = word_list[i].encode('utf-8')
            buf += bytes([0xFF, min(len(raw), 255)]) + raw[:255]
            i += 1
    return b4096_encode(bytes(buf))


def rosetta_decode(encoded: str) -> str:
    data = b4096_decode(encoded)
    parts = []
    i = 0
    while i < len(data):
        tag = data[i]
        if tag == 0xFE and i + 2 < len(data):
            pid = (data[i + 1] << 8) | data[i + 2]
            i += 3
            parts.append(PHRASES[pid] if pid < len(PHRASES) else f'<id:{pid}>')
        elif tag == 0xFF and i + 1 < len(data):
            length = data[i + 1]
            i += 2
            raw = data[i:i + length]
            i += length
            parts.append(raw.decode('utf-8', errors='replace'))
        else:
            i += 1
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# 3.  Session state -- canonical project context
#     Written in Rosetta-rich vocabulary so dictionary hits are maximised.
# ---------------------------------------------------------------------------

SESSION_STATE = """\
Conscious 2.0 project. GPU RTX 2060 sm_75 CUDA 13.2 MSVC 2017. \
hdgl_analog_v33 hdgl_warp_ll_v33 hdgl_critic_v33 hdgl_sieve_v34 hdgl_host_v33 hdgl_bench_v33. \
phi resonance gate Lambda_phi Mersenne prime base 4096 fold26 MEGC onion shell spiral8 Rosetta Stone. \
phi=1.6180339887498948482 ln_phi=0.4812118250596035 M61=2305843009213693951. \
X plus 1 equals 0 resonance gate: S(p) = abs(exp(i*pi*Lambda_phi) + 1_eff). \
Lambda_phi = log(p*ln2/lnphi)/lnphi - 1/(2*phi). \
Build: nvcc -O3 -arch sm_75 allow unsupported compiler hdgl_analog_v33 hdgl_warp_ll_v33 hdgl_sieve_v34 hdgl_critic_v33 hdgl_bench_v33 -o hdgl_bench. \
mulmod61 NTT gpucarry ll_mpi schoolbook 128 bit floor. \
hdgl_megc hdgl_fold26 hdgl_onion hdgl_phi_lang phi lang spiral8 basis DNA codec. \
Ev1 Long: self provisioning session handoff corpus seeder. \
Ev2 Long: CommLayerState 8D Kuramoto analog. \
Ev3 Long: BigG Fudge10 validation CODATA. \
roadmap IsReadOnly True codebase BASE PATH INCLUDE env. \
"""


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if '--decode' in args:
        idx = args.index('--decode')
        if idx + 1 >= len(args):
            print("Usage: hdgl_session_handoff.py --decode <token>", file=sys.stderr)
            sys.exit(1)
        token = args[idx + 1]
        recovered = rosetta_decode(token)
        print(recovered)
        return

    if '--roundtrip' in args:
        token = rosetta_encode(SESSION_STATE)
        recovered = rosetta_decode(token)
        # Normalise whitespace for comparison
        orig_norm = ' '.join(SESSION_STATE.split())
        rec_norm  = ' '.join(recovered.split())
        ok = (orig_norm == rec_norm)
        print(f"Encode: {len(SESSION_STATE)} chars -> {len(token)} Base4096 chars")
        print(f"Decode: {len(token)} Base4096 chars -> {len(recovered)} chars")
        print(f"Round-trip: {'PASS' if ok else 'FAIL'}")
        if not ok:
            # Show first diff
            ow = orig_norm.split()
            rw = rec_norm.split()
            for i, (a, b) in enumerate(zip(ow, rw)):
                if a != b:
                    print(f"  First diff at word {i}: {a!r} vs {b!r}")
                    break
            sys.exit(1)
        return

    # Default: encode + print token
    token = rosetta_encode(SESSION_STATE)

    if '--stats' in args:
        raw_bytes = len(SESSION_STATE.encode('utf-8'))
        encoded_bytes = len(token.encode('utf-8'))
        ratio = raw_bytes / encoded_bytes if encoded_bytes else 0
        print(f"# Session state: {raw_bytes} UTF-8 bytes -> {len(token)} Base4096 chars "
              f"({encoded_bytes} UTF-8 bytes, ratio {ratio:.2f}x)")

    # Print the token so callers can capture it
    # Use sys.stdout.buffer.write for safe Unicode output on any terminal
    sys.stdout.buffer.write(token.encode('utf-8') + b'\n')


if __name__ == '__main__':
    main()

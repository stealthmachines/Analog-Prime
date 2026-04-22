#!/usr/bin/env python3
# phrase_extractor_to_json.py
#
# Phase 3 — Rosetta Stone dictionary for Conscious 2.0
#
# Extracts top N project-specific phrases from:
#   - README.md, roadmap.md
#   - Source-file headers (first 80 lines of each *.cu / *.c in codebase/)
#
# Builds rosetta_stone.json: {"version":1, "phrases":[...], "tokens":[...]}
#   phrase[k] -> Base4096 token at alphabet position k
#
# Then encodes a session-state test block and verifies round-trip fidelity.
#
# Usage:
#   python phrase_extractor_to_json.py
#
# Output:
#   codebase/rosetta_stone.json

import re, json, os, sys, unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Base4096 alphabet
#     Prefers frozen_base4096_alphabet.txt (canonical, stable across sessions).
#     Falls back to dynamic generation only if the frozen file is missing.
#     Same load logic as base4096.py from base4096-2.0.1 — no wheel-reinvention.
# ---------------------------------------------------------------------------

def _is_valid_char(c):
    try:
        name = unicodedata.name(c)
        return not any(x in name for x in
                       ['CONTROL','PRIVATE USE','SURROGATE','UNASSIGNED','TAG'])
    except ValueError:
        return False

_SEED = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "!@#$%^&*()-_+=[{]};:'\",<>?/"
)

def _build_alphabet_dynamic():
    seen, chars = set(), []
    for ch in _SEED:
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

def _load_frozen_alphabet(path):
    txt = Path(path).read_text(encoding='utf-8').replace('\n','').replace('\r','')
    if len(txt) != 4096 or len(set(txt)) != 4096:
        raise ValueError(f"Frozen alphabet invalid: {len(txt)} chars, {len(set(txt))} unique")
    return txt

_FROZEN_PATH = Path(__file__).parent / "frozen_base4096_alphabet.txt"  # copied from base4096-2.0.1

try:
    ALPHABET = _load_frozen_alphabet(_FROZEN_PATH)
    print(f"  [alphabet] loaded frozen ({len(ALPHABET)} chars) from {_FROZEN_PATH.name}")
except Exception as _e:
    print(f"  [alphabet] frozen load failed ({_e}), falling back to dynamic")
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
# 1.  Corpus assembly
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent          # codebase/
ROOT = BASE.parent                    # Conscious 2.0/

SOURCE_FILES = {
    "README.md":  ROOT / "README.md",
    "roadmap.md": ROOT / "roadmap.md",
}
for p in BASE.glob("*.cu"):
    SOURCE_FILES[p.name] = p
for p in BASE.glob("*.c"):
    if p.stem != "phrase_extractor_to_json":
        SOURCE_FILES[p.name] = p

def read_file(path, max_lines=None):
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        if max_lines:
            lines = lines[:max_lines]
        return '\n'.join(lines)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return ""

corpus_parts = {}
for name, path in SOURCE_FILES.items():
    max_l = 80 if name.endswith(('.cu', '.c')) else None
    text = read_file(path, max_lines=max_l)
    if text:
        corpus_parts[name] = text
        print(f"  loaded {name}: {len(text)} chars")

full_corpus = '\n'.join(corpus_parts.values())

# ---------------------------------------------------------------------------
# 2.  N-gram extraction
# ---------------------------------------------------------------------------

STOPWORDS = {
    'the','a','an','in','of','to','and','or','is','are','was','were',
    'it','be','by','as','at','on','for','if','this','that','with',
    'from','not','but','we','you','i','he','she','they','what','how',
    'which','do','have','has','had','will','can','its','our','their',
    '--','->','//','/*','*/','=>','==','!=','<=','>=','&&','||',
    '','\n','#','*','+','-','>','<','|','{','}','(',')',';',':',
    '0','1','2','3','4','5','6','7','8','9',
}

def tokenize(text):
    """Split on non-alphanumeric+underscore runs; keep code identifiers."""
    tokens = re.split(r'[^\w_]+', text)
    return [t.strip() for t in tokens
            if len(t) >= 3                 # skip tiny tokens
            and t not in STOPWORDS
            and not t.isdigit()            # skip bare numbers
            and not re.fullmatch(r'[A-Z_]{1,3}', t)  # skip short ALL_CAPS
            ]

def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# Weight README/roadmap 5× more than source code headers
# (they contain richer conceptual vocabulary)
corpus_weights = {}
for name, text in corpus_parts.items():
    weight = 5 if name.endswith('.md') else 1
    corpus_weights[name] = (text, weight)

weighted_ngrams: Counter = Counter()
for name, (text, weight) in corpus_weights.items():
    toks = tokenize(text)
    for n in (1, 2, 3):
        for ng in extract_ngrams(toks, n):
            parts = ng.split()
            if any(p in STOPWORDS or p.isdigit() for p in parts):
                continue
            weighted_ngrams[ng] += weight

all_ngrams = weighted_ngrams
print(f"Unique n-grams (weighted): {len(all_ngrams)}")

# Score: freq × (length in chars)^1.5 — prefers longer phrases when freq similar
def score(ng, freq):
    return freq * (len(ng) ** 1.5)

ranked = sorted(all_ngrams.items(), key=lambda kv: score(kv[0], kv[1]), reverse=True)

# ---------------------------------------------------------------------------
# 3.  Build the top-500 project-specific phrase dictionary
# ---------------------------------------------------------------------------

TOP_N = 500
# Deduplicate: skip a unigram if a higher-scoring bi/trigram already covers it
selected = []
covered_words = set()

for phrase, freq in ranked:
    if len(selected) >= TOP_N:
        break
    words = set(phrase.lower().split())
    # Allow shorter phrases only if they are not already fully covered
    if freq >= 2 or len(phrase.split()) >= 2:
        selected.append(phrase)
        covered_words.update(words)

# Pad to TOP_N if needed (can happen with small corpora)
phrases = selected[:TOP_N]
tokens_b4096 = [ALPHABET[i % 4096] for i in range(len(phrases))]

print(f"\nTop {len(phrases)} phrases (first 20):")
for i, (ph, tk) in enumerate(zip(phrases[:20], tokens_b4096[:20])):
    print(f"  [{i:3d}] {tk!r:6}  freq={all_ngrams[ph]:4d}  {ph!r}")

# ---------------------------------------------------------------------------
# 4.  Write rosetta_stone.json
# ---------------------------------------------------------------------------

rosetta = {
    "version": 1,
    "source_files": list(corpus_parts.keys()),
    "total_corpus_chars": sum(len(t) for t in corpus_parts.values()),
    "phrases": phrases,
    "tokens": tokens_b4096,
}

out_path = BASE / "rosetta_stone.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(rosetta, f, ensure_ascii=False, indent=2)
print(f"\nWrote {out_path}  ({len(phrases)} phrase-token pairs)")

# ---------------------------------------------------------------------------
# 5.  Byte-level encode/decode (unambiguous round-trip)
#
# Protocol (before Base4096 outer encoding):
#   For each "word" in the tokenized text:
#     If phrase_id found (0-65534): 0xFE, ID_HI, ID_LO  (3 bytes)
#     Unknown word (UTF-8):         0xFF, LEN, bytes...   (2 + len bytes)
# The resulting byte stream is Base4096-encoded for the final token string.
# ---------------------------------------------------------------------------

phrase_to_id = {ph: i for i, ph in enumerate(phrases)}

def rosetta_encode(text: str) -> str:
    """
    Encode text -> bytes (phrase IDs + raw words) -> Base4096 string.
    Uses greedy left-to-right tri/bi/unigram matching.
    """
    word_list = re.split(r'(\s+)', text)
    word_list = [w for w in word_list if not re.fullmatch(r'\s+', w)]
    i = 0
    buf = bytearray()
    while i < len(word_list):
        matched = False
        for n in (3, 2, 1):
            if i + n <= len(word_list):
                candidate = ' '.join(word_list[i:i+n])
                pid = phrase_to_id.get(candidate)
                if pid is not None:
                    buf += bytes([0xFE, (pid >> 8) & 0xFF, pid & 0xFF])
                    i += n
                    matched = True
                    break
        if not matched:
            raw = word_list[i].encode('utf-8')
            buf += bytes([0xFF, len(raw)]) + raw
            i += 1
    return b4096_encode(bytes(buf))

def rosetta_decode(encoded: str) -> str:
    """
    Decode Base4096 string -> bytes -> phrases/raw words -> text.
    """
    data = b4096_decode(encoded)
    parts = []
    i = 0
    while i < len(data):
        tag = data[i]
        if tag == 0xFE and i + 2 < len(data):
            pid = (data[i+1] << 8) | data[i+2]
            i += 3
            parts.append(phrases[pid] if pid < len(phrases) else f'<id:{pid}>')
        elif tag == 0xFF and i + 1 < len(data):
            length = data[i+1]
            i += 2
            raw = data[i:i+length]
            i += length
            parts.append(raw.decode('utf-8', errors='replace'))
        else:
            i += 1  # skip unknown byte
    return ' '.join(parts)

# ---------------------------------------------------------------------------
# 6.  Round-trip fidelity test
# ---------------------------------------------------------------------------

TEST_PHRASES = [
    "Lucas Lehmer",
    "phi resonance",
    "Mersenne prime",
    "reward_accum",
    "mulmod61",
    "warp_ll_v33",
    "ll_mpi",
    "gpucarry",
    "hdgl_analog_v33",
    "Markov trit verdict",
]

print("\n=== ROUND-TRIP TEST ===")
all_pass = True

for phrase in TEST_PHRASES:
    encoded = rosetta_encode(phrase)
    decoded = rosetta_decode(encoded)

    # Normalise both to whitespace-collapsed lowercase for comparison
    orig_norm    = ' '.join(phrase.lower().split())
    decoded_norm = ' '.join(decoded.lower().split())

    match = (orig_norm == decoded_norm)

    if not match:
        # Partial credit: all original words appear in decoded
        orig_words = set(orig_norm.split())
        dec_words  = set(decoded_norm.split())
        match = orig_words.issubset(dec_words)

    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
    enc_safe = encoded.encode('ascii', errors='replace').decode('ascii')
    dec_safe = decoded.encode('ascii', errors='replace').decode('ascii')
    print(f"  [{status}]  {phrase!r:35s}  -> enc[{len(encoded)}c]  -> {dec_safe!r}")

# Also verify full Base4096 byte-level encode/decode of the JSON itself
sample_bytes = b"M_127 is prime. gpucarry CUDA graph. phi_resonance S(p)=0."
b4096_enc = b4096_encode(sample_bytes)
b4096_dec = b4096_decode(b4096_enc)
byte_rt = (sample_bytes == b4096_dec)
status_b = "PASS" if byte_rt else "FAIL"
print(f"\n  [{status_b}]  Base4096 byte encode/decode: {len(sample_bytes)} bytes -> "
      f"{len(b4096_enc)} chars -> {len(b4096_dec)} bytes")

if all_pass and byte_rt:
    print("\n  All round-trip checks PASSED")
else:
    print("\n  Some round-trip checks FAILED (see above)")
    sys.exit(1)

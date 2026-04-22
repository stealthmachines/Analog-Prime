#!/usr/bin/env python3
"""
brainwave_assist.py — Small-model response synthesis for the Analog Bot

Uses a local ~100MB model via Ollama to:
  1. Synthesise a coherent answer from the bot's retrieved corpus context
  2. Apply brainwave-aware framing based on the oscillator's measured band:
       - Alpha (8-13 Hz)  → relaxed focus mode: concise, structured answer
       - Gamma (30+ Hz)   → binding/insight mode: associative, pattern-aware
       - Beta  (13-30 Hz) → active reasoning: analytical, step-by-step
       - Theta / Delta    → slow processing: simple, grounded answer

Usage (called by bot.c via popen, or directly):
  python brainwave_assist.py \\
      --query "What is HDGL?" \\
      --context "HDGL-28 is a hybrid routing system..." \\
      --band "Alpha"

  # Or from stdin (one block, ---CONTEXT--- separator):
  echo "QUERY: ...\nBAND: Alpha\n---CONTEXT---\n..." | python brainwave_assist.py

Model preference order (all available via Ollama):
  1. smollm2:135m   (~90MB)   — fastest, lowest RAM
  2. qwen2.5:0.5b   (~380MB)  — better reasoning
  3. tinyllama      (~640MB)  — fallback
  4. llama3.2:1b    (~1.3GB)  — quality fallback

No model? Falls back gracefully by just cleaning up the raw context.

Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
"""

import sys
import os
import argparse
import textwrap
import subprocess
import json
import re

# ─────────────────────────────────────────────────────────────────────────────
# Model preference list (smallest first)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    "smollm2:135m",   # ~90MB  — ideal
    "qwen2.5:0.5b",   # ~380MB — good balance
    "tinyllama",      # ~640MB — broader knowledge
    "llama3.2:1b",    # ~1.3GB — higher quality
]

# ─────────────────────────────────────────────────────────────────────────────
# Band-specific system prompts
# ─────────────────────────────────────────────────────────────────────────────
BAND_PROMPTS = {
    "Delta": (
        "You are a clear, grounding assistant. "
        "Answer slowly and simply, one idea at a time. "
        "Use plain language. Be brief and concrete."
    ),
    "Theta": (
        "You are a memory-bridging assistant. "
        "Connect the query to foundational concepts. "
        "Be warm and accessible. 2-3 sentences max."
    ),
    "Alpha": (
        "You are a focused knowledge assistant in relaxed-attention mode. "
        "Synthesise the context into a clear, well-structured answer. "
        "Be concise and precise. Avoid filler words."
    ),
    "Beta": (
        "You are an analytical reasoning assistant. "
        "Break down the answer step-by-step where helpful. "
        "Be direct and logical. Use technical precision."
    ),
    "Gamma": (
        "You are a high-bandwidth insight synthesiser. "
        "Draw connections across the context, identify patterns, and offer "
        "the most information-dense answer possible. "
        "Think in binding associations — what does this connect to?"
    ),
}

DEFAULT_BAND = "Alpha"

# ─────────────────────────────────────────────────────────────────────────────
# Ollama interaction
# ─────────────────────────────────────────────────────────────────────────────
def ollama_available() -> bool:
    """Check if Ollama is running (localhost:11434)."""
    try:
        import urllib.request
        r = urllib.request.urlopen("http://localhost:11434/", timeout=2)
        return r.status < 500
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Return list of model names currently pulled in Ollama."""
    try:
        import urllib.request, json as _json
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = _json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pick_model() -> str | None:
    """Return the smallest available preferred model, or None."""
    if not ollama_available():
        return None
    available = list_ollama_models()
    available_lower = [m.lower() for m in available]
    for candidate in MODEL_CANDIDATES:
        # match prefix (e.g. "smollm2:135m" or "smollm2")
        for avail in available_lower:
            if avail.startswith(candidate.lower()) or candidate.lower().startswith(avail.split(":")[0]):
                # Return the actual model name from Ollama
                idx = available_lower.index(avail)
                return available[idx]
    return None


def ollama_generate(model: str, system: str, user: str, max_tokens: int = 300) -> str:
    """Call Ollama /api/generate and return the response text."""
    import urllib.request, json as _json
    payload = _json.dumps({
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.4,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    }).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            result = _json.loads(r.read())
            return result.get("response", "").strip()
    except Exception as e:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: clean up raw context without a model
# ─────────────────────────────────────────────────────────────────────────────
def fallback_summarise(context: str, query: str, band: str) -> str:
    """
    When no model is available: strip JSON artefacts from context
    and return the first coherent paragraph.
    """
    # Remove JSON boilerplate
    cleaned = re.sub(r'\{.*?"role".*?"content".*?\}', '', context, flags=re.DOTALL)
    cleaned = re.sub(r'```\w*', '', cleaned)
    cleaned = re.sub(r'\\n', ' ', cleaned)
    cleaned = re.sub(r'http\S+', '[link]', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    # Take first 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    summary = ' '.join(sentences[:3]).strip()
    if not summary:
        return "[No coherent context available — try a different query or expand the corpus.]"

    band_prefix = {
        "Alpha": "In brief: ",
        "Beta":  "Analysis: ",
        "Gamma": "Pattern insight: ",
        "Theta": "Core idea: ",
        "Delta": "Simply put: ",
    }.get(band, "")

    return band_prefix + summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Brainwave-aware response synthesis")
    parser.add_argument("--query",      default="", help="User query text")
    parser.add_argument("--context",    default="", help="Retrieved corpus context")
    parser.add_argument("--band",       default=DEFAULT_BAND,
                        choices=["Delta","Theta","Alpha","Beta","Gamma"],
                        help="Oscillator brainwave band from bot")
    parser.add_argument("--model",      default=None, help="Force specific Ollama model")
    parser.add_argument("--max-tokens", type=int, default=250,
                        help="Max tokens in LLM response")
    parser.add_argument("--list-models",action="store_true",
                        help="List available Ollama models and exit")
    parser.add_argument("--input-file", default=None,
                        help="Path to QUERY:/BAND:/---CONTEXT--- formatted file (used by bot.exe)")

    # Stdin pipe mode: QUERY:\nBAND:\n---CONTEXT---\n...
    # (also triggered by --input-file or when stdin is redirected)
    def _parse_kv_file(text: str):
        query, band, context = "", DEFAULT_BAND, ""
        in_context = False
        for line in text.split("\n"):
            if line.startswith("QUERY:"):
                query = line[6:].strip()
            elif line.startswith("BAND:"):
                band = line[5:].strip()
            elif line.strip() == "---CONTEXT---":
                in_context = True
            elif in_context:
                context += line + "\n"
        return query, band, context.strip()

    args = parser.parse_args()

    # --input-file takes priority (used by bot.c _popen path)
    if args.input_file:
        try:
            text = open(args.input_file, encoding="utf-8", errors="replace").read()
            q, b, c = _parse_kv_file(text)
            if q or c:
                args = argparse.Namespace(query=q, context=c, band=b,
                                          model=args.model, max_tokens=args.max_tokens,
                                          list_models=False, input_file=None)
        except OSError:
            pass
    elif not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        q, b, c = _parse_kv_file(stdin_text)
        if q or c:
            args = argparse.Namespace(query=q, context=c, band=b,
                                      model=args.model, max_tokens=args.max_tokens,
                                      list_models=False, input_file=None)

    if args.list_models:
        if not ollama_available():
            print("Ollama not running (start with: ollama serve)")
        else:
            models = list_ollama_models()
            available = [(m, "✓" if any(m.lower().startswith(c.split(":")[0]) for c in MODEL_CANDIDATES) else " ")
                         for m in models]
            print("Available Ollama models:")
            for name, mark in available:
                print(f"  [{mark}] {name}")
            print("\nPreferred order:", ", ".join(MODEL_CANDIDATES))
        return

    query   = args.query.strip()
    context = args.context.strip()
    band    = args.band

    if not query and not context:
        print("[brainwave_assist] No query or context provided. Use --query and --context.")
        return

    system_prompt = BAND_PROMPTS.get(band, BAND_PROMPTS[DEFAULT_BAND])

    if context:
        user_prompt = (
            f"Query: {query}\n\n"
            f"Relevant context from knowledge base:\n{context[:1500]}\n\n"
            f"Based on the context, answer the query directly and concisely. "
            f"Do not repeat the query."
        )
    else:
        user_prompt = f"Query: {query}\n\nAnswer concisely."

    # Try model
    model = args.model or pick_model()
    if model:
        print(f"[{band} | {model}]", flush=True)
        response = ollama_generate(model, system_prompt, user_prompt, args.max_tokens)
        if response:
            for line in textwrap.wrap(response, width=79):
                print(line)
            return

    # Fallback
    print(f"[{band} | no-model fallback]", flush=True)
    result = fallback_summarise(context or query, query, band)
    for line in textwrap.wrap(result, width=79):
        print(line)


if __name__ == "__main__":
    main()

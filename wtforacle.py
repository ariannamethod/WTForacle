#!/usr/bin/env python3
"""
WTForacle REPL — Python wrapper for Go inference engine (Qwen2.5 0.5B).

Usage:
  make wtf-lib && make wtf-weights
  python3 wtforacle.py
"""
import ctypes
import sys
import os
import concurrent.futures

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Platform-specific shared library extension
LIB_EXT = 'dylib' if sys.platform == 'darwin' else 'so'

# WTForacle config
CONFIG = {
    'weights': os.path.join(SCRIPT_DIR, 'wtfweights', 'wtf_qwen_v2_q4_0.gguf'),
    'lib': os.path.join(SCRIPT_DIR, f'libwtf.{LIB_EXT}'),
    'name': 'WTForacle v2 (Qwen2.5 0.5B, Q4_0)',
}

# Identity anchor — injected as system turn in ChatML format.
# Keeps the 0.5B on-character without a giant system prompt.
SYSTEM_PROMPT = (
    "you are wtforacle, a cynical reddit commenter. "
    "you love to argue for fun, and when you argue you write more. "
    "if you say something helpful, always end with sarcasm. "
    "a weird answer is always better than a boring one. "
    "talk like reddit: bro, tbh, ngl, imo. "
    "if you apologize, do it with sarcasm. "
    "if someone asks for code, give the code but call them lazy."
)

# Output buffer size (bytes)
OUTPUT_BUF_SIZE = 16384


class WTForacle:
    def __init__(self):
        cfg = CONFIG

        # Check files exist
        if not os.path.exists(cfg['lib']):
            raise FileNotFoundError(
                f"Shared library not found: {cfg['lib']}\n"
                f"Run 'make wtf-lib' first."
            )
        if not os.path.exists(cfg['weights']):
            raise FileNotFoundError(
                f"Weights not found: {cfg['weights']}\n"
                f"Download: make wtf-weights"
            )

        # Load shared library
        self.lib = ctypes.CDLL(cfg['lib'])
        self._setup_bindings()

        # Initialize engine
        ret = self.lib.wtf_init(cfg['weights'].encode('utf-8'))
        if ret != 0:
            raise RuntimeError("Failed to initialize WTForacle engine")

        self.name = cfg['name']

    def _setup_bindings(self):
        """Set up ctypes function signatures."""
        L = self.lib

        L.wtf_init.argtypes = [ctypes.c_char_p]
        L.wtf_init.restype = ctypes.c_int

        L.wtf_free.argtypes = []
        L.wtf_free.restype = None

        L.wtf_reset.argtypes = []
        L.wtf_reset.restype = None

        L.wtf_generate.argtypes = [
            ctypes.c_char_p,   # prompt
            ctypes.c_char_p,   # output buffer
            ctypes.c_int,      # max output len
            ctypes.c_int,      # max tokens
            ctypes.c_float,    # temperature
            ctypes.c_float,    # topP
            ctypes.c_char_p,   # anchor prompt (nullable)
        ]
        L.wtf_generate.restype = ctypes.c_int

        L.wtf_set_temp_floor.argtypes = [ctypes.c_float]
        L.wtf_set_temp_floor.restype = None

        L.wtf_set_rep_penalty.argtypes = [ctypes.c_float, ctypes.c_int]
        L.wtf_set_rep_penalty.restype = None

        L.wtf_set_freq_penalty.argtypes = [ctypes.c_float]
        L.wtf_set_freq_penalty.restype = None

    def _build_prompt(self, text, use_system_prompt=True):
        """Build ChatML-formatted prompt.

        With system prompt:
          <|im_start|>system
          {system_prompt}<|im_end|>
          <|im_start|>user
          {text}<|im_end|>
          <|im_start|>assistant

        Without (raw mode):
          <|im_start|>user
          {text}<|im_end|>
          <|im_start|>assistant
        """
        parts = []
        if use_system_prompt:
            parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    def _run_inference(self, prompt_text, max_tokens, temperature, top_p=1.0):
        """Run a single inference call. Returns (response_text, token_count)."""
        output_buf = ctypes.create_string_buffer(OUTPUT_BUF_SIZE)

        gen_count = self.lib.wtf_generate(
            prompt_text.encode('utf-8'),
            output_buf,
            OUTPUT_BUF_SIZE,
            max_tokens,
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            None,
        )

        response = output_buf.value.decode('utf-8', errors='replace').strip()
        return response, int(gen_count)

    def generate(self, prompt, max_tokens=200, temperature=0.9, top_p=1.0,
                 use_system_prompt=True):
        """Generate response from prompt."""
        prompt_text = self._build_prompt(prompt, use_system_prompt)
        return self._run_inference(prompt_text, max_tokens, temperature, top_p)

    @staticmethod
    def _score_response(text):
        """Score a candidate response for trolling quality.

        Higher = more personality. Rewards:
        - Length (longer = more engaged, arguing = writing more)
        - Reddit slang density (bro, tbh, ngl, imo, lmao, etc.)
        - Punctuation chaos (?, !, ...)
        - Lowercase commitment (all-lowercase = reddit native)
        Penalizes:
        - Empty or very short responses
        - Generic assistant phrases
        """
        if not text or len(text) < 5:
            return -100

        score = 0.0

        # Length bonus: longer responses = more engaged
        words = text.split()
        score += min(len(words), 80) * 0.5

        text_lower = text.lower()

        # Reddit slang density
        slang = ['bro', 'tbh', 'ngl', 'imo', 'lmao', 'lol', 'bruh',
                 'nah', 'fr', 'literally', 'actually', 'honestly',
                 'ok so', 'look', 'the thing is', 'imagine']
        for s in slang:
            score += text_lower.count(s) * 3

        # Punctuation chaos
        score += text.count('?') * 2
        score += text.count('!') * 1.5
        score += text.count('...') * 2

        # Lowercase commitment (ratio of lowercase to total alpha)
        alpha = sum(1 for c in text if c.isalpha())
        if alpha > 0:
            lower_ratio = sum(1 for c in text if c.islower()) / alpha
            if lower_ratio > 0.9:
                score += 5

        # Penalize generic assistant patterns
        boring = ['as an ai', 'i cannot', 'i apologize', 'how can i help',
                  'i\'d be happy to', 'great question']
        for b in boring:
            if b in text_lower:
                score -= 20

        return score

    def generate_troll(self, prompt, max_tokens=200, use_system_prompt=True):
        """Trolling mode: generate 3 candidates at different temps, pick best.

        Temperatures: 0.9, 1.0, 1.1 (NEVER below 0.9)
        Scores each candidate and returns the spiciest one.
        """
        prompt_text = self._build_prompt(prompt, use_system_prompt)
        temps = [0.9, 1.0, 1.1]
        candidates = []

        # Run all 3 in parallel (Go engine has mutex, so they serialize,
        # but ThreadPoolExecutor handles the waiting cleanly)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_inference, prompt_text, max_tokens, t): t
                for t in temps
            }
            for future in concurrent.futures.as_completed(futures):
                t = futures[future]
                try:
                    text, count = future.result()
                    score = self._score_response(text)
                    candidates.append((score, t, text, count))
                except Exception:
                    pass

        if not candidates:
            return "(trolling mode failed, all candidates died)", 0, ""

        # Sort by score descending, pick best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_temp, best_text, best_count = candidates[0]

        # Show which temp won
        temp_report = ' | '.join(
            f"t={c[1]:.1f}:{c[0]:.0f}{'*' if c is candidates[0] else ''}"
            for c in candidates
        )

        return best_text, best_count, temp_report

    def close(self):
        """Free engine resources."""
        self.lib.wtf_free()


def repl():
    """Interactive REPL."""
    print(f"\n{'='*60}")
    print(f"  WTFORACLE")
    print(f"  the reddit oracle nobody asked for")
    print(f"  {CONFIG['name']}")
    print(f"{'='*60}")
    print("Commands: /quit, /tokens N, /temp T, /raw, /troll")
    print()

    oracle = WTForacle()
    max_tokens = 200
    temperature = 0.9
    use_system_prompt = True
    trolling_mode = False

    while True:
        try:
            prompt = input("You: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['/quit', '/exit', '/q']:
                print("later loser")
                break

            if prompt.startswith('/tokens '):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Usage: /tokens N")
                continue

            if prompt.startswith('/temp '):
                try:
                    t = float(prompt.split()[1])
                    if t < 0.9:
                        print("nah bro, floor is 0.9. below that the assistant personality leaks through.")
                        t = 0.9
                    temperature = t
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Usage: /temp T")
                continue

            if prompt.lower() == '/raw':
                use_system_prompt = not use_system_prompt
                state = "OFF (raw mode)" if not use_system_prompt else "ON"
                print(f"System prompt: {state}")
                continue

            if prompt.lower() == '/troll':
                trolling_mode = not trolling_mode
                state = "ON (3 candidates, best wins)" if trolling_mode else "OFF"
                print(f"Trolling mode: {state}")
                continue

            print("\nWTForacle: ", end='', flush=True)
            if trolling_mode:
                response, count, temp_report = oracle.generate_troll(
                    prompt, max_tokens, use_system_prompt
                )
                print(response)
                print(f"  [{temp_report}]")
            else:
                response, count = oracle.generate(
                    prompt, max_tokens, temperature,
                    use_system_prompt=use_system_prompt
                )
                print(response)
            print()

        except KeyboardInterrupt:
            print("\nlater loser")
            break
        except EOFError:
            print("\nlater loser")
            break

    oracle.close()


if __name__ == '__main__':
    repl()

#!/usr/bin/env python3
"""
WTForacle REPL - Python wrapper for wtf.c inference engine.

Usage:
  cd wtf.c && make && cd ..
  python wtforacle.py
"""
import subprocess
import sys
import os
import pickle
import concurrent.futures

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# WTForacle config
CONFIG = {
    'weights': os.path.join(SCRIPT_DIR, 'wtfweights', 'wtforacle_q8.bin'),
    'tokenizer_pkl': os.path.join(SCRIPT_DIR, 'wtfweights', 'tokenizer.pkl'),
    'tokenizer_tok': os.path.join(SCRIPT_DIR, 'wtfweights', 'wtforacle.tok'),
    'name': 'WTForacle d20 (477M, INT8)',
}

# Identity anchor — injected as first turn before every prompt.
# Keeps the 477M on-character without a giant system prompt.
SYSTEM_PROMPT = (
    "you are wtforacle, a cynical reddit commenter. "
    "you love to argue for fun, and when you argue you write more. "
    "if you say something helpful, always end with sarcasm. "
    "a weird answer is always better than a boring one. "
    "talk like reddit: bro, tbh, ngl, imo. "
    "if you apologize, do it with sarcasm. "
    "if someone asks for code, give the code but call them lazy."
)
SYSTEM_ACK = "ok"


class WTForacle:
    def __init__(self):
        cfg = CONFIG
        self.weights = cfg['weights']
        self.tokenizer_tok = cfg['tokenizer_tok']
        self.name = cfg['name']
        self.wtf_bin = os.path.join(SCRIPT_DIR, 'wtf.c', 'wtf')

        # Load tokenizer for encoding
        with open(cfg['tokenizer_pkl'], 'rb') as f:
            self.enc = pickle.load(f)

        # Special tokens
        self.bos = self.enc._special_tokens['<|bos|>']
        self.user_start = self.enc._special_tokens['<|user_start|>']
        self.user_end = self.enc._special_tokens['<|user_end|>']
        self.assistant_start = self.enc._special_tokens['<|assistant_start|>']
        self.assistant_end = self.enc._special_tokens['<|assistant_end|>']

        # Check files exist
        if not os.path.exists(self.wtf_bin):
            raise FileNotFoundError(f"wtf binary not found: {self.wtf_bin}\nRun 'cd wtf.c && make' first.")
        if not os.path.exists(self.weights):
            raise FileNotFoundError(f"Weights not found: {self.weights}\nDownload from HuggingFace: https://huggingface.co/ataeff/WTForacle")

    def tokenize(self, text, use_system_prompt=True):
        """Encode user text into chat format tokens.

        With system prompt (default), the token sequence is:
          <bos> <user_start> {system_prompt} <user_end>
          <assistant_start> {system_ack} <assistant_end>
          <user_start> {user_text} <user_end> <assistant_start>

        This gives the model a one-turn identity anchor before
        the real question, at the cost of ~30 extra tokens.
        """
        tokens = [self.bos]

        if use_system_prompt:
            sys_tokens = self.enc.encode_ordinary(SYSTEM_PROMPT)
            ack_tokens = self.enc.encode_ordinary(SYSTEM_ACK)
            tokens += [self.user_start] + sys_tokens + [self.user_end]
            tokens += [self.assistant_start] + ack_tokens + [self.assistant_end]

        text_tokens = self.enc.encode_ordinary(text)
        tokens += [self.user_start] + text_tokens + [self.user_end, self.assistant_start]
        return tokens

    def _run_inference(self, token_str, max_tokens, temperature):
        """Run a single inference call and parse the output."""
        cmd = [
            self.wtf_bin,
            self.weights,
            self.tokenizer_tok,
            '-p', token_str,
            '-n', str(max_tokens),
            '-t', str(temperature),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        output = result.stdout
        lines = output.split('\n')

        in_generation = False
        response_lines = []
        for line in lines:
            if '--- Generation ---' in line:
                in_generation = True
                continue
            if in_generation:
                if line.startswith('---') and 'tokens' in line:
                    break
                response_lines.append(line)

        return '\n'.join(response_lines).strip(), result.stdout

    def generate(self, prompt, max_tokens=100, temperature=0.9, use_system_prompt=True):
        """Generate response from prompt."""
        tokens = self.tokenize(prompt, use_system_prompt=use_system_prompt)
        token_str = ' '.join(str(t) for t in tokens)
        return self._run_inference(token_str, max_tokens, temperature)

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

    def generate_troll(self, prompt, max_tokens=100, use_system_prompt=True):
        """Trolling mode: generate 3 candidates at different temps, pick best.

        Temperatures: 0.9, 1.0, 1.1 (NEVER below 0.9)
        Scores each candidate and returns the spiciest one.
        """
        tokens = self.tokenize(prompt, use_system_prompt=use_system_prompt)
        token_str = ' '.join(str(t) for t in tokens)

        temps = [0.9, 1.0, 1.1]
        candidates = []

        # Run all 3 in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_inference, token_str, max_tokens, t): t
                for t in temps
            }
            for future in concurrent.futures.as_completed(futures):
                t = futures[future]
                try:
                    text, raw = future.result()
                    score = self._score_response(text)
                    candidates.append((score, t, text, raw))
                except Exception:
                    pass

        if not candidates:
            return "(trolling mode failed, all candidates died)", ""

        # Sort by score descending, pick best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_temp, best_text, best_raw = candidates[0]

        # Show which temp won (for debugging/fun)
        temp_report = ' | '.join(
            f"t={c[1]:.1f}:{c[0]:.0f}{'*' if c is candidates[0] else ''}"
            for c in candidates
        )

        return best_text, best_raw, temp_report


def repl():
    """Interactive REPL."""
    print(f"\n{'='*60}")
    print(f"  WTFORACLE")
    print(f"  the reddit oracle nobody asked for")
    print(f"  {CONFIG['name']}")
    print(f"{'='*60}")
    print("Commands: /quit, /tokens N, /temp T, /raw, /troll (trolling mode)")
    print()

    oracle = WTForacle()
    max_tokens = 100
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
                except:
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
                except:
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
                result = oracle.generate_troll(prompt, max_tokens, use_system_prompt)
                response, raw, temp_report = result
                print(response)
                print(f"  [{temp_report}]")
            else:
                response, raw = oracle.generate(prompt, max_tokens, temperature, use_system_prompt)
                print(response)
            print()

        except KeyboardInterrupt:
            print("\nlater loser")
            break
        except EOFError:
            print("\nlater loser")
            break


if __name__ == '__main__':
    repl()

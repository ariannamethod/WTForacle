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
    "talk like reddit: bro, tbh, ngl, imo."
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

    def generate(self, prompt, max_tokens=100, temperature=0.9, use_system_prompt=True):
        """Generate response from prompt."""
        tokens = self.tokenize(prompt, use_system_prompt=use_system_prompt)
        token_str = ' '.join(str(t) for t in tokens)

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


def repl():
    """Interactive REPL."""
    print(f"\n{'='*60}")
    print(f"  WTFORACLE")
    print(f"  the reddit oracle nobody asked for")
    print(f"  {CONFIG['name']}")
    print(f"{'='*60}")
    print("Commands: /quit, /tokens N, /temp T, /raw (toggle system prompt)")
    print()

    oracle = WTForacle()
    max_tokens = 100
    temperature = 0.9
    use_system_prompt = True

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
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: /temp T")
                continue

            if prompt.lower() == '/raw':
                use_system_prompt = not use_system_prompt
                state = "OFF (raw mode)" if not use_system_prompt else "ON"
                print(f"System prompt: {state}")
                continue

            print("\nWTForacle: ", end='', flush=True)
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

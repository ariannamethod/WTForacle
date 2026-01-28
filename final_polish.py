#!/usr/bin/env python3
"""Final artistic polish - manual improvements and variety injection."""

import re
import random
from collections import defaultdict

EMOTICONS = [':)', ':P', 'XD', ':-/', ';)', '>:(', '¯\\_(ツ)_/¯', 'ಠ_ಠ']

# Variations for "we covered this" - should be more creative
WE_COVERED_VARIATIONS = [
    "we've been here before.",
    "déjà vu time.",
    "this again?",
    "oh this question.",
    "ancient repost energy.",
    "back to this one.",
    "circling back i see.",
    "ah yes, the classic.",
    "timeless question.",
    "eternal return of the same question.",
]

# Variations for "sir this is"
SIR_THIS_IS_VARIATIONS = [
    "sir this is",
    "bro this is",
    "ma this is",
    "bestie this is",
    "friend this is",
]

def parse_dataset(filepath):
    """Parse Q&A pairs from dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    pairs = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.lower().startswith('q: '):
            question = line[3:]
            i += 1
            answer_lines = []
            while i < len(lines) and not lines[i].strip().lower().startswith('q: '):
                answer_line = lines[i].strip()
                if answer_line.lower().startswith('a: '):
                    answer_lines.append(answer_line[3:])
                elif answer_line:
                    answer_lines.append(answer_line)
                i += 1
            answer = '\n'.join(answer_lines)
            if answer:
                pairs.append((question, answer))
        else:
            i += 1

    return pairs

def vary_we_covered_this(answer):
    """Replace 'we covered this' with variations."""
    if answer.lower().startswith('we covered this'):
        # Replace with variation
        variation = random.choice(WE_COVERED_VARIATIONS)
        rest = answer[15:]  # Remove "we covered this"
        return variation + rest
    return answer

def vary_sir_this_is(answer):
    """Vary the 'sir this is' pattern."""
    if answer.lower().startswith('sir this is'):
        variation = random.choice(SIR_THIS_IS_VARIATIONS)
        rest = answer[11:]  # Remove "sir this is"
        return variation + rest
    return answer

def add_more_emoticons(answer):
    """Add emoticons more liberally where they fit."""
    # Skip if already has emoticon
    if any(e in answer for e in EMOTICONS):
        return answer

    # 30% chance to add emoticon
    if random.random() > 0.70:
        return answer

    lines = answer.split('\n')

    # Add to last line if it ends with period
    if lines[-1].endswith('.'):
        emoticon = random.choice(EMOTICONS)
        lines[-1] += f' {emoticon}'

    return '\n'.join(lines)

def improve_boring_patterns(answer):
    """Fix remaining boring patterns."""

    # Pattern: "X. Y. Z." with no connectors - add variety
    lines = answer.split('\n')
    if len(lines) >= 4:
        # Check if it's all short declarative statements
        short_lines = [l for l in lines if len(l.split()) <= 4]
        if len(short_lines) >= 3:
            # This is a list-style answer, add some flow
            connectors = ['also', 'plus', 'btw', 'oh and', 'and']
            for i in range(1, min(len(lines), 4)):
                if random.random() > 0.5:
                    lines[i] = f"{random.choice(connectors)} {lines[i]}"

            return '\n'.join(lines)

    return answer

def inject_slang_naturally(answer):
    """Add more slang to keep voice consistent."""

    # Skip if already has plenty of slang
    slang_words = ['bro', 'ngl', 'fr', 'lowkey', 'bruh', 'lmao', 'tbh', 'deadass', 'no cap', 'ong']
    has_slang = any(word in answer.lower() for word in slang_words)

    if has_slang:
        return answer

    # Add slang intro to some answers
    if random.random() > 0.90:  # 10% get slang injection
        intros = ['ngl', 'tbh', 'lowkey', 'fr fr']
        intro = random.choice(intros)
        # Add to beginning
        return f"{intro} {answer}"

    return answer

def final_polish(input_path, output_path):
    """Final artistic polish."""
    print("Reading dataset for final polish...")
    pairs = parse_dataset(input_path)
    print(f"Loaded {len(pairs)} Q&A pairs")

    # Apply improvements
    improved_patterns = 0
    added_emoticons = 0
    added_slang = 0

    for i in range(len(pairs)):
        q, a = pairs[i]
        original_a = a

        # Apply transformations
        a = vary_we_covered_this(a)
        a = vary_sir_this_is(a)
        a = improve_boring_patterns(a)
        a = inject_slang_naturally(a)
        a = add_more_emoticons(a)

        if a != original_a:
            improved_patterns += 1

        # Track emoticon additions
        if any(e in a for e in EMOTICONS) and not any(e in original_a for e in EMOTICONS):
            added_emoticons += 1

        pairs[i] = (q, a)

    print(f"\nFinal improvements:")
    print(f"  Patterns improved: {improved_patterns}")
    print(f"  Emoticons added: {added_emoticons}")

    # Final stats
    emoticon_count = sum(1 for q, a in pairs if any(e in a for e in EMOTICONS))
    wtf_mentions = sum(1 for q, a in pairs if 'wtforacle' in a.lower())

    print(f"\nFinal dataset stats:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Emoticons: {emoticon_count} ({100*emoticon_count/len(pairs):.1f}%)")
    print(f"  WTForacle mentions: {wtf_mentions} ({100*wtf_mentions/len(pairs):.1f}%)")

    # Write output
    print(f"\nWriting final polished dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q, a in pairs:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")

    print("\n✨ DONE! Dataset artistically polished. ✨")

def main():
    input_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_polished_dataset.txt'
    output_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_polished_dataset.txt'
    final_polish(input_path, output_path)

if __name__ == '__main__':
    random.seed(42)
    main()

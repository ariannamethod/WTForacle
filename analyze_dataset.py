#!/usr/bin/env python3
"""Analyze WTForacle dataset for patterns and issues."""

import re
from collections import Counter, defaultdict

def parse_dataset(filepath):
    """Parse Q&A pairs from dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by Q: pattern
    pairs = []
    lines = content.strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Q: '):
            question = line[3:]
            i += 1
            # Collect answer lines until next Q: or end
            answer_lines = []
            while i < len(lines) and not lines[i].strip().startswith('Q: '):
                answer_line = lines[i].strip()
                if answer_line.startswith('A: '):
                    answer_lines.append(answer_line[3:])
                elif answer_line:  # continuation of answer
                    answer_lines.append(answer_line)
                i += 1
            answer = '\n'.join(answer_lines)
            if answer:
                pairs.append((question, answer))
        else:
            i += 1

    return pairs

def analyze_patterns(pairs):
    """Find repetitive patterns in answers."""

    # Opening phrases (first 5 words)
    openings = []
    # Closing phrases (last 5 words)
    closings = []
    # Common phrases (3+ words)
    phrases = []
    # Questions vs statements
    question_answers = 0

    for q, a in pairs:
        words = a.split()
        if len(words) >= 5:
            openings.append(' '.join(words[:5]))
            closings.append(' '.join(words[-5:]))

        # Count question marks in answer
        if '?' in a:
            question_answers += 1

        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrases.append(' '.join(words[i:i+3]))

    print("=" * 60)
    print("OPENING PHRASES (most common):")
    print("=" * 60)
    for phrase, count in Counter(openings).most_common(20):
        if count > 5:
            print(f"{count:3d}x  {phrase}")

    print("\n" + "=" * 60)
    print("CLOSING PHRASES (most common):")
    print("=" * 60)
    for phrase, count in Counter(closings).most_common(20):
        if count > 5:
            print(f"{count:3d}x  {phrase}")

    print("\n" + "=" * 60)
    print("REPEATED 3-WORD PHRASES:")
    print("=" * 60)
    for phrase, count in Counter(phrases).most_common(30):
        if count > 10:
            print(f"{count:3d}x  {phrase}")

    print(f"\n{question_answers} answers contain questions")

def find_duplicates(pairs):
    """Find duplicate questions and answers."""

    q_counter = Counter(q for q, a in pairs)
    a_counter = Counter(a for q, a in pairs)

    print("\n" + "=" * 60)
    print("DUPLICATE QUESTIONS:")
    print("=" * 60)
    dupe_qs = [(q, c) for q, c in q_counter.most_common() if c > 1]
    print(f"Found {len(dupe_qs)} duplicate questions")
    for q, count in dupe_qs[:10]:
        print(f"{count}x  {q[:60]}...")

    print("\n" + "=" * 60)
    print("DUPLICATE ANSWERS:")
    print("=" * 60)
    dupe_as = [(a, c) for a, c in a_counter.most_common() if c > 1]
    print(f"Found {len(dupe_as)} duplicate answers")
    for a, count in dupe_as[:10]:
        print(f"{count}x  {a[:60]}...")

def analyze_length(pairs):
    """Analyze answer lengths."""
    lengths = [len(a.split()) for q, a in pairs]

    print("\n" + "=" * 60)
    print("ANSWER LENGTH STATS:")
    print("=" * 60)
    print(f"Total Q&A pairs: {len(pairs)}")
    print(f"Avg length: {sum(lengths)/len(lengths):.1f} words")
    print(f"Min: {min(lengths)} words")
    print(f"Max: {max(lengths)} words")

    # Find very short (< 5 words) and very long (> 50 words)
    short = [(q, a) for q, a in pairs if len(a.split()) < 5]
    long = [(q, a) for q, a in pairs if len(a.split()) > 50]

    print(f"\nVery short answers (< 5 words): {len(short)}")
    print("Examples:")
    for q, a in short[:5]:
        print(f"  Q: {q[:50]}")
        print(f"  A: {a}")

    print(f"\nVery long answers (> 50 words): {len(long)}")
    print("Examples:")
    for q, a in long[:3]:
        print(f"  Q: {q[:50]}")
        print(f"  A: {a[:100]}...")

def check_wtforacle_mentions(pairs):
    """Count self-references."""
    mentions = 0
    for q, a in pairs:
        if 'wtforacle' in a.lower():
            mentions += 1

    print("\n" + "=" * 60)
    print("WTFORACLE MENTIONS:")
    print("=" * 60)
    print(f"{mentions} answers mention 'wtforacle' ({100*mentions/len(pairs):.1f}%)")

def main():
    filepath = '/Users/ataeff/Downloads/WTForacle/wtforacle_clean_dataset.txt'
    pairs = parse_dataset(filepath)

    analyze_patterns(pairs)
    find_duplicates(pairs)
    analyze_length(pairs)
    check_wtforacle_mentions(pairs)

if __name__ == '__main__':
    main()

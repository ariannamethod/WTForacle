#!/usr/bin/env python3
"""Enhanced polishing - fix repetitive patterns, improve variety, add emoticons."""

import re
import random
from collections import Counter

# Emoticon pool
EMOTICONS = [':)', ':P', 'XD', ':-/', ';)', '>:(', '¯\\_(ツ)_/¯', 'ಠ_ಠ']

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

def fix_also_spam(answer):
    """Fix repetitive 'also X. also Y. also Z.' patterns."""
    lines = answer.split('\n')
    also_count = sum(1 for line in lines if line.strip().startswith('also '))

    # If more than 3 'also' lines, rewrite
    if also_count > 3:
        also_lines = [line.strip()[5:] for line in lines if line.strip().startswith('also ')]
        non_also_lines = [line for line in lines if not line.strip().startswith('also ')]

        # Rewrite with variety
        if len(also_lines) >= 4:
            # Combine into more natural flow
            new_answer = non_also_lines[0] if non_also_lines else also_lines[0]
            # Add rest with connectors
            connectors = ['also', 'plus', 'and', 'oh and', 'bonus:', 'btw']
            for i, item in enumerate(also_lines[:5]):  # Limit to 5
                if i == 0:
                    new_answer += f"\n{item}."
                else:
                    connector = connectors[min(i-1, len(connectors)-1)]
                    new_answer += f"\n{connector} {item}."

            return new_answer

    return answer

def add_emoticons_naturally(answer):
    """Add emoticons where they fit naturally."""
    # Don't add to every answer
    if random.random() > 0.25:  # Only 25% get emoticons
        return answer

    # Find good insertion points
    sentences = re.split(r'([.!?])', answer)

    # Add emoticon after certain patterns
    for i in range(len(sentences)-1):
        sent = sentences[i].lower()

        # After funny/sarcastic statements
        if any(word in sent for word in ['bro', 'lmao', 'tragic', 'congrats', 'sure', 'totally']):
            if random.random() > 0.5:
                emoticon = random.choice([':)', ':P', 'XD', ':-/', '¯\\_(ツ)_/¯'])
                sentences[i] += f' {emoticon}'
                break

        # After annoyed statements
        if any(word in sent for word in ['wtf', 'seriously', 'again', 'stop']):
            if random.random() > 0.5:
                emoticon = random.choice([':-/', '>:(', 'ಠ_ಠ'])
                sentences[i] += f' {emoticon}'
                break

    return ''.join(sentences)

def improve_answer_variety(q, a):
    """Make answers more varied and interesting."""

    # Fix "also" spam
    a = fix_also_spam(a)

    # Sometimes add question responses
    if random.random() > 0.95 and '?' not in a:
        question_intros = [
            "question: ",
            "real question: ",
            "better question: ",
            "actually: ",
        ]
        # Add question intro to first sentence sometimes
        a = random.choice(question_intros) + a

    # Add emoticons naturally
    a = add_emoticons_naturally(a)

    return a

def enhance_weak_short_answers(q, a):
    """Improve answers that are too short/weak."""
    word_count = len(a.split())

    # Very short answers (under 5 words) - beef them up
    if word_count < 5 and word_count > 2:
        spicy_additions = [
            " deadass.",
            " fr fr.",
            " no cap.",
            " ngl.",
            " ong.",
            " :P",
            " lmao",
        ]
        return a + random.choice(spicy_additions)

    return a

def ensure_voice_consistency(pairs):
    """Make sure the WTForacle voice is consistent throughout."""

    improved = 0
    for i in range(len(pairs)):
        q, a = pairs[i]

        # Check if answer is too formal/polite
        if any(word in a.lower() for word in ['please note', 'i would recommend', 'it is important']):
            # This doesn't match voice, needs rewrite
            # For now, just flag it
            pass

        # Improve variety
        new_a = improve_answer_variety(q, a)
        if new_a != a:
            improved += 1
            pairs[i] = (q, new_a)

        # Enhance weak answers
        new_a = enhance_weak_short_answers(q, pairs[i][1])
        if new_a != pairs[i][1]:
            pairs[i] = (q, new_a)

    return pairs, improved

def find_repetitive_openings(pairs):
    """Find and report most common answer openings."""
    openings = []
    for q, a in pairs:
        first_words = ' '.join(a.split()[:3])
        if first_words:
            openings.append(first_words.lower())

    common = Counter(openings).most_common(30)
    print("\nMost common answer openings:")
    for opening, count in common:
        if count > 10:
            print(f"  {count:3d}x  {opening}")

def polish_v2(input_path, output_path):
    """Enhanced polishing pass."""
    print("Reading polished dataset...")
    pairs = parse_dataset(input_path)
    print(f"Loaded {len(pairs)} Q&A pairs")

    # Analyze patterns
    find_repetitive_openings(pairs)

    # Improve variety and consistency
    print("\nEnhancing variety and voice consistency...")
    pairs, improved = ensure_voice_consistency(pairs)
    print(f"Improved {improved} answers")

    # Count emoticons
    emoticon_count = sum(1 for q, a in pairs if any(e in a for e in EMOTICONS))
    print(f"Answers with emoticons: {emoticon_count} ({100*emoticon_count/len(pairs):.1f}%)")

    # Write output
    print(f"\nWriting enhanced dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q, a in pairs:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")

    print("\nDone! Enhanced polishing complete.")

def main():
    input_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_polished_dataset.txt'
    output_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_polished_dataset.txt'
    polish_v2(input_path, output_path)

if __name__ == '__main__':
    random.seed(42)
    main()

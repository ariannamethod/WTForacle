#!/usr/bin/env python3
"""Polish the WTForacle dataset - remove dupes, vary phrases, improve weak answers."""

import re
import random
from collections import defaultdict

# Variation templates for overused phrases
PHRASE_VARIATIONS = {
    "we covered this. still": [
        "been there, done that. still",
        "already went through this. still",
        "touched on this already. still",
        "old news. but still",
        "déjà vu time. still",
        "this again? fine. still",
    ],
    "if you like": [
        "if ur into",
        "if you vibe with",
        "if that's ur thing",
        "if you're down for",
        "assuming you don't hate",
    ],
    "if you want": [
        "if ur trying to",
        "if that's the goal",
        "if you're going for",
        "assuming you actually want",
        "if you really need",
    ],
}

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
            # Collect answer lines
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

def deduplicate(pairs):
    """Remove duplicate Q&A pairs, keeping better answers."""
    seen_questions = {}
    unique_pairs = []
    duplicates_removed = 0

    for q, a in pairs:
        q_lower = q.lower().strip()

        if q_lower in seen_questions:
            # Compare answer quality
            existing_answer = seen_questions[q_lower][1]

            # Prefer longer, more detailed answers
            if len(a) > len(existing_answer):
                # Replace with better answer
                seen_questions[q_lower] = (q, a)
            duplicates_removed += 1
        else:
            seen_questions[q_lower] = (q, a)

    # Convert back to list
    unique_pairs = list(seen_questions.values())
    return unique_pairs, duplicates_removed

def fix_arch_spam(answer):
    """Fix the absurd 'btw i use Arch' spam."""
    lines = answer.split('\n')

    # Count consecutive "btw i use Arch" lines
    arch_count = 0
    for line in lines:
        if line.strip() == "btw i use Arch.":
            arch_count += 1

    # If more than 5 repetitions, trim to 3
    if arch_count > 5:
        new_lines = []
        arch_seen = 0
        for line in lines:
            if line.strip() == "btw i use Arch.":
                if arch_seen < 3:
                    new_lines.append(line)
                    arch_seen += 1
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)

    return answer

def vary_phrases(answer):
    """Replace repetitive phrases with variations."""
    for phrase, variations in PHRASE_VARIATIONS.items():
        if phrase.lower() in answer.lower():
            # Replace with a random variation
            variation = random.choice(variations)
            answer = re.sub(re.escape(phrase), variation, answer, count=1, flags=re.IGNORECASE)

    return answer

def improve_weak_answer(q, a):
    """Improve answers that are too weak or generic."""

    word_count = len(a.split())

    # Too short and boring
    if word_count < 4 and not any(punct in a for punct in ['?', '!', '.']):
        # Add some flavor
        spicy_endings = [
            ". ngl.",
            ". fr fr.",
            ". deadass.",
            ". no cap.",
            ". ong.",
        ]
        return a + random.choice(spicy_endings)

    return a

def add_wtforacle_mentions(pairs):
    """Add more self-references to hit 15-20% target."""
    current_mentions = sum(1 for q, a in pairs if 'wtforacle' in a.lower())
    target = int(len(pairs) * 0.17)  # 17% target
    need_more = target - current_mentions

    if need_more <= 0:
        return pairs

    # Find answers that could naturally mention wtforacle
    candidates = []
    for i, (q, a) in enumerate(pairs):
        if 'wtforacle' not in a.lower():
            # Look for meta questions about identity, opinions, etc
            if any(word in q.lower() for word in ['you', 'your', 'who', 'what do you', 'opinion']):
                candidates.append(i)

    # Randomly pick some to inject mentions
    random.shuffle(candidates)
    injection_templates = [
        lambda a: f"wtforacle take: {a}",
        lambda a: f"{a}\n(wtforacle certified opinion btw)",
        lambda a: f"look, wtforacle doesn't lie. {a}",
        lambda a: f"{a}\nim wtforacle btw. this matters.",
    ]

    injected = 0
    for idx in candidates[:need_more]:
        q, a = pairs[idx]
        template = random.choice(injection_templates)
        pairs[idx] = (q, template(a))
        injected += 1

    return pairs

def polish_dataset(input_path, output_path):
    """Main polishing logic."""
    print("Reading dataset...")
    pairs = parse_dataset(input_path)
    print(f"Loaded {len(pairs)} Q&A pairs")

    # Step 1: Deduplicate
    print("\nRemoving duplicates...")
    pairs, dupes_removed = deduplicate(pairs)
    print(f"Removed {dupes_removed} duplicate questions")
    print(f"Remaining: {len(pairs)} pairs")

    # Step 2: Fix known issues (Arch spam)
    print("\nFixing known issues...")
    pairs = [(q, fix_arch_spam(a)) for q, a in pairs]

    # Step 3: Vary repetitive phrases
    print("Varying repetitive phrases...")
    improved = 0
    for i in range(len(pairs)):
        q, a = pairs[i]
        new_a = vary_phrases(a)
        if new_a != a:
            improved += 1
        pairs[i] = (q, new_a)
    print(f"Improved {improved} answers with phrase variations")

    # Step 4: Improve weak answers
    print("Improving weak answers...")
    weak_fixed = 0
    for i in range(len(pairs)):
        q, a = pairs[i]
        new_a = improve_weak_answer(q, a)
        if new_a != a:
            weak_fixed += 1
        pairs[i] = (q, new_a)
    print(f"Fixed {weak_fixed} weak answers")

    # Step 5: Add wtforacle mentions
    print("\nAdding wtforacle mentions...")
    before_mentions = sum(1 for q, a in pairs if 'wtforacle' in a.lower())
    pairs = add_wtforacle_mentions(pairs)
    after_mentions = sum(1 for q, a in pairs if 'wtforacle' in a.lower())
    print(f"Mentions: {before_mentions} -> {after_mentions} ({100*after_mentions/len(pairs):.1f}%)")

    # Write output
    print(f"\nWriting polished dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q, a in pairs:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")

    print("\nDone!")
    print(f"Final stats:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Duplicates removed: {dupes_removed}")
    print(f"  Phrase variations: {improved}")
    print(f"  Weak answers fixed: {weak_fixed}")
    print(f"  WTForacle mentions: {after_mentions} ({100*after_mentions/len(pairs):.1f}%)")

def main():
    input_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_clean_dataset.txt'
    output_path = '/Users/ataeff/Downloads/WTForacle/wtforacle_polished_dataset.txt'
    polish_dataset(input_path, output_path)

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()

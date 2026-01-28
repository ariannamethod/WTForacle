"""Convert q:/a: format WTForacle dataset to nanochat JSONL format."""
import json
import re

def convert_wtf_to_jsonl(input_path, output_path):
    conversations = []
    current_q = None
    current_a_lines = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Match q: or Q: (case insensitive)
            if line.lower().startswith('q: ') or line.lower().startswith('q:'):
                # Save previous pair
                if current_q and current_a_lines:
                    a_text = ' '.join(current_a_lines)
                    conversations.append([
                        {"role": "user", "content": current_q},
                        {"role": "assistant", "content": a_text}
                    ])
                # Get question (remove q: prefix)
                current_q = re.sub(r'^[qQ]:\s*', '', line)
                current_a_lines = []

            # Match a: or A:
            elif line.lower().startswith('a: ') or line.lower().startswith('a:'):
                answer = re.sub(r'^[aA]:\s*', '', line)
                current_a_lines.append(answer)

            elif line.strip() and current_a_lines:
                # Continuation of answer (multi-line answers)
                current_a_lines.append(line.strip())

        # Last pair
        if current_q and current_a_lines:
            a_text = ' '.join(current_a_lines)
            conversations.append([
                {"role": "user", "content": current_q},
                {"role": "assistant", "content": a_text}
            ])

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"Converted {len(conversations)} conversations")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    return len(conversations)

if __name__ == "__main__":
    count = convert_wtf_to_jsonl(
        "/Users/ataeff/Downloads/WTForacle/wtforacle_final_dataset.txt",
        "/Users/ataeff/Downloads/WTForacle/wtforacle_identity.jsonl"
    )
    print(f"\nReady for Lambda training with {count} identity conversations!")

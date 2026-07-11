import json

INPUT_FILE = "synthetic_greek_legal_dataset.jsonl"
OUTPUT_FILE = "chatml_legal_dataset.jsonl"
SYSTEM_PROMPT = "Είσαι ένας κορυφαίος νομικός σύμβουλος εξειδικευμένος στην Ελληνική Νομοθεσία. Απάντησε με ακρίβεια."

print(f"Ξεκινάει η μετατροπή του {INPUT_FILE} σε ChatML...")

converted_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if not line.strip():
            continue
            
        data = json.loads(line)
        
        # Φτιάχνουμε τη δομή ChatML (ShareGPT style)
        chatml_format = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["instruction"]},
                {"role": "assistant", "content": data["output"]}
            ]
        }
        
        f_out.write(json.dumps(chatml_format, ensure_ascii=False) + '\n')
        converted_count += 1

print(f"Επιτυχία! Μετατράπηκαν {converted_count} ερωταπαντήσεις. Το νέο αρχείο είναι το: {OUTPUT_FILE}")

import pandas as pd
import json
import requests
import logging
import threading
import time
import os

# --- Settings ---
START_FROM_SCRATCH = False 
TARGET_FOLDER = "greek-legal-code/subject"  # Εστιάζουμε ΜΟΝΟ στο subject για αποφυγή duplicates

# --- Logging & Monitoring ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

file_lock = threading.Lock()
OUTPUT_FILE = "synthetic_greek_legal_dataset.jsonl"
CHECKPOINT_FILE = "processed_docs.log"
FAILED_FILE = "failed_docs.log"

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def save_checkpoint(split_name, idx):
    with file_lock:
        with open(CHECKPOINT_FILE, 'a') as f:
            f.write(f"{split_name}_{idx}\n")

def save_failed(split_name, idx):
    with file_lock:
        with open(FAILED_FILE, 'a') as f:
            f.write(f"{split_name}_{idx}\n")

def generate_qa_locally(legal_text, max_retries=5):
    # Dynamic Slices εκμεταλλευόμενοι το 32K Context Window!
    slices = [30000, 25000, 20000, 15000, 10000]
    
    for attempt in range(max_retries):
        current_slice = slices[min(attempt, len(slices)-1)]
        truncated_text = legal_text[:current_slice]

        prompt = f"""Είσαι ένας ανώτατος νομικός σύμβουλος εξειδικευμένος στην Ελληνική Νομοθεσία. 
Ανάλυσε το παρακάτω νομικό κείμενο.
Δημιούργησε ακριβώς 3 υψηλής ποιότητας ερωταπαντήσεις (Q&A) κατάλληλες για fine-tuning.

ΚΑΝΟΝΕΣ (ΑΠΑΡΕΓΚΛΙΤΟΙ):
1. ΕΞΕΙΔΙΚΕΥΣΗ: Οι ερωτήσεις πρέπει να είναι συγκεκριμένες και να ενσωματώνουν απαραιτήτως τον τύπο του νομοθετήματος (π.χ. Νόμος, Π.Δ., Α.Ν.) και τον αριθμό του άρθρου/νόμου.
2. ΑΚΡΙΒΕΙΑ: Οι απαντήσεις πρέπει να βασίζονται αποκλειστικά στο παρεχόμενο κείμενο.
3. ΕΙΣΑΓΩΓΙΚΑ: Μην χρησιμοποιείς ΠΟΤΕ διπλά εισαγωγικά (") μέσα στο κείμενο των ερωτήσεων ή των απαντήσεων. Χρησιμοποίησε ΜΟΝΟ μονά εισαγωγικά (').

Νομικό Κείμενο:
{truncated_text}

Απάντησε ΑΥΣΤΗΡΑ και ΜΟΝΟ με ένα έγκυρο JSON αντικείμενο στη δομή:
{{"qa_pairs": [{{"instruction": "Ερώτηση...", "output": "Απάντηση..."}}]}}"""

        payload = {
            "model": "qwen3.6-mtp", 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  
            "max_tokens": 3000,  # Άνετος χώρος για τις απαντήσεις
            "response_format": {"type": "json_object"} # Εξαναγκασμός JSON από τον server
        }
        
        server_ip = "100.126.179.69"
        server_port = "8080"
        api_endpoint = f"http://{server_ip}:{server_port}/v1/chat/completions"
        
        try:
            response = requests.post(api_endpoint, json=payload, timeout=300)
            
            if response.status_code == 200:
                resp_data = response.json()
                result = resp_data['choices'][0]['message']['content'].strip()
                
                clean_result = result.replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(clean_result)
                return parsed_json.get("qa_pairs", [])
            else:
                logging.warning(f"Attempt {attempt + 1}/{max_retries}: Server status {response.status_code}. Retrying with smaller slice...")
                time.sleep(2)
                
        except json.JSONDecodeError as je:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: JSON parse error: {je}. Retrying...")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Network error: {e}. Retrying...")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries}: Unexpected error: {e}")
            break
            
    return []

def process_row(split_name, idx, row):
    text = row['text']
    qa_pairs = generate_qa_locally(text)
    
    if qa_pairs:
        with file_lock:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for pair in qa_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    f.flush()
    else:
        save_failed(split_name, idx)
                    
    save_checkpoint(split_name, idx)
    return idx, len(qa_pairs)

def main():
    if START_FROM_SCRATCH:
        logging.info("START_FROM_SCRATCH is True. Deleting old logs and datasets...")
        for file_to_delete in [OUTPUT_FILE, CHECKPOINT_FILE, FAILED_FILE]:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                logging.info(f"Deleted {file_to_delete}")

    processed_set = load_checkpoint()
    
    # Ορίζουμε ρητά τα 3 splits του φακέλου subject
    splits = ["train", "validation", "test"]
    
    for split in splits:
        file_path = f"{TARGET_FOLDER}/{split}-00000-of-00001.parquet"
        
        if not os.path.exists(file_path):
            logging.error(f"Το αρχείο {file_path} δεν βρέθηκε. Παράλειψη.")
            continue
            
        logging.info(f"📦 Φόρτωση αρχείου: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Φιλτράρισμα βάσει του σύνθετου κλειδιού (π.χ. train_0) για αποφυγή index collision
        df_to_process = df[~df.index.map(lambda idx: f"{split}_{idx}").isin(processed_set)]
        
        logging.info(f"🚀 Έναρξη παραγωγής για το split '{split}'. Απομένουν {len(df_to_process)} από {len(df)} εγγραφές.")
        
        if df_to_process.empty:
            logging.info(f"Το split '{split}' είναι ήδη πλήρως επεξεργασμένο.")
            continue

        total_pairs = 0
        
        # 4 Workers (συνιστάται για parallel 4 του llama-server)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, split, idx, row) for idx, row in df_to_process.iterrows()]
            
            for i, future in enumerate(as_completed(futures), 1):
                doc_idx, generated_count = future.result()  
                total_pairs += generated_count
                if generated_count > 0 or i % 10 == 0:
                    logging.info(f"[{split.upper()}] Πρόοδος: {i}/{len(df_to_process)} (Doc ID: {doc_idx}) | Παράχθηκαν {generated_count} Q&As | Σύνολο split: {total_pairs}")

if __name__ == "__main__":
    main()

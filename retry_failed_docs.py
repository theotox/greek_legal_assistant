import pandas as pd
import json
import requests
import logging
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Logging & Monitoring ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retry_pipeline.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

file_lock = threading.Lock()
OUTPUT_FILE = "synthetic_greek_legal_dataset.jsonl"
CHECKPOINT_FILE = "processed_docs.log"

# --- File Pointers ---
ORIGINAL_FAILED_FILE = "failed_docs.log"
STILL_FAILED_FILE = "still_failed_docs.log"
TARGET_FOLDER = "greek-legal-code/subject"

def load_original_failed_docs():
    """
    Διαβάζει το failed_docs.log και ομαδοποιεί τα IDs ανά split.
    Επιστρέφει π.χ.: {'train': [15, 23], 'validation': [4], 'test': []}
    """
    failed_dict = {'train': [], 'validation': [], 'test': []}
    if not os.path.exists(ORIGINAL_FAILED_FILE):
        return failed_dict
        
    with open(ORIGINAL_FAILED_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if '_' in line:
                split_name, idx_str = line.split('_')
                if split_name in failed_dict and idx_str.isdigit():
                    failed_dict[split_name].append(int(idx_str))
                    
    return failed_dict

def save_checkpoint(split_name, idx):
    with file_lock:
        with open(CHECKPOINT_FILE, 'a') as f:
            f.write(f"{split_name}_{idx}\n")

def save_still_failed(split_name, idx):
    with file_lock:
        with open(STILL_FAILED_FILE, 'a') as f:
            f.write(f"{split_name}_{idx}\n")

def generate_qa_locally(legal_text, max_retries=5):
    # Dynamic slicing για το 32K Context Window
    slices = [30000, 25000, 20000, 15000, 10000]
    
    for attempt in range(max_retries):
        current_slice = slices[min(attempt, len(slices)-1)]
        truncated_text = legal_text[:current_slice]
        
        prompt = f"""Είσαι ένας ανώτατος νομικός σύμβουλος εξειδικευμένος στην Ελληνική Νομοθεσία. 
Ανάλυσε το παρακάτω νομικό κείμενο.
Δημιούργησε ακριβώς 3 υψηλής ποιότητας ερωταπαντήσεις (Q&A) κατάλληλες για fine-tuning.

ΚΑΝΟΝΕΣ (ΑΠΑΡΕΓΚΛΙΤΟΙ):
1. ΕΞΕΙΔΙΚΕΥΣΗ: Οι ερωτήσεις πρέπει να είναι συγκεκριμένες και να ενσωματώνουν απαραιτήτως τον τύπο του νομοθετήματος και τον αριθμό του άρθρου/νόμου.
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
            "max_tokens": 3000,  
            "response_format": {"type": "json_object"} 
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
                logging.warning(f"Attempt {attempt + 1}/{max_retries}: Server returned status {response.status_code}. Retrying with smaller text slice...")
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
        save_checkpoint(split_name, idx)
        return idx, len(qa_pairs), True  
    else:
        save_still_failed(split_name, idx)
        return idx, 0, False  

def main():
    failed_dict = load_original_failed_docs()
    
    total_failed = sum(len(indices) for indices in failed_dict.values())
    if total_failed == 0:
        logging.info(f"Δεν βρέθηκαν αποτυχημένα έγγραφα στο {ORIGINAL_FAILED_FILE}. Τερματισμός.")
        return
        
    logging.info(f"Βρέθηκαν συνολικά {total_failed} αποτυχημένα έγγραφα στο {ORIGINAL_FAILED_FILE}.")
    
    # Καθαρισμός του αρχείου που θα δεχτεί όσα αποτύχουν και δεύτερη φορά
    if os.path.exists(STILL_FAILED_FILE):
        os.remove(STILL_FAILED_FILE)
        
    total_recovered = 0
    total_qas_generated = 0
        
    for split_name, failed_indices in failed_dict.items():
        if not failed_indices:
            continue
            
        file_path = f"{TARGET_FOLDER}/{split_name}-00000-of-00001.parquet"
        if not os.path.exists(file_path):
            logging.error(f"Αδυναμία φόρτωσης {file_path}. Παράλειψη αυτού του split.")
            continue
            
        logging.info(f"📦 Φόρτωση του Parquet '{split_name}' για ανάκτηση {len(failed_indices)} εγγράφων...")
        df = pd.read_parquet(file_path)
        
        # Φιλτράρισμα του dataframe για να κρατήσουμε μόνο τα failed IDs
        df_to_process = df[df.index.isin(failed_indices)]
        
        # Επεξεργασία με 4 Workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, split_name, idx, row) for idx, row in df_to_process.iterrows()]
            
            for i, future in enumerate(as_completed(futures), 1):
                doc_idx, generated_count, success = future.result()  
                if success:
                    total_recovered += 1
                    total_qas_generated += generated_count
                
                logging.info(f"[{split_name.upper()}] Πρόοδος: {i}/{len(df_to_process)} (Doc ID: {doc_idx}) | Ανακτήθηκαν: {total_recovered} έγγραφα συνολικά | Σύνολο Q&As: {total_qas_generated}")
                
    logging.info(f"✅ Ολοκληρώθηκε! Ανακτήθηκαν {total_recovered} από τα {total_failed} έγγραφα. Τα υπόλοιπα καταγράφηκαν στο {STILL_FAILED_FILE}.")

if __name__ == "__main__":
    main()

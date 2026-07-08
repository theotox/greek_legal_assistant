import pandas as pd
import json
import os
import glob

def convert_all_dataset_parquet_files():
    base_dir = "greek-legal-code"
    
    # Χρησιμοποιούμε τη glob για να βρούμε όλα τα .parquet αρχεία σε όλους τους υποφακέλους
    parquet_files = glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        print(f"❌ Δεν βρέθηκαν αρχεία Parquet στον φάκελο '{base_dir}'.")
        return

    print(f"🔍 Βρέθηκαν {len(parquet_files)} αρχεία Parquet για μετατροπή.\n")

    for parquet_path in parquet_files:
        # Δημιουργούμε το όνομα του αρχείου εξόδου (αλλάζουμε απλά την κατάληξη σε .json)
        json_path = parquet_path.replace(".parquet", "_human_readable.json")
        
        print(f"📦 Επεξεργασία: {parquet_path}")
        try:
            # Διάβασμα Parquet
            df = pd.read_parquet(parquet_path)
            
            # Μετατροπή σε λίστα από dictionaries "ως έχει"
            raw_data = df.to_dict(orient='records')
            
            # Αποθήκευση σε Pretty Printed JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Επιτυχής εξαγωγή -> {json_path} ({len(raw_data)} εγγραφές)\n")
            
        except Exception as e:
            print(f"❌ Σφάλμα κατά τη μετατροπή του αρχείου {parquet_path}: {e}\n")

    print("🏁 Όλα τα αρχεία μετατράπηκαν!")

if __name__ == "__main__":
    convert_all_dataset_parquet_files()

import os
import shutil

# Directory originale del dataset
source_dir = "/work/debiasing/gcapitani/data/sun397/SUN397"

# Directory di output per gli split
output_dir = "/work/debiasing/datasets/sun397"
train_output_dir = os.path.join(output_dir, "train")
test_output_dir = os.path.join(output_dir, "val")

# File di split
test_file_path = "Testing_01.txt"
train_file_path = "Training_01.txt"

# Leggi i file di split
with open(test_file_path, "r") as f:
    test_files = f.read().strip().splitlines()

with open(train_file_path, "r") as f:
    train_files = f.read().strip().splitlines()

# Funzione per copiare i file nelle directory di output
def copy_files(file_list, source_dir, output_dir):
    for file_path in file_list:
        # Percorso completo dell'immagine nella directory originale
        full_file_path = os.path.join(source_dir, file_path[1:])  # Rimuove il primo slash

        # Percorso di destinazione (mantiene la struttura delle classi)
        class_name = file_path.split('/')[2]  # Estrae il nome della classe
        destination_dir = os.path.join(output_dir, class_name)
        os.makedirs(destination_dir, exist_ok=True)
        
        # Copia il file
        try:
            shutil.copy(full_file_path, destination_dir)
        except FileNotFoundError:
            print(f"File non trovato: {full_file_path}")

# Copia i file per train e test
copy_files(train_files, source_dir, train_output_dir)
copy_files(test_files, source_dir, test_output_dir)

print(f"Split creati con successo:\n - Train: {train_output_dir}\n - Test: {test_output_dir}")

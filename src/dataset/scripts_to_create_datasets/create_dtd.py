### PROCESS DTD DATASET
import os
import shutil
from pathlib import Path
base_dir = '/work/debiasing/datasets'
downloaded_data_path = f"{base_dir}/dtd/images"
output_path = f"{base_dir}/dtd"

def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = input_path.split('/')[:-1][0]
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path)
        output_file_path = os.path.join(output_class_folder, filename)
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")

process_dataset(
    f'{base_dir}/dtd/labels/train.txt', downloaded_data_path, os.path.join(output_path, "train")
)
process_dataset(
    f'{base_dir}/dtd/labels/test.txt', downloaded_data_path, os.path.join(output_path, "val")
)
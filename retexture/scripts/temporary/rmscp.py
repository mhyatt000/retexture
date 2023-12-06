
from tqdm import tqdm
import os

def remove_directories_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in tqdm(list(file.readlines())):
            dir_path = line.strip()
            if os.path.isdir(dir_path):
                # os.rmdir(dir_path)
                os.system(f'rm -rf {dir_path}')
            else:
                print(f"Directory not found: {dir_path}")

# Usage
remove_directories_from_file('./contents.txt')

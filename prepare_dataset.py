import os
import shutil
import random

SOURCE_REAL = os.path.join("dataset", "real")
SOURCE_AI = os.path.join("dataset", "ai")
TARGET_BASE = os.path.join("dataset", "processed")

# Set random seed for scientific reproducibility
random.seed(42)

real_files = [os.path.join(SOURCE_REAL, f) for f in os.listdir(SOURCE_REAL) if f.endswith('.jpg')]
ai_files = [os.path.join(SOURCE_AI, f) for f in os.listdir(SOURCE_AI) if f.endswith('.jpg')]

# Balance classes strictly to match the minority count (3,333)
target_count = min(len(real_files), len(ai_files))
real_files = random.sample(real_files, target_count)
ai_files = random.sample(ai_files, target_count)

print(f"Balanced Dataset: {target_count} Real images and {target_count} AI images.")

def split_and_copy(file_list, class_name):
    random.shuffle(file_list)
    n_total = len(file_list)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    splits = {
        'train': file_list[:n_train],
        'val': file_list[n_train:n_train + n_val],
        'test': file_list[n_train + n_val:]
    }
    
    for split_name, files in splits.items():
        dest_dir = os.path.join(TARGET_BASE, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for fpath in files:
            shutil.copy(fpath, os.path.join(dest_dir, os.path.basename(fpath)))
            
    print(f"[{class_name}] Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

split_and_copy(real_files, "real")
split_and_copy(ai_files, "ai")
print("\nDataset preparation complete. Saved to 'dataset/processed/'.")
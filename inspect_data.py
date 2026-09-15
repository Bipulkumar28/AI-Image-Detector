import os
from PIL import Image
from tqdm import tqdm

REAL_DIR = os.path.join("dataset", "real")
AI_DIR = os.path.join("dataset", "ai")

def inspect_folder(folder_path, label_name):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    corrupted = 0
    resolutions = set()
    modes = set()
    
    print(f"\n--- Auditing {label_name} Images ---")
    print(f"Total files found: {len(files)}")
    
    for fname in tqdm(files):
        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as img:
                img.verify() # Verify file integrity
            with Image.open(fpath) as img:
                resolutions.add(img.size)
                modes.add(img.mode)
        except Exception:
            corrupted += 1
            
    print(f"Corrupted files: {corrupted}")
    print(f"Image dimensions found: {resolutions}")
    print(f"Color modes found: {modes}")
    return len(files) - corrupted

real_valid = inspect_folder(REAL_DIR, "REAL")
ai_valid = inspect_folder(AI_DIR, "AI")
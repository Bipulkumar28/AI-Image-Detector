import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Target directories
REAL_DIR = os.path.join("dataset", "real")
AI_DIR = os.path.join("dataset", "ai")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)

print("=== Downloading Public Dataset: Parveshiiii/AI-vs-Real ===")

# Force load by turning off strict split verification checks
ds = load_dataset(
    "Parveshiiii/AI-vs-Real", 
    split="train", 
    verification_mode="no_checks"
)

# Inspect dataset features/keys
sample = ds[0]
print(f"Dataset Keys Found: {list(sample.keys())}")

# Determine key names dynamically
image_key = 'image' if 'image' in sample else [k for k in sample.keys() if 'img' in k or 'image' in k][0]
label_key = [k for k in sample.keys() if k != image_key][0]

print(f"Using Image Key: '{image_key}', Label Key: '{label_key}'")

real_count = 0
ai_count = 0

print("Processing and saving images to disk...")

for idx, item in enumerate(tqdm(ds)):
    img_data = item[image_key]
    
    # Ensure image is PIL format
    if not isinstance(img_data, Image.Image):
        img = Image.open(img_data).convert('RGB')
    else:
        img = img_data.convert('RGB')
        
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    raw_label = item[label_key]
    
    # Check numeric or string representations
    if raw_label in [1, '1', 'real', 'REAL', 'Real']:
        real_count += 1
        img.save(os.path.join(REAL_DIR, f"real_{real_count:05d}.jpg"), "JPEG", quality=92)
    else:
        ai_count += 1
        img.save(os.path.join(AI_DIR, f"ai_{ai_count:05d}.jpg"), "JPEG", quality=92)

print(f"\nSUCCESS: Download complete!")
print(f"Saved Real Images: {real_count}")
print(f"Saved AI Images:   {ai_count}")
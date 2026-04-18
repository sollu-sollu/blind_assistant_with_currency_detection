# ================================================================
#  BLIND ASSISTANT — YOLO26n CURRENCY TRAINING  (Google Colab)
#  Runtime → Change runtime type → T4 GPU  ← DO THIS FIRST
# ================================================================
#  HOW TO USE:
#  Copy each ### CELL ### block below into its own Colab code cell.
#  Run them ONE BY ONE in order.
# ================================================================

"""
To download the dataset
!wget -O cur_data.zip https://data.mendeley.com/public-api/zip/48ympv8jjf/download/1
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 ── Install packages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
!pip install -q ultralytics
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 ── Upload & Extract Dataset  (READ CAREFULLY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
HOW TO GET THE DATASET:
  1. Open this page in your browser:
       https://data.mendeley.com/datasets/48ympv8jjf/1
  2. Click the orange "Download All" button
  3. Sign-in if asked (free Mendeley account)
  4. A zip file will download to your PC (name may vary, e.g. "archive.zip")

HOW TO UPLOAD TO COLAB:
  Option A (Recommended — Faster):
    - Upload the zip to your Google Drive
    - In Colab: Files panel → Mount Drive button
    - Copy the zip to /content/:
        !cp "/content/drive/MyDrive/YOUR_ZIP_NAME.zip" /content/

  Option B (Direct upload — slower for big files):
    - In Colab left panel: click the Files icon → Upload button
    - Select the zip file from your PC

THEN RUN THIS CELL:
"""

import os
import zipfile
import shutil

RAW_DATA = "raw_data"

def extract_dataset():
    # Find whatever zip file exists in /content/
    zips = [f for f in os.listdir('/content') if f.endswith('.zip')]
    if not zips:
        print("❌ No zip file found in /content/. Please upload it first.")
        print("   See the instructions above.")
        return False

    zip_path = f"/content/{zips[0]}"
    print(f"Found zip: {zip_path}")

    if os.path.exists(RAW_DATA):
        print(f"'{RAW_DATA}' already exists — skipping extraction.")
        return True

    print(f"Extracting to '{RAW_DATA}/'...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DATA)
    print("✅ Extraction complete.")
    return True

extract_dataset()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 ── Inspect: See EXACT folder names inside raw_data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
This cell prints the EXACT folder names inside the dataset.
You MUST run this before CELL 4 and check the output.
Update CLASS_MAP in CELL 4 to match the folder names you see here.
"""

def inspect_raw_data(path=RAW_DATA):
    if not os.path.exists(path):
        print(f"❌ '{path}' not found. Run CELL 2 first.")
        return

    print(f"\n📂 Contents of '{path}':\n")
    top_items = sorted(os.listdir(path))
    for item in top_items:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            img_count = len([
                f for f in os.listdir(full)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ])
            # Also check one level deeper (in case there's a subfolder)
            subdirs = [d for d in os.listdir(full) if os.path.isdir(os.path.join(full, d))]
            if subdirs:
                print(f"  📁 [{item}]  → contains sub-folders: {subdirs[:5]}")
                for sd in subdirs[:5]:
                    sd_path = os.path.join(full, sd)
                    sc = len([f for f in os.listdir(sd_path)
                              if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
                    print(f"       📁 [{sd}]  → {sc} images")
            else:
                print(f"  📁 [{item}]  → {img_count} images")
        else:
            print(f"  📄 {item}")

    print("\n⚠️  Copy the EXACT folder names shown above into CLASS_MAP in CELL 4.")

inspect_raw_data()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 ── Update CLASS_MAP and Prepare YOLO Dataset Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
⚠️  IMPORTANT: Look at the output of CELL 3.
Update the CLASS_MAP keys below to match the EXACT folder names you saw.

Examples of what the dataset folder might look like:
  If CELL 3 showed:  📁 [10]           →  Use "10": 0
  If CELL 3 showed:  📁 [Rs10]         →  Use "Rs10": 0
  If CELL 3 showed:  📁 [10_rupee]     →  Use "10_rupee": 0
  If CELL 3 showed:  📁 [note_10]      →  Use "note_10": 0
"""

import os
import shutil

DATASET_PATH = "currency_dataset"
RAW_DATA     = "raw_data"
TRAIN_RATIO  = 0.90

# ── UPDATE THESE KEYS to match EXACT folder names from CELL 3 output ──
CLASS_MAP = {
    "10":   0,
    "20":   1,
    "50":   2,
    "100":  3,
    "200":  4,
    "500":  5,
    "2000": 6,
}
# ─────────────────────────────────────────────────────────────────────

CLASS_NAMES = {v: f"{k}_rupee" for k, v in CLASS_MAP.items()}


def find_images_recursive(folder):
    """Find all images in a folder, including sub-directories."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                images.append(os.path.join(root, f))
    return sorted(images)


def prepare_yolo_data():
    # Create YOLO folder structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_PATH, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_PATH, f'labels/{split}'), exist_ok=True)

    total_train, total_val = 0, 0
    not_found = []

    for class_folder_name, class_id in CLASS_MAP.items():
        full_path = os.path.join(RAW_DATA, class_folder_name)

        if not os.path.isdir(full_path):
            not_found.append(class_folder_name)
            print(f"  ⚠  Folder '{class_folder_name}' not found. Check CLASS_MAP.")
            continue

        images = find_images_recursive(full_path)
        if not images:
            print(f"  ⚠  No images found in '{class_folder_name}'.")
            continue

        split_idx   = int(len(images) * TRAIN_RATIO)
        class_label = CLASS_NAMES[class_id]

        for i, img_path in enumerate(images):
            split       = 'train' if i < split_idx else 'val'
            img_ext     = os.path.splitext(img_path)[1]
            img_unique  = f"{class_folder_name}_{i:05d}{img_ext}"
            dst_img     = os.path.join(DATASET_PATH, f'images/{split}', img_unique)
            label_name  = f"{class_folder_name}_{i:05d}.txt"
            dst_label   = os.path.join(DATASET_PATH, f'labels/{split}', label_name)

            shutil.copy(img_path, dst_img)

            # Full-image bounding box (normalized YOLO format)
            # cx=0.5, cy=0.5, w=0.9, h=0.9 covers ~90% of the image
            with open(dst_label, 'w') as lf:
                lf.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

        train_n = split_idx
        val_n   = len(images) - split_idx
        total_train += train_n
        total_val   += val_n
        print(f"  ✅ {class_folder_name} rupee → {train_n} train | {val_n} val")

    print(f"\n📊 Dataset prepared: {total_train} train | {total_val} val images")

    if not_found:
        print(f"\n⚠️  These folders were not found: {not_found}")
        print("   Update CLASS_MAP keys to match the exact folder names from CELL 3.")

    if total_train == 0:
        print("\n❌ ERROR: 0 images found! Dataset not ready for training.")
        print("   1. Re-run CELL 3 to see the actual folder names.")
        print("   2. Update CLASS_MAP keys in this cell to match them.")
        print("   3. Re-run this cell.")
    else:
        print("\n✅ Run CELL 5 next to create the YAML config.")

prepare_yolo_data()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 ── Create YOLO data.yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_yaml():
    names_block = "\n".join(
        f"  {class_id}: {CLASS_NAMES[class_id]}"
        for class_id in sorted(CLASS_NAMES.keys())
    )
    yaml_content = f"""# Indian Currency Detection - YOLO26n
path: /content/{DATASET_PATH}
train: images/train
val:   images/val

nc: {len(CLASS_MAP)}
names:
{names_block}
"""
    with open('currency_data.yaml', 'w') as f:
        f.write(yaml_content)

    print("✅ currency_data.yaml created:\n")
    print(yaml_content)

create_yaml()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 ── Verify dataset (check image counts before training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_dataset():
    for split in ['train', 'val']:
        img_dir   = os.path.join(DATASET_PATH, f'images/{split}')
        lbl_dir   = os.path.join(DATASET_PATH, f'labels/{split}')
        img_count = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        lbl_count = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
        status    = "✅" if img_count > 0 and img_count == lbl_count else "❌"
        print(f"  {status} {split}: {img_count} images, {lbl_count} labels")

    if os.path.exists('currency_data.yaml'):
        print("  ✅ currency_data.yaml exists")
    else:
        print("  ❌ currency_data.yaml missing — run CELL 5 first")

print("Dataset verification:")
verify_dataset()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 ── Train YOLO26n
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from ultralytics import YOLO

def start_training():
    print("=" * 50)
    print("  Starting YOLO26n Fine-Tuning  🚀")
    print("=" * 50)

    model = YOLO('yolo26n.pt')  # Auto-downloads if not present

    results = model.train(
        data     = 'currency_data.yaml',
        epochs   = 60,    # 60 is good; reduce to 30 for a quick test
        imgsz    = 320,   # 320×320 — fast on RPi 4
        batch    = 32,    # T4 GPU handles this easily
        device   = 0,     # CUDA GPU
        name     = 'indian_currency_yolo26',
        patience = 15,    # Stop early if no improvement
        augment  = True,  # Built-in data augmentation
        cache    = True,  # Cache images in RAM for speed
        exist_ok = True,  # Don't error if output folder exists
    )

    print("\n Validating best model...")
    best = YOLO('runs/detect/indian_currency_yolo26/weights/best.pt')
    m    = best.val(data='currency_data.yaml', verbose=False)
    print(f"\n  mAP@50:    {m.box.map50:.3f}")
    print(f"  mAP@50-95: {m.box.map:.3f}")
    return best

best_model = start_training()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 ── Export model for Raspberry Pi 4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_for_rpi():
    from ultralytics import YOLO
    model = YOLO('runs/detect/indian_currency_yolo26/weights/best.pt')

    print("Exporting to NCNN (fastest format for Raspberry Pi 4 ARM CPU)...")
    model.export(format='ncnn', imgsz=320)

    print("Export done! ✅")
    print("\n📦 Files to copy to your Raspberry Pi:")
    print("   → runs/detect/indian_currency_yolo26/weights/best_ncnn_model/")
    print("      (copy the entire folder, rename it to 'currency_yolo26n_ncnn_model')")
    print("\nOR use the .pt file directly (easier):")
    print("   → runs/detect/indian_currency_yolo26/weights/best.pt")
    print("      (copy and rename to 'currency_yolo26n.pt')")

export_for_rpi()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 ── Download the trained model to your PC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
Run this to download best.pt to your local PC:

from google.colab import files
files.download('runs/detect/indian_currency_yolo26/weights/best.pt')

Then rename it 'currency_yolo26n.pt' and copy to:
  c:\\Rajendran\\blind\\currency_yolo26n.pt
"""

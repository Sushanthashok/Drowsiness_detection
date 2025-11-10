# training/prepare_mrl.py
import os, shutil, random
from pathlib import Path

SRC = Path("S:\drowsiness_detection\data")  # <-- CHANGE to where you extracted the MRL dataset
DEST = Path("../data/eyes")  # relative to /training

TRAIN_RATIO = 0.8  # 80% train, 20% val

def main():
    train_open = DEST/"train"/"open"; train_open.mkdir(parents=True, exist_ok=True)
    train_closed = DEST/"train"/"closed"; train_closed.mkdir(parents=True, exist_ok=True)
    val_open = DEST/"val"/"open"; val_open.mkdir(parents=True, exist_ok=True)
    val_closed = DEST/"val"/"closed"; val_closed.mkdir(parents=True, exist_ok=True)

    all_imgs = list(SRC.rglob("*.png"))
    print(f"Found {len(all_imgs)} images")

    open_imgs = [f for f in all_imgs if "_1_" in f.name]   # open eyes
    closed_imgs = [f for f in all_imgs if "_0_" in f.name] # closed eyes

    print(f"Open: {len(open_imgs)} | Closed: {len(closed_imgs)}")

    def split_and_copy(imgs, dst_open, dst_val):
        random.shuffle(imgs)
        cut = int(len(imgs)*TRAIN_RATIO)
        return imgs[:cut], imgs[cut:]

    train_o, val_o = split_and_copy(open_imgs, train_open, val_open)
    train_c, val_c = split_and_copy(closed_imgs, train_closed, val_closed)

    for i, f in enumerate(train_o):
        shutil.copy2(f, train_open/f.name)
    for i, f in enumerate(val_o):
        shutil.copy2(f, val_open/f.name)
    for i, f in enumerate(train_c):
        shutil.copy2(f, train_closed/f.name)
    for i, f in enumerate(val_c):
        shutil.copy2(f, val_closed/f.name)

    print("✅ Dataset organized into train/val splits!")

if __name__ == "__main__":
    main()

"""
Training script — PyTorch version.
Fine-tunes EfficientNet-B0 on APTOS 2019 Blindness Detection dataset.

Usage:
    python train.py --data_dir data/aptos2019 --epochs 30 --batch_size 32

Expected layout:
    data/aptos2019/
        train_images/   (PNG/JPG fundus images)
        train.csv       (columns: id_code, diagnosis)

Output:
    dr_efficientnet_weights.pt   (saved to project root)
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import build_model, DEVICE
from preprocessing import preprocess_image

torch.manual_seed(42)
np.random.seed(42)


# ── Dataset ───────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    def __init__(self, records: list[dict], img_dir: str, augment: bool = False):
        self.records  = records
        self.img_dir  = img_dir
        self.augment  = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        # Support both direct ID and full relative path (for folder-based)
        if os.path.sep in rec["id_code"] or "/" in rec["id_code"]:
            img_path = os.path.join(self.img_dir, rec["id_code"])
        else:
            img_path = os.path.join(self.img_dir, rec["id_code"] + ".png")
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.imread(img_path) # Now we actually read it!
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, enhance=True)  # (224,224,3) float32

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k=k)

        # HWC → CHW
        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        label  = torch.tensor(rec["label"], dtype=torch.long)
        return tensor, label


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Map names to labels ──
    NAME_MAP = {
        "no_dr": 0, "normal": 0, "0": 0,
        "mild": 1, "1": 1,
        "moderate": 2, "2": 2,
        "severe": 3, "3": 3,
        "proliferate_dr": 4, "proliferative_dr": 4, "4" : 4
    }

    # ── Data Detection ──
    csv_path = os.path.join(args.data_dir, "train.csv")
    records = []
    img_dir = args.data_dir

    if os.path.exists(csv_path):
        # CSV-based loading
        df = pd.read_csv(csv_path)
        records = [{"id_code": r.id_code, "label": int(r.diagnosis)} for r in df.itertuples()]
        img_dir = os.path.join(args.data_dir, "train_images")
        print(f"[SUCCESS] Found CSV-based dataset with {len(records)} images.")
    else:
        print(f"[INFO] No train.csv found. Scanning recursively for folders...")
        # Walk through ALL subfolders to find one containing our target labels
        for root, dirs, _ in os.walk(args.data_dir):
            # Check if current directory has folders that look like labels
            found_labels = [d for d in dirs if d.lower().strip() in NAME_MAP]
            if len(found_labels) >= 3: # Found at least 3 classes
                print(f"[SUCCESS] Found dataset structure at: {root}")
                for dname in found_labels:
                    label = NAME_MAP[dname.lower().strip()]
                    full_d = os.path.join(root, dname)
                    for f in os.listdir(full_d):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Store relative path from current 'root'
                            records.append({"id_code": os.path.join(dname, f), "label": label})
                img_dir = root
                break
        
        if not records:
            print("[ERROR] Could not find any data (CSV or folders) in " + args.data_dir)
            return

    labels = [r["label"] for r in records]
    tr_rec, val_rec = train_test_split(records, test_size=0.15, stratify=labels, random_state=42)

    tr_ds  = APTOSDataset(tr_rec,  img_dir, augment=True)
    val_ds = APTOSDataset(val_rec, img_dir, augment=False)
    
    # Use 0 workers for Windows stability in smaller hackathon environments
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.3)

    best_acc = 0.0
    print(f"[INFO] Training on {DEVICE} | {len(tr_rec)} train / {len(val_rec)} val samples")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_dl, criterion, optimizer)
        vl_loss, vl_acc = val_epoch(model, val_dl, criterion)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"Val loss: {vl_loss:.4f} acc: {vl_acc:.4f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "dr_efficientnet_weights.pt")
            print(f"  ✅ Saved best model (val_acc={best_acc:.4f})")

    # Phase 2: High-Accuracy Fine-tune
    if args.fine_tune:
        print("\n[PHASE 2] Unfreezing ALL layers for High-Accuracy Fine-Tuning...")
        for param in model.parameters():
            param.requires_grad = True # EXAMINE EVERYTHING!
        
        # Super low learning rate to not "break" the brain
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
        
        for epoch in range(1, args.fine_tune_epochs + 1):
            tr_loss, tr_acc = train_epoch(model, tr_dl, criterion, optimizer)
            vl_loss, vl_acc = val_epoch(model, val_dl, criterion)
            print(f"[FINE-TUNE] Epoch {epoch:02d}/{args.fine_tune_epochs} | "
                  f"Train Acc: {tr_acc:.4f} | Val Acc: {vl_acc:.4f}")
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(model.state_dict(), "dr_efficientnet_weights.pt")
                print(f"  🔥 NEW BEST: {vl_acc*100:.2f}%")

    print(f"\n[DONE] Best validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         default="data/aptos2019")
    parser.add_argument("--epochs",           type=int, default=30)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--fine_tune",        action="store_true", default=False)
    parser.add_argument("--fine_tune_epochs", type=int, default=15)
    main(parser.parse_args())

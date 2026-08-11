"""
H.E.M.A. — robustness check: repeats the train/val split + training with multiple
random seeds. The held-out unseen-site test set (Ahmadiyya Muslim Hospital) is
identical across all seeds — only the train/val shuffle of the remaining hospitals
changes. This tells us whether the AUROC gap we saw with seed=42 is a stable
finding or an artifact of one particular split.

Run from C:\\Users\\NYASA\\Downloads\\delivery
"""
import os
import csv
import random
import shutil
import subprocess
import json
import statistics

DATA_DIR = r"C:\Users\NYASA\Downloads\output"
CSV_PATH = os.path.join(DATA_DIR, "clean data.csv")
HELD_OUT_HOSPITAL = "Ahmadiyya Muslim Hospital"
VAL_FRACTION = 0.15
LABELS = ["Anemic", "Non-anemic"]
SEEDS = [42, 123, 7]

def build_split(seed, out_root):
    random.seed(seed)
    id_to_hospital = {}
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            id_to_hospital[row["IMAGE_ID"].strip()] = row["HOSPITAL"].strip()

    pool = {l: [] for l in LABELS}
    held = {l: [] for l in LABELS}
    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".png"):
                continue
            image_id = os.path.splitext(fname)[0]
            hospital = id_to_hospital.get(image_id)
            if hospital is None:
                continue
            full_path = os.path.join(folder, fname)
            (held if hospital == HELD_OUT_HOSPITAL else pool)[label].append(full_path)

    split_map = {"train": {l: [] for l in LABELS}, "val": {l: [] for l in LABELS}, "test_unseen_site": held}
    for label in LABELS:
        files = pool[label][:]
        random.shuffle(files)
        n_val = round(len(files) * VAL_FRACTION)
        split_map["val"][label] = files[:n_val]
        split_map["train"][label] = files[n_val:]

    for split, label_files in split_map.items():
        for label in LABELS:
            dest_dir = os.path.join(out_root, split, label)
            os.makedirs(dest_dir, exist_ok=True)
            for src_path in label_files[label]:
                shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

def main():
    all_results = []
    for seed in SEEDS:
        out_root = rf"C:\Users\NYASA\Downloads\clean_output_split_seed{seed}"
        results_dir = rf".\results_seed{seed}"
        print(f"\n=== Seed {seed}: building split ===")
        build_split(seed, out_root)

        print(f"=== Seed {seed}: training (this will take a while on CPU) ===")
        subprocess.run([
            "python", "train.py",
            "--data_root", out_root,
            "--architecture", "efficientnet_b0",
            "--epochs_head", "5",
            "--epochs_finetune", "10",
            "--out_dir", results_dir,
        ], check=True)

        with open(os.path.join(results_dir, "results.json")) as f:
            all_results.append(json.load(f))

    print("\n\n=== AGGREGATE ACROSS SEEDS ===")
    for split_key in ["in_domain_val", "unseen_site_test"]:
        aurocs = [r[split_key]["auroc"] for r in all_results]
        accs = [r[split_key]["accuracy"] for r in all_results]
        print(f"\n{split_key}:")
        print(f"  AUROC:    {[round(a,3) for a in aurocs]}  mean={statistics.mean(aurocs):.3f}  stdev={statistics.stdev(aurocs):.3f}")
        print(f"  Accuracy: {[round(a,3) for a in accs]}  mean={statistics.mean(accs):.3f}  stdev={statistics.stdev(accs):.3f}")

if __name__ == "__main__":
    main()
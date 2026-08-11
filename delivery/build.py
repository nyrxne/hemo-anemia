import os
import csv
import random
import shutil

random.seed(42)

DATA_DIR = r"C:\Users\NYASA\Downloads\output"
CSV_PATH = os.path.join(DATA_DIR, "clean data.csv")
OUT_ROOT = r"C:\Users\NYASA\Downloads\clean_output_split"
HELD_OUT_HOSPITAL = "Ahmadiyya Muslim Hospital"
VAL_FRACTION = 0.15
LABELS = ["Anemic", "Non-anemic"]

# 1. Load IMAGE_ID -> HOSPITAL from the metadata CSV
id_to_hospital = {}
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row["IMAGE_ID"].strip()
        id_to_hospital[image_id] = row["HOSPITAL"].strip()

print(f"Loaded {len(id_to_hospital)} rows from clean data.csv")

# 2. Walk output/Anemic and output/Non-anemic, look up each file's hospital
buckets = {"test_unseen_site": {l: [] for l in LABELS},
           "pool": {l: [] for l in LABELS}}  # pool = will become train+val

missing = []
for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".png"):
            continue
        image_id = os.path.splitext(fname)[0]
        hospital = id_to_hospital.get(image_id)
        if hospital is None:
            missing.append(fname)
            continue
        full_path = os.path.join(folder, fname)
        if hospital == HELD_OUT_HOSPITAL:
            buckets["test_unseen_site"][label].append(full_path)
        else:
            buckets["pool"][label].append(full_path)

if missing:
    print(f"WARNING: {len(missing)} image files had no matching row in the CSV "
          f"(not placed in any split): {missing[:10]}{'...' if len(missing) > 10 else ''}")

# 3. Split the pool into train/val, stratified by label
train_files = {l: [] for l in LABELS}
val_files = {l: [] for l in LABELS}
for label in LABELS:
    files = buckets["pool"][label][:]
    random.shuffle(files)
    n_val = round(len(files) * VAL_FRACTION)
    val_files[label] = files[:n_val]
    train_files[label] = files[n_val:]

# 4. Copy into OUT_ROOT/split/label/filename
split_map = {
    "train": train_files,
    "val": val_files,
    "test_unseen_site": buckets["test_unseen_site"],
}

for split, label_files in split_map.items():
    for label in LABELS:
        dest_dir = os.path.join(OUT_ROOT, split, label)
        os.makedirs(dest_dir, exist_ok=True)
        for src_path in label_files[label]:
            shutil.copy2(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

# 5. Report final counts for verification
print("\n=== Split summary ===")
for split, label_files in split_map.items():
    counts = {label: len(label_files[label]) for label in LABELS}
    print(f"{split}: {counts}  (total {sum(counts.values())})")

# 6. Sanity check: confirm test_unseen_site really is only the held-out hospital
test_ids = set()
for label in LABELS:
    for path in buckets["test_unseen_site"][label]:
        test_ids.add(os.path.splitext(os.path.basename(path))[0])
hospitals_in_test = {id_to_hospital[i] for i in test_ids}
print(f"\nHospital(s) present in test_unseen_site: {hospitals_in_test}")
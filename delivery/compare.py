import csv
from collections import defaultdict

CSV_PATH = r"C:\Users\NYASA\Downloads\output\clean data.csv"
HELD_OUT = "Ahmadiyya Muslim Hospital"

train_rows, test_rows = [], []
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        (test_rows if row["HOSPITAL"].strip() == HELD_OUT else train_rows).append(row)

def summarize(rows, label):
    hb = [float(r["HB_LEVEL"]) for r in rows]
    age = [float(r["Age(Months)"]) for r in rows]
    anemic = sum(1 for r in rows if r["REMARK"].strip() == "Anemic")
    print(f"\n{label} (n={len(rows)})")
    print(f"  HB_LEVEL: mean={sum(hb)/len(hb):.2f} min={min(hb)} max={max(hb)}")
    print(f"  Age(Months): mean={sum(age)/len(age):.1f} min={min(age)} max={max(age)}")
    print(f"  Class balance: {anemic} anemic / {len(rows)-anemic} non-anemic ({anemic/len(rows):.1%} anemic)")
    sev = defaultdict(int)
    for r in rows: sev[r["Severity"].strip()] += 1
    print(f"  Severity distribution: {dict(sev)}")

summarize(train_rows, "Train+val hospitals (combined)")
summarize(test_rows, f"Held-out: {HELD_OUT}")
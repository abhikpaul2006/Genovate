import os

TEST_ROOT = r"D:\Dataset_Rerelease\Models\Model_1\test"

event_dir = os.path.join(TEST_ROOT, "event_18")
print(f"== Inside {event_dir} ==")
files = sorted(os.listdir(event_dir))
for f in files:
    print(" ", f)

print("\n== Row counts / columns for each CSV in event_18 ==")
import csv
for f in files:
    path = os.path.join(event_dir, f)
    if f.endswith(".csv"):
        with open(path, "r") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            n_rows = sum(1 for _ in reader)
        print(f"\n--- {f} ---")
        print("columns:", header)
        print("num data rows:", n_rows)

# list all 29 event folders to confirm naming/IDs
print("\n== All event_* folders in test ==")
all_entries = sorted(os.listdir(TEST_ROOT))
event_folders = [e for e in all_entries if e.startswith("event_")]
print(len(event_folders), "event folders:")
print(event_folders)
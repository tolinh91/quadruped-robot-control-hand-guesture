import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "hagrid_dataset.csv"

df = pd.read_csv(DATASET_PATH, dtype=str, header=0)
print(f"Tổng số cột đọc được: {len(df.columns)}")
print("Tên các cột:", list(df.columns[:5]), "...", list(df.columns[-2:]))

for idx in range(min(3, len(df))):
    row = df.iloc[idx]
    print(f"\n--- THỬ PARSE DÒNG {idx + 1} ---")
    label = str(row.iloc[-1]).strip()
    print(f"Label: '{label}'")
    
    feats_raw = row.iloc[:-1].values
    errors = []
    for col_idx, val in enumerate(feats_raw):
        try:
            val_str = str(val).strip().replace(",", ".")
            float(val_str)
        except Exception as e:
            errors.append((col_idx, val, str(e)))
            
    if errors:
        print(f"❌ Tìm thấy {len(errors)} ô bị lỗi không ép kiểu được:")
        for err in errors[:5]: # In 5 lỗi đầu
            print(f"   + Cột index {err[0]}: giá trị = '{err[1]}' | Lỗi = {err[2]}")
    else:
        print("✓ Dòng này parse float HOÀN HẢO!")
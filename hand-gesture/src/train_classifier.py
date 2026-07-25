import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from gesture_classifier import GestureClassifier

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DATASET_PATH = ROOT_DIR / "datasets" / "hagrid_dataset.csv"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "gesture_classifier_xgb.pkl"


def main():
    if not DATASET_PATH.exists():
        print(f"[-] LỖI: Không tìm thấy file {DATASET_PATH}")
        return

    print(f"[+] Đang đọc dataset từ: {DATASET_PATH}")

    # Đọc CSV
    df = pd.read_csv(DATASET_PATH, header=0)

    # Tách X (features) và y (labels)
    X_df = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]

    # Lọc các dòng sạch
    valid_y_mask = y_raw.notna() & (
        ~y_raw.astype(str).str.lower().isin(["nan", "none", "label"])
    )
    X_converted = X_df.apply(pd.to_numeric, errors="coerce")
    valid_X_mask = ~X_converted.isna().any(axis=1)

    final_mask = valid_y_mask & valid_X_mask

    X_data = X_converted[final_mask].to_numpy(dtype=np.float32)
    y_data = y_raw[final_mask].astype(str).str.strip().values

    print(f"[+] Số lượng mẫu sạch: {len(X_data)}")

    # Trích xuất 78 đặc trưng nâng cao từ 21 keypoints
    dummy_classifier = GestureClassifier.__new__(GestureClassifier)
    X_features = []

    for row in X_data:
        # Nếu CSV chứa 63 đặc trưng (21 points * 3 coords x,y,z)
        if len(row) == 63:
            pts = row.reshape(21, 3)
        # Nếu CSV chứa 42 đặc trưng (21 points * 2 coords x,y)
        elif len(row) == 42:
            pts = row.reshape(21, 2)
        else:
            pts = row[:63].reshape(21, 3)

        feats = dummy_classifier.extract_features(pts)
        X_features.append(feats)

    X_features = np.array(X_features)
    print(
        f"[+] Đã trích xuất xong vector đặc trưng nâng cao: {X_features.shape[1]} đặc trưng/mẫu"
    )

    # Encode label dạng chuỗi sang số cho XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    # Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện XGBoost
    print("[+] Đang huấn luyện XGBoost Classifier...")
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Đánh giá
    y_pred = clf.predict(X_test_scaled)
    target_names = label_encoder.classes_

    print("\n[✓] BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MỚI:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Lưu Pipeline (Scaler + Model + LabelEncoder)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "scaler": scaler,
        "model": clf,
        "label_encoder": label_encoder,
    }
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"[✓] Đã lưu Pipeline thành công tại: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
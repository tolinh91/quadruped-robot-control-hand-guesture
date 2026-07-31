import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from gesture_classifier import GestureClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "datasets" / "gesture_dataset.csv"
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
    valid_y_mask = y_raw.notna() & (~y_raw.astype(str).str.lower().isin(["nan", "none", "label"]))
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
    print(f"[+] Đã trích xuất xong vector đặc trưng nâng cao: {X_features.shape[1]} đặc trưng/mẫu")

    # Encode label dạng chuỗi sang số cho XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    # Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )

    # ==========================================================
    # Standard Scaler
    # ==========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================================
    # Train XGBoost
    # ==========================================================

    print("[+] Đang huấn luyện XGBoost...")
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    start_train = time.perf_counter()
    clf.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - start_train

    # ==========================================================
    # Predict
    # ==========================================================

    start_infer = time.perf_counter()
    y_pred = clf.predict(X_test_scaled)
    infer_time = time.perf_counter() - start_infer
    infer_per_sample = infer_time / len(X_test)
    target_names = label_encoder.classes_

    # ==========================================================
    # Accuracy
    # ==========================================================

    accuracy = accuracy_score(y_test, y_pred)

    # ==========================================================
    # Classification Report
    # ==========================================================

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
    )

    print("\n")
    print("=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)

    print(f"Accuracy               : {accuracy:.4f}")

    print(f"Macro Precision        : {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall           : {report['macro avg']['recall']:.4f}")
    print(f"Macro F1-score         : {report['macro avg']['f1-score']:.4f}")

    print(f"Weighted Precision     : {report['weighted avg']['precision']:.4f}")
    print(f"Weighted Recall        : {report['weighted avg']['recall']:.4f}")
    print(f"Weighted F1-score      : {report['weighted avg']['f1-score']:.4f}")

    print(f"Training Time          : {train_time:.3f} s")
    print(f"Inference Time         : {infer_time:.6f} s")
    print(f"Inference / Sample     : {infer_per_sample*1000:.4f} ms")

    print()
    print(classification_report(
        y_test,
        y_pred,
        target_names=target_names
    ))
    
    # ==========================================================
    # Confusion Matrix
    # ==========================================================

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix\n")
    print(cm)
    
    # ==========================================================
    # Lưu Pipeline
    # ==========================================================

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipeline = {
        "scaler": scaler,
        "model": clf,
        "label_encoder": label_encoder,
    }
    joblib.dump(pipeline,MODEL_SAVE_PATH)
    print(f"\n[✓] Đã lưu Pipeline tại: {MODEL_SAVE_PATH}")

    # ==========================================================
    # Lưu Metrics
    # ==========================================================

    metrics = pd.DataFrame([{
        "Accuracy": accuracy,
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
        "Training Time (s)": train_time,
        "Inference Time (s)": infer_time,
        "Inference (ms/sample)": infer_per_sample * 1000
    }])
    metrics_path = ROOT_DIR / "models" / "xgboost_mp_metrics.csv"
    metrics.to_csv(metrics_path,index=False)
    print(f"[✓] Đã lưu Metrics tại: {metrics_path}")


if __name__ == "__main__":
    main()
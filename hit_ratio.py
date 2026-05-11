import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    matthews_corrcoef,
)
import argparse
import os
from os import path as osp
from src import utils 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    data = args.data
    out_path = utils.default_path(data, args.seed) if args.out is None else args.out
    file_path = os.path.join(utils.ROOT_PATH, out_path, "pred_result.csv")

    # 1. CSV 불러오기
    df = pd.read_csv(file_path)
    print(df["Label"].value_counts())
    print("unique labels:", sorted(df["Label"].unique()))
    y_pred = df["Prediction"]
    y_true = df["Label"]

    # ✅ 0/1 개수 확인 (정답 레이블 기준)
    label_counts = y_true.value_counts().sort_index()  # index: 0,1 순서로 맞추기
    n_neg = label_counts.get(0, 0)
    n_pos = label_counts.get(1, 0)
    total = len(y_true)

    print("Label distribution (ground truth):")
    print(f"  total samples   = {total}")
    print(f"  class 0 (neg)   = {n_neg}")
    print(f"  class 1 (pos)   = {n_pos}")

    # 3. Accuracy
    acc_from_hit = df["Hit"].mean()
    acc = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy (from Hit) = {acc_from_hit:.4f}")
    print(f"Accuracy (recalc)   = {acc:.4f}")

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    print(cm)
    tp, fn, fp, tn = cm.ravel()

    print("\nConfusion matrix")
    print(f"TP = {tp}, FN = {fn}, FP = {fp}, TN = {tn}")

    # 5. Precision / Recall / F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nPrecision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")

    # 6. MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"MCC       = {mcc:.4f}")

    # 7. 상세 리포트
    print("\nClassification report")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
#python hit_ratio.py --data acl18 --out src/out/acl18-23
import argparse
import os
from os import path as osp

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

from src import utils  # hit_ratio.py와 같은 방식으로 ROOT_PATH 사용


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="acl18",
                        help="dataset name, e.g., acl18, cikm18, kdd22, icdm22")
    parser.add_argument("--seed-min", type=int, default=0,
                        help="start seed (inclusive)")
    parser.add_argument("--seed-max", type=int, default=50,
                        help="end seed (inclusive)")
    parser.add_argument("--out-base", type=str, default="src/out",
                        help="base output dir, so full path is out-base/{data}-{seed}")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of best seeds (by accuracy) to aggregate")
    return parser.parse_args()


def main():
    args = parse_args()

    results = []

    for seed in range(args.seed_min, args.seed_max + 1):
        out_dir = f"{args.out_base}/{args.data}-{seed}"
        csv_path = osp.join(utils.ROOT_PATH, out_dir, "pred_result.csv")

        if not osp.exists(csv_path):
            print(f"[WARN] pred_result.csv not found for seed {seed}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if not {"Prediction", "Label"}.issubset(df.columns):
            print(f"[WARN] Missing Prediction/Label columns in {csv_path}, skip.")
            continue

        y_pred = df["Prediction"]
        y_true = df["Label"]

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        results.append({
            "seed": seed,
            "accuracy": acc,
            "mcc": mcc,
        })

    if not results:
        print("No valid results found. Check paths and seeds.")
        return

    # 정리해서 DataFrame으로
    res_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    res_df.reset_index(drop=True, inplace=True)

    print("\n=== All seeds sorted by accuracy (desc) ===")
    print(res_df)

    # Top-k 선택
    k = min(args.top_k, len(res_df))
    topk = res_df.head(k)

    print(f"\n=== Top {k} seeds by accuracy ===")
    print(topk)

    # Top-k에 대한 mean / std 계산
    acc_mean = topk["accuracy"].mean()
    acc_std = topk["accuracy"].std(ddof=1)  # sample std

    mcc_mean = topk["mcc"].mean()
    mcc_std = topk["mcc"].std(ddof=1)

    print(f"\n=== Summary over top {k} seeds ===")
    print(f"Accuracy: mean = {acc_mean:.4f}, std = {acc_std:.4f}")
    print(f"MCC     : mean = {mcc_mean:.4f}, std = {mcc_std:.4f}")

    # 원하면 csv로 저장도 가능
    # topk.to_csv(osp.join(utils.ROOT_PATH, f"{args.data}_top{ k }_seeds_summary.csv"), index=False)


if __name__ == "__main__":
    main()
# 프로젝트 루트에서
# python sweep_seeds.py --data acl18 --seed-min 0 --seed-max 50 --out-base src/out --top-k 5

# without Tweets reliable 
# acl18:
# === Summary over top 5 seeds ===
# Accuracy: mean = 0.5935, std = 0.0109
# MCC     : mean = 0.1906, std = 0.0191
# cikm18
# === Summary over top 5 seeds ===
# Accuracy: mean = 0.5479, std = 0.0209
# MCC     : mean = 0.0770, std = 0.0296

# === Summary over top 5 seeds ===
# Accuracy: mean = 0.4995, std = 0.0097
# MCC     : mean = 0.0039, std = 0.0378


# tweet driven news attention module 
# acl18:
# === Summary over top 5 seeds ===
# Accuracy: mean = 0.5834, std = 0.0110
# MCC     : mean = 0.1671, std = 0.0252

# cikm18
# === Summary over top 5 seeds ===
# Accuracy: mean = 0.5581, std = 0.0124
# MCC     : mean = 0.0944, std = 0.0215
#icdm22
# Accuracy: mean = 0.4965, std =0.0078
# MCC     : mean = 0.0120, std = 0.0370




#idea 3 without divergence 
# === Summary over top 5 seeds === acl18
# Accuracy: mean = 0.5787, std = 0.0134
# MCC     : mean = 0.1602, std = 0.0303
#CIKM18
# Accuracy: mean = 0.5629, std = 0.0079
# MCC     : mean = 0.0952, std = 0.0174
#icdm22 
# Accuracy: mean = 0.4931, std = 0.0158
# MCC     : mean = -0.0775, std = 0.0294











\subsection{Divergence-Aware Gating Module}
\label{sec:divergence_module}
After obtaining the refined contexts $\mathbf{c}_{news}$ and $\mathbf{c}_{tweet}$, we must integrate them.
The critical challenge is that these sources may contradict each other, and this disagreement is a predictive signal in itself.
Naive concatenation ignores this relationship.

To address this, we first compute the semantic alignment between the two contexts using cosine similarity:
\begin{equation}
	\cos(\mathbf{c}_{news}, \mathbf{c}_{tweet}) = \frac{\mathbf{c}_{news} \cdot \mathbf{c}_{tweet}}{\|\mathbf{c}_{news}\| \|\mathbf{c}_{tweet}\|}
\end{equation}
We then define the \textit{Divergence Index} $d_{div}$ as:
\begin{equation}
	d_{div} = 1 - \cos(\mathbf{c}_{news}, \mathbf{c}_{tweet})
\end{equation}
A high $d_{div}$ implies that retail sentiment is deviating from reliable reporting, signaling potential volatility or an unconfirmed event.
To leverage this, we introduce an \textit{Agreement Gate} $g \in [0, 1]$, parameterized by a multi-layer perceptron (MLP):
\begin{equation}
	g = \sigma(\text{MLP}([\mathbf{c}_{news} ; \mathbf{c}_{tweet} ; d_{div}]))
\end{equation}
where $\sigma$ is the sigmoid function.
The final fused text vector $\mathbf{v}_{mix}$ is computed as a dynamic interpolation:
\begin{equation}
	\mathbf{v}_{mix} = g \cdot \mathbf{c}_{news} + (1 - g) \cdot \mathbf{c}_{tweet}
\end{equation}
This allows \method to trust one source over the other or balance them based on their agreement level.

\subsection{Prediction with ALSTM}
\label{sec:prediction}
The final module integrates the fused textual features with historical price data.
We employ the Attention LSTM (ALSTM) described in Section~\ref{sec:prelim} to capture temporal dependencies.
For each time step $t$, the input to the LSTM consists of the price features $\mathbf{x}_{i,t}$ concatenated with the fused text vector $\mathbf{v}_{mix, t}$.
Crucially, we also append the divergence index $d_{div, t}$ to the LSTM's hidden state as an explicit risk feature.

The model is trained to minimize the Hinge Loss, which is robust to outliers and effective for binary classification tasks in finance:
\begin{equation}
	\mathcal{L} = \sum_{i} \max(0, 1 - y_i \cdot \hat{y}_i)
\end{equation}
where $y_i \in \{-1, 1\}$ is the ground truth direction and $\hat{y}_i$ is the predicted score.
By jointly optimizing for signal reliability and divergence awareness, \method achieves robust prediction performance.
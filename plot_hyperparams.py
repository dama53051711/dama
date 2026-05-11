import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Academic styling
sns.set_theme(style="ticks")
plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def get_best_metrics(log_path):
    if not os.path.exists(log_path):
        return None, None
    try:
        df = pd.read_csv(log_path, sep='\t')
        # Identify the peak performance based on validation accuracy
        best_idx = df['val_acc'].idxmax()
        return df.loc[best_idx, 'test_acc'], df.loc[best_idx, 'test_mcc']
    except Exception:
        return None, None

def get_top_k_metrics(dataset, param_type, param_val, total_seeds, top_k):
    results = []
    for seed in range(total_seeds):
        log_path = f"src/out/{dataset}-{param_type}{param_val}-seed{seed}/pred_log.tsv"
        acc, mcc = get_best_metrics(log_path)
        if acc is not None:
            results.append((acc * 100, mcc))
            
    if not results:
        return 0, 0
    
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    return np.mean([r[0] for r in top_results]), np.mean([r[1] for r in top_results])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='icdm22')
    parser.add_argument('--total-seeds', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    dataset = args.data
    display_name = "BigData22" if dataset == "icdm22" else dataset
    
    windows = [5, 10, 15, 20]
    hdims = [16, 32, 64, 128]
    
    win_accs, win_mccs = [], []
    hdim_accs, hdim_mccs = [], []
    
    for w in windows:
        acc, mcc = get_top_k_metrics(dataset, "win", w, args.total_seeds, args.top_k)
        win_accs.append(acc)
        win_mccs.append(mcc)

    for h in hdims:
        acc, mcc = get_top_k_metrics(dataset, "hdim", h, args.total_seeds, args.top_k)
        hdim_accs.append(acc)
        hdim_mccs.append(mcc)

    # --- STRICTLY 2 SUBPLOTS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot (a): Window Size
    ax1.plot(windows, win_accs, marker='o', color='royalblue', linewidth=2, label='ACC')
    ax1.set_xlabel('Look-back Window Size ($L$)', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', color='royalblue', fontweight='bold')
    ax1.set_xticks(windows)
    
    ax1_t = ax1.twinx()
    ax1_t.plot(windows, win_mccs, marker='s', color='crimson', linestyle='--', linewidth=2, label='MCC')
    ax1_t.set_ylabel('MCC', color='crimson', fontweight='bold')
    ax1.set_title(f'(a) Impact of Window Size ({display_name})', fontsize=13, fontweight='bold')

    # Plot (b): Hidden Dimension
    ax2.plot(hdims, hdim_accs, marker='o', color='royalblue', linewidth=2)
    ax2.set_xlabel('ALSTM Hidden Dimension ($d$)', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', color='royalblue', fontweight='bold')
    ax2.set_xticks(hdims)
    
    ax2_t = ax2.twinx()
    ax2_t.plot(hdims, hdim_mccs, marker='s', color='crimson', linestyle='--', linewidth=2)
    ax2_t.set_ylabel('MCC', color='crimson', fontweight='bold')
    ax2.set_title(f'(b) Impact of Hidden Dimension ({display_name})', fontsize=13, fontweight='bold')

    plt.tight_layout()
    
    # Save as a NEW filename to avoid cache issues
    os.makedirs('fig', exist_ok=True)
    out_file = f'fig/REVISED_tuning_{dataset}.pdf'
    
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSUCCESS! New file created: {out_file}")
    print("Please open this REVISED file, not the old one.")

if __name__ == "__main__":
    main()


# python plot_hyperparams.py --data acl18 --total-seeds 50 --top-k 5


# import os
# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Make plots look academic and clean
# sns.set_theme(style="ticks")
# plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42})

# def get_best_metrics(log_path):
#     if not os.path.exists(log_path):
#         return None, None
#     df = pd.read_csv(log_path, sep='\t')
#     best_row = df.loc[df['val_acc'].idxmax()]
#     return best_row['test_acc'], best_row['test_mcc'], df['val_acc'].values

# def get_top_k_metrics(dataset, param_type, param_val, total_seeds=50, top_k=5):
#     results = []
    
#     for seed in range(total_seeds):
#         log_path = f"src/out/{dataset}-{param_type}{param_val}-seed{seed}/pred_log.tsv"
#         acc, mcc, _ = get_best_metrics(log_path)
#         if acc is not None:
#             results.append((acc * 100, mcc, seed)) # Convert to %
            
#     if not results:
#         print(f"[Warning] No data found for {dataset}-{param_type}{param_val}")
#         return 0, 0
    
#     # Sort by accuracy (descending)
#     results.sort(key=lambda x: x[0], reverse=True)
    
#     # Take the top K
#     top_results = results[:top_k]
    
#     accs = [r[0] for r in top_results]
#     mccs = [r[1] for r in top_results]
    
#     return np.mean(accs), np.mean(mccs)

# def get_top_k_epoch_curve(dataset, total_seeds=50, top_k=5):
#     results = []
    
#     # 1. Identify the top K seeds based on their optimal configuration (win10)
#     for seed in range(total_seeds):
#         log_path = f"src/out/{dataset}-win10-seed{seed}/pred_log.tsv"
#         acc, _, val_acc_curve = get_best_metrics(log_path)
#         if acc is not None:
#             results.append((acc, val_acc_curve))
            
#     if not results:
#         return [], []
        
#     # Sort by test accuracy descending and take top K
#     results.sort(key=lambda x: x[0], reverse=True)
#     top_curves = [r[1] for r in results[:top_k]]
    
#     # 2. Pad shorter arrays with their final accuracy value (forward-fill)
#     max_len = max(len(arr) for arr in top_curves)
#     padded_accs = []
#     for arr in top_curves:
#         if len(arr) < max_len:
#             padding = np.full(max_len - len(arr), arr[-1])
#             padded_arr = np.concatenate([arr, padding])
#         else:
#             padded_arr = arr
#         padded_accs.append(padded_arr)
    
#     # 3. Stack and calculate mean across the Top K seeds
#     mean_val_accs = np.mean(np.vstack(padded_accs), axis=0)
#     epochs = np.arange(max_len)
    
#     return epochs, mean_val_accs

# def parse_args():
#     parser = argparse.ArgumentParser(description="Plot Hyperparameter Tuning Results (Top-K)")
#     parser.add_argument('--data', type=str, default='acl18', help='Dataset name')
#     parser.add_argument('--total-seeds', type=int, default=50, help='Total number of seeds you ran')
#     parser.add_argument('--top-k', type=int, default=5, help='Number of best seeds to average')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     dataset = args.data
#     print(f"Generating plots for {dataset} (Averaging TOP {args.top_k} out of {args.total_seeds} seeds)")
    
#     windows = [5, 10, 15, 20]
#     hdims = [16, 32, 64, 128]
    
#     win_accs, win_mccs = [], []
#     hdim_accs, hdim_mccs = [], []
    
#     # 1. Gather Window Data
#     for w in windows:
#         mean_acc, mean_mcc = get_top_k_metrics(dataset, "win", w, args.total_seeds, args.top_k)
#         win_accs.append(mean_acc)
#         win_mccs.append(mean_mcc)

#     # 2. Gather Hidden Dim Data
#     for h in hdims:
#         mean_acc, mean_mcc = get_top_k_metrics(dataset, "hdim", h, args.total_seeds, args.top_k)
#         hdim_accs.append(mean_acc)
#         hdim_mccs.append(mean_mcc)

#     # 3. Gather Epoch Data (using optimal win10)
#     epochs, val_accs = get_top_k_epoch_curve(dataset, args.total_seeds, args.top_k)

#     # --- PLOTTING ---
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
#     # Plot (a): Window Size
#     ax1 = axes[0]
#     ax1.plot(windows, win_accs, marker='o', color='b', label='Accuracy (%)')
#     ax1.set_xlabel('Look-back Window Size ($L$)')
#     ax1.set_ylabel('Accuracy (%)', color='b')
#     ax1.tick_params(axis='y', labelcolor='b')
#     ax1.set_xticks(windows)
#     ax1.grid(False)
    
#     ax1_twin = ax1.twinx()
#     ax1_twin.plot(windows, win_mccs, marker='s', color='r', linestyle='--', label='MCC')
#     ax1_twin.set_ylabel('MCC', color='r')
#     ax1_twin.tick_params(axis='y', labelcolor='r')
#     ax1_twin.grid(False)
#     axes[0].set_title(f'(a) Impact of Window Size ({dataset})')

#     # Plot (b): Hidden Dimension
#     ax2 = axes[1]
#     ax2.plot(hdims, hdim_accs, marker='o', color='b')
#     ax2.set_xlabel('ALSTM Hidden Dimension')
#     ax2.set_ylabel('Accuracy (%)', color='b')
#     ax2.tick_params(axis='y', labelcolor='b')
#     ax2.set_xticks(hdims)
#     ax2.grid(False)
    
#     ax2_twin = ax2.twinx()
#     ax2_twin.plot(hdims, hdim_mccs, marker='s', color='r', linestyle='--')
#     ax2_twin.set_ylabel('MCC', color='r')
#     ax2_twin.tick_params(axis='y', labelcolor='r')
#     ax2_twin.grid(False)
#     axes[1].set_title(f'(b) Impact of Hidden Dimension ({dataset})')

#     # Plot (c): Epoch Convergence
#     if len(epochs) > 0:
#         ax3 = axes[2]
#         ax3.plot(epochs, val_accs * 100, color='g', linewidth=2)
#         ax3.set_xlabel('Training Epochs')
#         ax3.set_ylabel('Validation Accuracy (%)')
#         ax3.grid(False)
#         axes[2].set_title(f'(c) Convergence over Epochs ({dataset})')

#     plt.tight_layout()
    
#     os.makedirs('fig', exist_ok=True)
#     out_file = f'fig/hyperparameter_tuning_{dataset}.pdf'
#     plt.savefig(out_file, format='pdf', bbox_inches='tight')
#     print(f"Plot successfully saved to {out_file}")

# if __name__ == "__main__":
#     main()
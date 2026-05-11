import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Configuration
NOISE_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
BASE_DIR = "src/out"  # Update this to where your output folders are located
DATASET = "icdm22"    # Update if using a different dataset name
SEED = 0              # The seed you used for the run

def load_data(noise_ratio):
    """
    Constructs the path and loads the pickle file for a specific noise ratio.
    Expected path: src/out/{DATASET}-{SEED}-noise{noise_ratio}/noise_analysis_{noise_ratio}.pkl
    """
    # Note: Adjust the path construction to match your run.sh structure exactly
    # Based on your run.sh: OUT="out/${DATA}-${SEED}-noise${NOISE}"
    # folder_name = f"{DATASET}-"
    file_name = f"noise_analysis_{noise_ratio}.pkl"
    print("file_name", file_name)
    path = os.path.join(BASE_DIR, file_name)
    
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return None
        
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_noise_filtering_grid():
    # Set up the figure grid (e.g., 1 row, 4 columns)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    sns.set_style("whitegrid")
    
    for i, noise in enumerate(NOISE_RATIOS):
        ax = axes[i]
        data = load_data(noise)
        
        if data is None:
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue
            
        # Extract weights
        real_weights = np.array(data['real'])
        noise_weights = np.array(data['noise'])
        
        # Prepare DataFrame for Seaborn
        df_real = pd.DataFrame({'Weight': real_weights, 'Type': 'Real Tweets'})
        df_noise = pd.DataFrame({'Weight': noise_weights, 'Type': 'Noise Vectors'})
        df_plot = pd.concat([df_real, df_noise])
        
        # Plot Density (KDE)
        # We assume weights sum to 1, so they are usually small.
        # Log scale might help if the separation is extreme, but linear is standard.
        sns.kdeplot(
            data=df_plot, 
            x='Weight', 
            hue='Type', 
            fill=True, 
            common_norm=False, 
            palette={'Real Tweets': '#1f77b4', 'Noise Vectors': '#d62728'},
            alpha=0.3,
            ax=ax,
            linewidth=2
        )
        
        # Formatting
        ax.set_title(f"Noise Ratio = {noise}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Attention Weight", fontsize=12)
        if i == 0:
            ax.set_ylabel("Density", fontsize=12)
        else:
            ax.set_ylabel("")
        
        # Calculate separation metric (e.g., mean difference)
        mean_real = real_weights.mean()
        mean_noise = noise_weights.mean()
        ax.axvline(mean_real, color='#1f77b4', linestyle='--', alpha=0.6)
        ax.axvline(mean_noise, color='#d62728', linestyle='--', alpha=0.6)
        
        # Add annotation
        ax.text(0.95, 0.95, f"Δ Mean: {mean_real - mean_noise:.4f}", 
                transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    save_path = f"q2_noise_filtering_analysis_{DATASET}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_noise_filtering_grid()
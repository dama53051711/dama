import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = "cikm18-0"
# 1. Load Data
try:
    # Replace with your actual path
    with open(f"src/out/{data}/attn_weights_q2.pkl", "rb") as f:
        data = pickle.load(f)
        real_weights = data["real"]
        noise_weights = data["noise"]
except FileNotFoundError:
    # Synthetic data for demonstration
    print("Using synthetic data for demo...")
    # Real: Most near 0, but some reach 0.1 or 0.2 (The "Needle")
    real_weights = np.concatenate([
        np.random.exponential(scale=0.002, size=4000), # Irrelevant Real
        np.random.normal(loc=0.05, scale=0.01, size=100) # Relevant Real
    ])
    # Noise: Strictly near 0
    noise_weights = np.random.exponential(scale=0.001, size=4100)

# 2. Setup Plot
sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(8, 5))

# 3. Plot Histograms with LOG SCALE on Y-Axis
# This visualizes the "Tail" (High weights) which is otherwise invisible
sns.histplot(real_weights, color="#1f77b4", label="Real Tweets", 
             element="step", stat="density", common_norm=False, fill=True, alpha=0.3,
             log_scale=(False, True)) # <--- KEY CHANGE: Log Scale on Y

sns.histplot(noise_weights, color="#d62728", label="Injected Noise", 
             element="step", stat="density", common_norm=False, fill=True, alpha=0.3,
             log_scale=(False, True)) # <--- KEY CHANGE: Log Scale on Y

# 4. Formatting
plt.title("Attention Weight Distribution (Log Scale)", fontsize=16, weight='bold')
plt.xlabel("Attention Weight", fontsize=14)
plt.ylabel("Log Density", fontsize=14)
plt.legend(title="Source Type", fontsize=12)

# Zoom in to the interesting area (ignore the massive empty space on right if any)
# Adjust xlim based on your max real weight
max_real = np.max(real_weights)
plt.xlim(0, max_real * 1.1) 

# Add Annotation to explain the graph
plt.annotate('Relevant Tweets\n(High Attention)', 
             xy=(max_real * 0.5, 10), 
             xytext=(max_real * 0.6, 100),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12)

plt.tight_layout()
# plt.savefig("attention_histogram_log_q2.pdf", dpi=300)
plt.savefig(f"attention_histogram_log_q2_{data}.png", dpi=1000)
plt.show()
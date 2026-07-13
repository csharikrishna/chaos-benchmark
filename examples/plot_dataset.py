import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid", context="paper")

# Load data
df = pd.read_csv("chaos_benchmark_dataset_demo.csv")

# Create plots directory
os.makedirs("plots", exist_ok=True)

# 1. Lyapunov Exponent Distribution by System and Label
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x="system", y="lyapunov_exponent", hue="label", jitter=True, alpha=0.7, dodge=True)
plt.axhline(0, color='red', linestyle='--', alpha=0.5, label="Chaos Threshold (0)")
plt.yscale('symlog', linthresh=0.001)
plt.title("Lyapunov Exponent Distribution by Dynamical System")
plt.ylabel("Maximal Lyapunov Exponent (Symlog Scale)")
plt.legend(title="Regime")
plt.tight_layout()
plt.savefig("plots/lyapunov_distribution.png", dpi=300)
plt.close()

# 2. Feature Separation: Spectral Entropy vs Variance
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="spectral_entropy", y="variance", hue="label", style="system", alpha=0.8)
plt.yscale('log')
plt.title("Feature Separation: Spectral Entropy vs Variance")
plt.tight_layout()
plt.savefig("plots/feature_separation.png", dpi=300)
plt.close()

# 3. Forecast Horizon by System (for Chaotic regime only)
plt.figure(figsize=(8, 6))
chaotic_df = df[df['label'] == 'Chaotic'].copy()
sns.boxplot(data=chaotic_df, x="system", y="forecast_horizon", color="lightblue")
sns.swarmplot(data=chaotic_df, x="system", y="forecast_horizon", color="darkblue", alpha=0.6)
plt.title("Forecast Horizon in Chaotic Regimes")
plt.ylabel("Forecast Horizon (Simulation Steps)")
plt.tight_layout()
plt.savefig("plots/forecast_horizon.png", dpi=300)
plt.close()

print("Plots successfully generated in the 'plots/' directory!")

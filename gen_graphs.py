import matplotlib.pyplot as plt
import pandas as pd
import wandb

# 1. Initialize the W&B API and fetch the run
api = wandb.Api()
run = api.run("riley_h-icl/DaedalusProject/uhthli39")

print(f"Fetching metrics for run: {run.name}...")

# 2. Define the specific metrics
metrics = [
    "env/success_rate",
    "env/crash_rate",
    "env/episodic_reward",
    "loss/critic_1",
    "loss/alpha",
    "loss/actor",
]

# 3. Stream history into a pandas DataFrame
history_iter = run.scan_history(keys=metrics)
df = pd.DataFrame(history_iter)

if df.empty:
    raise ValueError(
        "The dataframe is empty. Double-check your run path or metric keys."
    )

# Clean up rows where all requested metrics are NaN
df = df.dropna(subset=[m for m in metrics if m in df.columns], how="all")

# FIX: Define the explicit X-axis data array cleanly
x_data = df["_step"] if "_step" in df.columns else df.index

x_data *= 100

# ==============================================================================
# GRAPH 1: Environment Performance Rates (Success vs. Crash)
# ==============================================================================
plt.figure(figsize=(10, 5))

if "env/success_rate" in df.columns:
    plt.plot(
        x_data,
        df["env/success_rate"],
        label="Success Rate",
        color="#2ca02c",
        linewidth=1.5,
    )
if "env/crash_rate" in df.columns:
    plt.plot(
        x_data,
        df["env/crash_rate"],
        label="Crash Rate",
        color="#d62728",
        linewidth=1.5,
    )

plt.title("Environment Success vs. Crash Rate Profiles", fontsize=14, pad=12)
plt.xlabel("Training Steps", fontsize=11)
plt.ylabel("Rate (Normalized / Percentage)", fontsize=11)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ==============================================================================
# GRAPH 2: Training Progress & Loss Landscape (2x2 Subplots)
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-Left: Episodic Reward
if "env/episodic_reward" in df.columns:
    axs[0, 0].plot(x_data, df["env/episodic_reward"], color="#1f77b4", alpha=0.8)
axs[0, 0].set_title("Episodic Reward Matrix", fontsize=12, weight="bold")
axs[0, 0].set_ylabel("Reward Value")

# Top-Right: Critic 1 Loss
if "loss/critic_1" in df.columns:
    axs[0, 1].plot(x_data, df["loss/critic_1"], color="#ff7f0e", alpha=0.8)
axs[0, 1].set_title("Critic 1 Value Loss ($L_Q$)", fontsize=12, weight="bold")
axs[0, 1].set_ylabel("MSE Loss")

# Bottom-Left: Actor Policy Loss
if "loss/actor" in df.columns:
    axs[1, 0].plot(x_data, df["loss/actor"], color="#9467bd", alpha=0.8)
axs[1, 0].set_title("Actor Policy Loss ($L_\\pi$)", fontsize=12, weight="bold")
axs[1, 0].set_ylabel("Policy Gradient Loss")

# Bottom-Right: Entropy Alpha Loss
if "loss/alpha" in df.columns:
    axs[1, 1].plot(x_data, df["loss/alpha"], color="#e377c2", alpha=0.8)
axs[1, 1].set_title(
    "Temperature Alpha Loss ($L_\\mu$)", fontsize=12, weight="bold"
)
axs[1, 1].set_ylabel("Entropy Alpha Loss")

# Global formatting across all subplots
for ax in axs.flat:
    ax.set_xlabel("Training Steps")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    f"Algorithmic Performance Dynamics: Run [{run.id}]",
    fontsize=16,
    weight="bold",
    y=0.98,
)
plt.tight_layout()

# Display plots
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_monitor_plots(log_dir):
    # save reward and step plot
    
    monitor_path = Path(log_dir) / "monitor.csv"
    
    if not log_dir:  # do nothing if log_dir is empty (eval calls)
        return
    if not monitor_path.exists():
        print(f"Path doesn't exists: {monitor_path}")
        return
    df = pd.read_csv(monitor_path, skiprows=1)

    # r = episode reward, l = episode length/steps
    df["timesteps"] = df["l"].cumsum()
    df["reward_ma"] = df["r"].rolling(20, min_periods=1).mean()
    df["steps_ma"] = df["l"].rolling(20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["r"], alpha=0.25, label="episode reward")
    plt.plot(df["timesteps"], df["reward_ma"], label="reward rolling mean (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(log_dir) / "reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["l"], alpha=0.25, label="episode steps")
    plt.plot(df["timesteps"], df["steps_ma"], label="steps rolling mean (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(log_dir) / "steps_curve.png", dpi=150)
    plt.close()

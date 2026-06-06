import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_monitor_plots(log_dir):
    # save reward and step plot
    
    if not log_dir:  # do nothing if log_dir is empty (eval calls)
        return
    
    log_dir = Path(log_dir)
    monitor_path = log_dir / "monitor.csv"

    if not monitor_path.exists():
        print(f"Path doesn't exists: {monitor_path}")
        return
    df = pd.read_csv(monitor_path, skiprows=1)
    if df.empty:
        return

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
    plt.savefig(log_dir / "reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["l"], alpha=0.25, label="episode steps")
    plt.plot(df["timesteps"], df["steps_ma"], label="steps rolling mean (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "steps_curve.png", dpi=150)
    plt.close()

    if "task_success" not in df.columns:
        print("No task_success column in monitor.csv; success-rate plot will be available after rerunning training.")
        return

    df["task_success"] = df["task_success"].astype(float)
    df["success_rate_ma"] = df["task_success"].rolling(20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["task_success"], alpha=0.25, label="episode success")
    plt.plot(df["timesteps"], df["success_rate_ma"], label="success rate rolling mean (20)")
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_dir / "success_rate_curve.png", dpi=150)
    plt.close()

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

def save_monitor_plots_by_target(log_dir, rolling_window=20):
    """
    Save reward/steps plots separated by target type.

    Requires:
        Monitor(..., info_keywords=("target",))

    Expected monitor.csv columns:
        r = reward
        l = episode length
        target = current target name
    """

    monitor_path = Path(log_dir) / "monitor.csv"

    if not log_dir:
        return

    if not monitor_path.exists():
        print(f"Path doesn't exist: {monitor_path}")
        return

    df = pd.read_csv(monitor_path, skiprows=1)

    if "target" not in df.columns:
        # print("No target column found in monitor.csv, calling default plotting")
        save_monitor_plots(log_dir)
        return

    # cumulative timesteps
    df["timesteps"] = df["l"].cumsum()

    targets = sorted(df["target"].dropna().unique())
    if len(targets) == 1 and targets[0] == "N/A":
        # print("Only N/A target found, calling default plotting")
        save_monitor_plots(log_dir)
        return
    
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    # reward Plot
    plt.figure(figsize=(10, 5))

    for i, target in enumerate(targets):
        target_df = df[df["target"] == target].copy()

        target_df["reward_ma"] = (
            target_df["r"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )

        plt.plot(
            target_df["timesteps"],
            target_df["r"],
            alpha=0.15,
            label=f"{target} reward",
            color=colors[i % len(colors)]
        )

        plt.plot(
            target_df["timesteps"],
            target_df["reward_ma"],
            linewidth=2,
            label=f"{target} reward MA({rolling_window})",
            color=colors[i % len(colors)]
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Reward by Target")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        Path(log_dir) / "reward_curve_by_target.png",
        dpi=150
    )

    plt.close()

    # steps Plot
    plt.figure(figsize=(10, 5))

    for i, target in enumerate(targets):
        target_df = df[df["target"] == target].copy()

        target_df["steps_ma"] = (
            target_df["l"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )

        plt.plot(
            target_df["timesteps"],
            target_df["l"],
            alpha=0.15,
            label=f"{target} steps",
            color=colors[i % len(colors)]
        )

        plt.plot(
            target_df["timesteps"],
            target_df["steps_ma"],
            linewidth=2,
            label=f"{target} steps MA({rolling_window})",
            color=colors[i % len(colors)]
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Steps")
    plt.title("Episode Steps by Target")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        Path(log_dir) / "steps_curve_by_target.png",
        dpi=150
    )

    plt.close()

    # success rate plot
    plt.figure(figsize=(10, 5))

    for i, target in enumerate(targets):
        target_df = df[df["target"] == target].copy()

        if "task_success" not in target_df.columns:
            continue

        target_df["task_success"] = target_df["task_success"].astype(float)

        target_df["success_rate_ma"] = (
            target_df["task_success"]
            .rolling(rolling_window, min_periods=1)
            .mean()
        )

        plt.plot(
            target_df["timesteps"],
            target_df["task_success"],
            alpha=0.15,
            label=f"{target} success",
            color=colors[i % len(colors)]
        )

        plt.plot(
            target_df["timesteps"],
            target_df["success_rate_ma"],
            linewidth=2,
            label=f"{target} success rate MA({rolling_window})",
            color=colors[i % len(colors)]
        )
    
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Target")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(Path(log_dir) / "success_rate_curve_by_target.png", dpi=150)

    plt.close()
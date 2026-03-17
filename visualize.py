"""
Visualization and analysis tools for ATC RL model evaluation.

Usage:
    python visualize.py --model results/best_model/best_model.zip

Commands:
    python visualize.py training          # Plot training curves from eval logs
    python visualize.py trajectory        # Plot aircraft trajectories for one episode
    python visualize.py evaluate          # Run evaluation and plot metrics
    python visualize.py compare           # Compare multiple checkpoints
 
    To run the visualization in a specific run directory, use the --run-dir argument:

    python visualize.py compare --run-dir results/test_03drift_40conflict
    python visualize.py evaluate --run-dir results/test_03drift_40conflict
    python visualize.py training --run-dir results/test_03drift_40conflict
    python visualize.py trajectory --run-dir results/test_03drift_40conflict

"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from stable_baselines3 import SAC

from atcenv.env import Environment, NUMBER_INTRUDERS_STATE
from atcenv.sb3_wrapper import ATCEnvWrapper, ACTION_FREQUENCY

# Normalization (must match sb3_wrapper.py)
INTRUDER_DIST_NORM = 50000.0
INTRUDER_POS_NORM = 13000.0
TARGET_DIST_NORM = 200000.0


def normalize_obs(raw_obs):
    """Normalize a single agent's raw observation (same as wrapper)."""
    obs = np.array(raw_obs, dtype=np.float32)
    n = NUMBER_INTRUDERS_STATE
    obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
    obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM
    obs[4*n:5*n] = obs[4*n:5*n] / np.pi
    obs[5*n]     = (obs[5*n] - 230.0) / 30.0
    obs[5*n+1]   = (obs[5*n+1] - 230.0) / 30.0
    obs[5*n+2]   = (obs[5*n+2] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


# ────────────────────────── TRAINING CURVES ──────────────────────────

def plot_training_curves(eval_log_path="results/eval_logs/evaluations.npz",
                         save_path="results/plots/training_curves.png"):
    """Plot reward and episode length curves from EvalCallback logs."""
    data = np.load(eval_log_path)
    timesteps = data["timesteps"]
    results = data["results"]      # (n_evals, n_episodes)
    ep_lengths = data["ep_lengths"]

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)
    mean_length = ep_lengths.mean(axis=1)
    std_length = ep_lengths.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve
    ax = axes[0]
    ax.plot(timesteps, mean_reward, color="#2196F3", linewidth=2, label="Mean reward")
    ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                    alpha=0.2, color="#2196F3")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Training Reward Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Episode length curve
    ax = axes[1]
    ax.plot(timesteps, mean_length, color="#4CAF50", linewidth=2, label="Mean ep length")
    ax.fill_between(timesteps, mean_length - std_length, mean_length + std_length,
                    alpha=0.2, color="#4CAF50")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Length")
    ax.set_title("Episode Length Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.show()


# ────────────────────────── TRAJECTORY PLOT ──────────────────────────

def record_episode(model, num_flights=5, deploy_all=True):
    """Run one episode and record all aircraft trajectories."""
    env = Environment(num_flights=num_flights)
    raw_obs_list = env.reset(num_flights)

    # Record initial positions and targets
    trajectories = {i: {"x": [], "y": [], "conflicts": []} for i in range(num_flights)}
    targets = {}
    for i, f in enumerate(env.flights):
        targets[i] = (f.target.x, f.target.y)

    done = False
    step = 0
    total_conflicts = 0

    while not done:
        n_active = len(env.flights) - len(env.done)

        # Get actions
        actions = np.zeros((n_active, 2), dtype=np.float32)
        if deploy_all:
            agent_idx = 0
            for i in range(num_flights):
                if i not in env.done:
                    obs = normalize_obs(raw_obs_list[agent_idx])
                    action, _ = model.predict(obs, deterministic=True)
                    actions[agent_idx] = action
                    agent_idx += 1

        # Record positions before stepping
        for i, f in enumerate(env.flights):
            trajectories[i]["x"].append(f.position.x)
            trajectories[i]["y"].append(f.position.y)
            trajectories[i]["conflicts"].append(i in env.conflicts)

        # Step with ACTION_FREQUENCY
        for _ in range(ACTION_FREQUENCY):
            raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
            if done_t or done_e:
                done = True
                break
            actions = np.zeros((len(env.flights) - len(env.done), 2), dtype=np.float32)

        total_conflicts += len(env.conflicts)
        step += 1

    # Record final positions
    for i, f in enumerate(env.flights):
        trajectories[i]["x"].append(f.position.x)
        trajectories[i]["y"].append(f.position.y)
        trajectories[i]["conflicts"].append(i in env.conflicts)

    env.close()

    # Get airspace polygons for plotting
    airspace_coords = list(env.airspace.polygon.exterior.coords)
    restricted_airspace_coords = list(env.restricted_airspace.polygon.exterior.coords) if env.restricted_airspace else []

    return trajectories, targets, airspace_coords, restricted_airspace_coords, total_conflicts


def plot_trajectories(model_path, num_flights=5, deploy_all=True,
                      save_path="results/plots/trajectories.png"):
    """Plot aircraft trajectories for one episode."""
    model = SAC.load(model_path)
    trajectories, targets, airspace, restricted_airspace, total_conflicts = record_episode(
        model, num_flights, deploy_all
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.Set1(np.linspace(0, 1, num_flights))

    # Draw airspace boundary
    airspace_x = [p[0]/1000 for p in airspace]
    airspace_y = [p[1]/1000 for p in airspace]
    ax.plot(airspace_x, airspace_y, "k--", alpha=0.3, linewidth=1, label="Airspace")

    # Draw restricted airspace boundary
    if restricted_airspace:
        restricted_x = [p[0]/1000 for p in restricted_airspace]
        restricted_y = [p[1]/1000 for p in restricted_airspace]
        ax.plot(restricted_x, restricted_y, "g-", alpha=0.6, linewidth=2, label="Restricted Airspace")

    for i in range(num_flights):
        traj = trajectories[i]
        x = np.array(traj["x"]) / 1000  # Convert to km
        y = np.array(traj["y"]) / 1000
        conflicts = traj["conflicts"]

        # Plot trajectory with color indicating conflicts
        for j in range(len(x) - 1):
            color = "red" if conflicts[j] else colors[i]
            linewidth = 3 if conflicts[j] else 1.5
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], color=color, linewidth=linewidth)

        # Start position (circle)
        ax.plot(x[0], y[0], "o", color=colors[i], markersize=10, zorder=5)
        ax.annotate(f"A{i}", (x[0], y[0]), fontsize=8, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 8),
                    textcoords="offset points")

        # Target position (star)
        tx, ty = targets[i][0]/1000, targets[i][1]/1000
        ax.plot(tx, ty, "*", color=colors[i], markersize=15, zorder=5)

    # Legend
    start_patch = mpatches.Patch(color="gray", label="● Start")
    target_patch = mpatches.Patch(color="gray", label="★ Target")
    conflict_patch = mpatches.Patch(color="red", label="— Conflict")
    ax.legend(handles=[start_patch, target_patch, conflict_patch], loc="upper right")

    mode = "All agents" if deploy_all else "Single actor"
    ax.set_title(f"Aircraft Trajectories ({mode}, {num_flights} flights, "
                 f"{total_conflicts} conflicts)", fontsize=13)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory plot to {save_path}")
    plt.show()


# ────────────────────────── EVALUATION METRICS ──────────────────────────

def run_evaluation(model_path, n_episodes=30, num_flights=5, deploy_all=True):
    """Run evaluation and return per-episode metrics."""
    model = SAC.load(model_path)
    env = Environment(num_flights=num_flights)

    metrics = {
        "conflicts": [],
        "targets_reached": [],
        "episode_length": [],
        "total_drift": [],
    }

    for ep in range(n_episodes):
        raw_obs_list = env.reset(num_flights)
        done = False
        ep_conflicts = 0
        ep_drift = 0.0
        step = 0

        while not done:
            n_active = len(env.flights) - len(env.done)
            actions = np.zeros((n_active, 2), dtype=np.float32)

            if deploy_all:
                agent_idx = 0
                for i in range(num_flights):
                    if i not in env.done:
                        obs = normalize_obs(raw_obs_list[agent_idx])
                        action, _ = model.predict(obs, deterministic=True)
                        actions[agent_idx] = action
                        agent_idx += 1

            for _ in range(ACTION_FREQUENCY):
                raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
                if done_t or done_e:
                    done = True
                    break
                actions = np.zeros((len(env.flights) - len(env.done), 2), dtype=np.float32)

            ep_conflicts += len(env.conflicts)
            # Sum absolute drift for all active agents
            for i, f in enumerate(env.flights):
                if i not in env.done:
                    ep_drift += abs(f.drift)
            step += 1

        metrics["conflicts"].append(ep_conflicts)
        metrics["targets_reached"].append(len(env.done))
        metrics["episode_length"].append(step)
        metrics["total_drift"].append(ep_drift)

    env.close()
    return metrics


def plot_evaluation(model_path, n_episodes=30, num_flights=5,
                    save_path="results/plots/evaluation.png"):
    """Run evaluation and plot summary metrics."""
    print(f"Running {n_episodes} episodes...")
    metrics = run_evaluation(model_path, n_episodes, num_flights, deploy_all=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Conflicts per episode
    ax = axes[0, 0]
    ax.bar(range(len(metrics["conflicts"])), metrics["conflicts"], color="#F44336", alpha=0.7)
    ax.axhline(np.mean(metrics["conflicts"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['conflicts']):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Conflicts")
    ax.set_title("Conflicts per Episode")
    ax.legend()

    # Targets reached
    ax = axes[0, 1]
    ax.bar(range(len(metrics["targets_reached"])), metrics["targets_reached"],
           color="#4CAF50", alpha=0.7)
    ax.axhline(num_flights, color="black", linestyle="--", label=f"Max: {num_flights}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Targets Reached")
    ax.set_title("Targets Reached per Episode")
    ax.set_ylim(0, num_flights + 1)
    ax.legend()

    # Episode length distribution
    ax = axes[1, 0]
    ax.hist(metrics["episode_length"], bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Episode Length (RL steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")

    # Average drift per episode
    ax = axes[1, 1]
    ax.bar(range(len(metrics["total_drift"])), metrics["total_drift"], color="#FF9800", alpha=0.7)
    ax.axhline(np.mean(metrics["total_drift"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['total_drift']):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Drift (rad)")
    ax.set_title("Cumulative Drift per Episode")
    ax.legend()

    conflict_free = sum(1 for c in metrics["conflicts"] if c == 0)
    fig.suptitle(f"Evaluation Summary — {conflict_free}/{n_episodes} conflict-free episodes "
                 f"({100*conflict_free/n_episodes:.0f}%)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation plot to {save_path}")
    plt.show()


# ────────────────────────── CHECKPOINT COMPARISON ──────────────────────────

def compare_checkpoints(checkpoint_dir="results/checkpoints/",
                        n_episodes=20, num_flights=5,
                        save_path="results/plots/checkpoint_comparison.png"):
    """Compare performance across training checkpoints."""
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")
    ], key=lambda f: int(f.split("_")[-2]))

    steps = []
    mean_conflicts = []
    mean_targets = []
    conflict_free_pct = []

    for ckpt in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        step_num = int(ckpt.split("_")[-2])
        steps.append(step_num)
        print(f"Evaluating {ckpt} ({step_num} steps)...")
        metrics = run_evaluation(ckpt_path, n_episodes, num_flights, deploy_all=True)
        mean_conflicts.append(np.mean(metrics["conflicts"]))
        mean_targets.append(np.mean(metrics["targets_reached"]))
        cf = sum(1 for c in metrics["conflicts"] if c == 0)
        conflict_free_pct.append(100 * cf / n_episodes)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(steps, mean_conflicts, "o-", color="#F44336", linewidth=2, markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Conflicts / Episode")
    ax.set_title("Conflict Rate vs Training")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, mean_targets, "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax.axhline(num_flights, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Targets Reached")
    ax.set_title("Target Completion vs Training")
    ax.set_ylim(0, num_flights + 1)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, conflict_free_pct, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Conflict-Free Episodes (%)")
    ax.set_title("Safety Rate vs Training")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {save_path}")
    plt.show()


# ────────────────────────── MAIN ──────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATC Model Visualization Tools")
    parser.add_argument("command", choices=["training", "trajectory", "evaluate", "compare"],
                        help="Which visualization to run")
    parser.add_argument("--run-dir", type=str, default="results",
                        help="The results directory to analyze (e.g., results/test_03drift_40conflict)")
    parser.add_argument("--model-name", type=str, default="best_model/best_model.zip",
                        help="Which model inside the run-dir to evaluate")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--num-flights", type=int, default=5)
    args = parser.parse_args()

    # Construct the full paths based on the run-dir
    model_path = os.path.join(args.run_dir, args.model_name)
    eval_log_path = os.path.join(args.run_dir, "eval_logs", "evaluations.npz")
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")

    if args.command == "training":
        plot_training_curves(eval_log_path=eval_log_path, 
                             save_path=os.path.join(args.run_dir, "plots", "training_curves.png"))

    elif args.command == "trajectory":
        plot_trajectories(model_path, args.num_flights, deploy_all=True,
                          save_path=os.path.join(args.run_dir, "plots", "trajectories.png"))

    elif args.command == "evaluate":
        plot_evaluation(model_path, args.episodes, args.num_flights,
                        save_path=os.path.join(args.run_dir, "plots", "evaluation.png"))

    elif args.command == "compare":
        compare_checkpoints(checkpoint_dir=checkpoint_dir, 
                            n_episodes=args.episodes, 
                            num_flights=args.num_flights,
                            save_path=os.path.join(args.run_dir, "plots", "checkpoint_comparison.png"))

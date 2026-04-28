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
    python visualize.py evaluate --run-dir results/bigger_network_finetune1 --workers 8 --episodes 500
    python visualize.py training --run-dir results/expanded_obs_matrix_4
    python visualize.py trajectory --run-dir results/bigger_network_finetune1
    python visualize.py compare --run-dir results/alert_shared_reward_ALL_AGENTS
    python visualize.py evaluate --run-dir results/minimal_reward_ALL_AGENTS
    python visualize.py training --run-dir results/test_03drift_40conflict
    python visualize.py trajectory --run-dir results/minimal_reward_ALL_AGENTS
    python visualize.py evaluate --run-dir results/05drift_08conflict_1.5target_02proximity_ALL_AGENTS --episodes 100 --workers 8
    python visualize.py compare --run-dir results/05drift_06conflict_01target_02proximity_ALL_AGENTS --workers 8
    python visualize.py training --run-dir results/<run> --workers 8
    python visualize.py trajectory --run-dir results/rel_velocity_obs --workers 8
    python visualize.py trajectory --run-dir results/basic_policy --workers 8

    python visualize.py evaluate --run-dir results/4_intruders_unlocked_physics --no-random-heading
    python visualize.py evaluate --run-dir results/minimal_reward_ALL_AGENTS --no-random-heading --workers 8
    python visualize.py evaluate --run-dir results/fine_5_steps_airspaces_v2 --no-random-heading --workers 8 --episodes 100
    
    python visualize.py compare --run-dir results/minimal_reward_ALL_AGENTS --no-random-heading

"""
import argparse
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from stable_baselines3 import SAC

from atcenv.env import Environment, NUMBER_INTRUDERS_STATE
from atcenv.sb3_wrapper import ATCEnvWrapper, ACTION_FREQUENCY

# Normalization (must match sb3_wrapper.py)
INTRUDER_DIST_NORM = 50000.0
INTRUDER_POS_NORM = 13000.0
TARGET_DIST_NORM = 200000.0
SPEED_NORM = 300.0


def normalize_obs(raw_obs):
    """Normalize a single agent's raw observation (same as wrapper)."""
    obs = np.array(raw_obs, dtype=np.float32)
    n = NUMBER_INTRUDERS_STATE
    obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
    obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM
    obs[4*n:5*n] = obs[4*n:5*n] / np.pi
    obs[5*n:6*n] = obs[5*n:6*n] / SPEED_NORM
    obs[6*n:7*n] = obs[6*n:7*n] / SPEED_NORM
    obs[7*n]     = (obs[7*n] - 230.0) / 30.0
    obs[7*n+1]   = (obs[7*n+1] - 230.0) / 30.0
    obs[7*n+2]   = (obs[7*n+2] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)
    
    # Restricted airspace flags (already in [0, 1] range)
    # obs[7*n+5] and obs[7*n+6] - no normalization needed
    
    # Closest 1 vertex of restricted airspace (distance, dx, dy)
    point_start = 7 * n + 7
    if point_start < len(obs):
        # Normalize Distance
        obs[point_start] = (obs[point_start] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
        # Normalize dx
        if point_start + 1 < len(obs):
            obs[point_start + 1] = obs[point_start + 1] / INTRUDER_POS_NORM
        # Normalize dy
        if point_start + 2 < len(obs):
            obs[point_start + 2] = obs[point_start + 2] / INTRUDER_POS_NORM
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

def record_episode(model, num_flights=5, deploy_all=True, random_heading=True):
    """Run one episode and record all aircraft trajectories."""
    env = Environment(num_flights=num_flights, random_init_heading=random_heading)
    raw_obs_list = env.reset(num_flights)

    # Record initial positions and targets
    trajectories = {i: {"x": [], "y": [], "conflicts": [], "restricted_intrusions": []} for i in range(num_flights)}
    targets = {}
    for i, f in enumerate(env.flights):
        targets[i] = (f.target.x, f.target.y)

    done = False
    step = 0
    total_conflicts = 0
    total_restricted_intrusions = 0

    while not done:
        # Get actions
        current_actions = {}
        if deploy_all:
            agent_idx = 0
            for i in range(num_flights):
                if i not in env.done:
                    obs = normalize_obs(raw_obs_list[agent_idx])
                    action, _ = model.predict(obs, deterministic=True)
                    current_actions[i] = action
                    agent_idx += 1

        # Record positions before stepping
        for i, f in enumerate(env.flights):
            trajectories[i]["x"].append(f.position.x)
            trajectories[i]["y"].append(f.position.y)
            trajectories[i]["conflicts"].append(i in env.conflicts)
            trajectories[i]["restricted_intrusions"].append(i in env.restricted_airspace_intrusions)

        # Step with ACTION_FREQUENCY
        for _ in range(ACTION_FREQUENCY):
            active_indices = [i for i in range(num_flights) if i not in env.done]
            actions = np.zeros((len(active_indices), 2), dtype=np.float32)
            for idx, agent_num in enumerate(active_indices):
                actions[idx] = current_actions.get(agent_num, np.array([0.0, 0.0], dtype=np.float32))

            raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
            if done_t or done_e:
                done = True
                break

        total_conflicts += len(env.conflicts)
        total_restricted_intrusions += len(env.restricted_airspace_intrusions)
        step += 1

    # Record final positions
    for i, f in enumerate(env.flights):
        trajectories[i]["x"].append(f.position.x)
        trajectories[i]["y"].append(f.position.y)
        trajectories[i]["conflicts"].append(i in env.conflicts)
        trajectories[i]["restricted_intrusions"].append(i in env.restricted_airspace_intrusions)

    env.close()

    # Get airspace polygons for plotting
    airspace_coords = list(env.airspace.polygon.exterior.coords)
    restricted_airspace_coords = list(env.restricted_airspace.polygon.exterior.coords) if env.restricted_airspace else []

    return trajectories, targets, airspace_coords, restricted_airspace_coords, total_conflicts, total_restricted_intrusions


def plot_trajectories(model_path, num_flights=5, deploy_all=True,
                      save_path="results/plots/trajectories.png", random_heading=True):
    """Plot aircraft trajectories for one episode."""
    model = SAC.load(model_path)
    trajectories, targets, airspace, restricted_airspace, total_conflicts, total_restricted_intrusions = record_episode(
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
        restricted_intrusions = traj["restricted_intrusions"]

        # Plot trajectory with color indicating conflicts and restricted airspace intrusions
        for j in range(len(x) - 1):
            if conflicts[j]:
                color = "red"
                linewidth = 3
            elif restricted_intrusions[j]:
                color = "yellow"
                linewidth = 2
            else:
                color = colors[i]
                linewidth = 1.5
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
    restricted_patch = mpatches.Patch(color="yellow", label="— Restricted Intrusion")
    ax.legend(handles=[start_patch, target_patch, conflict_patch, restricted_patch], loc="upper right")

    mode = "All agents" if deploy_all else "Single actor"
    ax.set_title(f"Aircraft Trajectories ({mode}, {num_flights} flights, "
                 f"{total_conflicts} conflicts, {total_restricted_intrusions} restricted intrusions)", fontsize=13)
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

def _eval_episodes_worker(model_path, episode_indices, num_flights, deploy_all, random_heading=True):
    """Worker function that runs a batch of episodes (used by ProcessPoolExecutor)."""
    model = SAC.load(model_path)
    env = Environment(num_flights=num_flights, random_init_heading=random_heading)

    local_metrics = {
        "conflicts": [],
        "restricted_intrusions": [],
        "targets_reached": [],
        "episode_length": [],
        "total_drift": [],
        "restricted_intrusions": [],
    }

    for ep in episode_indices:
        raw_obs_list = env.reset(num_flights)
        done = False
        ep_conflicts = 0
        ep_restricted_intrusions = 0
        ep_drift = 0.0
        ep_restricted_intrusions = 0
        step = 0

        while not done:
            current_actions = {}
            if deploy_all:
                agent_idx = 0
                for i in range(num_flights):
                    if i not in env.done:
                        obs = normalize_obs(raw_obs_list[agent_idx])
                        action, _ = model.predict(obs, deterministic=True)
                        current_actions[i] = action
                        agent_idx += 1

            for _ in range(ACTION_FREQUENCY):
                active_indices = [i for i in range(num_flights) if i not in env.done]
                actions = np.zeros((len(active_indices), 2), dtype=np.float32)
                for idx, agent_num in enumerate(active_indices):
                    actions[idx] = current_actions.get(agent_num, np.array([0.0, 0.0], dtype=np.float32))

                raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
                if done_t or done_e:
                    done = True
                    break

            ep_conflicts += len(env.conflicts)
            ep_restricted_intrusions += len(env.restricted_airspace_intrusions)
            for i, f in enumerate(env.flights):
                if i not in env.done:
                    ep_drift += abs(f.drift)
            step += 1

        local_metrics["conflicts"].append(ep_conflicts)
        local_metrics["restricted_intrusions"].append(ep_restricted_intrusions)
        local_metrics["targets_reached"].append(len(env.done))
        local_metrics["episode_length"].append(step)
        local_metrics["total_drift"].append(ep_drift)
        local_metrics["restricted_intrusions"].append(ep_restricted_intrusions)

    env.close()
    return local_metrics


def run_evaluation(model_path, n_episodes=30, num_flights=5, deploy_all=True, workers=1, random_heading=True):
    """Run evaluation and return per-episode metrics (parallelized across workers)."""
    if workers <= 1:
        # Single-process fallback
        return _eval_episodes_worker(model_path, list(range(n_episodes)), num_flights, deploy_all, random_heading)

    # Split episodes across workers
    episode_batches = [[] for _ in range(workers)]
    for ep in range(n_episodes):
        episode_batches[ep % workers].append(ep)
    # Remove empty batches
    episode_batches = [b for b in episode_batches if b]

    metrics = {
        "conflicts": [],
        "restricted_intrusions": [],
        "targets_reached": [],
        "episode_length": [],
        "total_drift": [],
        "restricted_intrusions": [],
    }

    with ProcessPoolExecutor(max_workers=len(episode_batches)) as executor:
        futures = [
            executor.submit(_eval_episodes_worker, model_path, batch, num_flights, deploy_all, random_heading)
            for batch in episode_batches
        ]
        for future in futures:
            result = future.result()
            for key in metrics:
                metrics[key].extend(result[key])

    return metrics


def plot_evaluation(model_path, n_episodes=30, num_flights=5,
                    save_path="results/plots/evaluation.png", workers=1, random_heading=True):
    """Run evaluation and plot summary metrics."""
    print(f"Running {n_episodes} episodes across {workers} worker(s)...")
    metrics = run_evaluation(model_path, n_episodes, num_flights, deploy_all=True, workers=workers, random_heading=random_heading)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Conflicts per episode
    ax = axes[0, 0]
    ax.bar(range(len(metrics["conflicts"])), metrics["conflicts"], color="#F44336", alpha=0.7)
    ax.axhline(np.mean(metrics["conflicts"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['conflicts']):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Conflicts")
    ax.set_title("Conflicts per Episode")
    ax.legend()

    # Targets reached (with mean and max)
    ax = axes[0, 1]
    ax.bar(range(len(metrics["targets_reached"])), metrics["targets_reached"],
           color="#4CAF50", alpha=0.7)
    mean_targets = np.mean(metrics["targets_reached"])
    ax.axhline(mean_targets, color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {mean_targets:.1f}")
    ax.axhline(num_flights, color="black", linestyle="--", linewidth=2, 
               label=f"Max: {num_flights}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Targets Reached")
    ax.set_title("Targets Reached per Episode")
    ax.set_ylim(0, num_flights + 1)
    ax.legend()

    # Restricted Intrusions per episode
    ax = axes[0, 2]
    ax.bar(range(len(metrics["restricted_intrusions"])), metrics["restricted_intrusions"], color="#8E24AA", alpha=0.7)
    ax.axhline(np.mean(metrics["restricted_intrusions"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['restricted_intrusions']):.2f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Intrusions")
    ax.set_title("Restricted Area Violations")
    ax.legend()

    # Episode length distribution
    ax = axes[0, 2]
    ax.hist(metrics["episode_length"], bins=20, color="#2196F3", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Episode Length (RL steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")

    # Average drift per episode
    ax = axes[1, 0]
    ax.bar(range(len(metrics["total_drift"])), metrics["total_drift"], color="#FF9800", alpha=0.7)
    ax.axhline(np.mean(metrics["total_drift"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['total_drift']):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Drift (rad)")
    ax.set_title("Cumulative Drift per Episode")
    ax.legend()
    
    # Hide the unused 6th subplot
    axes[1, 2].axis('off')

    # Restricted airspace intrusions per episode
    ax = axes[1, 1]
    ax.bar(range(len(metrics["restricted_intrusions"])), metrics["restricted_intrusions"], color="#9C27B0", alpha=0.7)
    ax.axhline(np.mean(metrics["restricted_intrusions"]), color="black", linestyle="--",
               label=f"Mean: {np.mean(metrics['restricted_intrusions']):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Restricted Intrusions")
    ax.set_title("Restricted Airspace Intrusions per Episode")
    ax.legend()

    # Summary statistics text
    ax = axes[1, 2]
    ax.axis("off")
    conflict_free = sum(1 for c in metrics["conflicts"] if c == 0)
    restricted_free = sum(1 for r in metrics["restricted_intrusions"] if r == 0)
    
    # Text terminal output
    print("\n" + "="*50)
    print(f" EVALUATION SUMMARY ({n_episodes} EPISODES)")
    print("="*50)
    print(f" Conflict-Free Episodes:      {conflict_free}/{n_episodes} ({100*conflict_free/n_episodes:.1f}%)")
    print(f" Restricted-Free Episodes:    {restricted_free}/{n_episodes} ({100*restricted_free/n_episodes:.1f}%)")
    print("-" * 50)
    print(f" Mean Conflicts/Episode:      {np.mean(metrics['conflicts']):.2f}")
    print(f" Mean Restricted Vulns/Ep:    {np.mean(metrics['restricted_intrusions']):.2f}")
    print(f" Mean Targets Reached/Ep:     {np.mean(metrics['targets_reached']):.2f} / {num_flights}")
    print(f" Mean Cumulative Drift/Ep:    {np.mean(metrics['total_drift']):.2f} rad")
    print("="*50 + "\n")

    # Matplotlib summary text block
    summary_text = (
        f"Evaluation Summary\n"
        f"{'═' * 30}\n"
        f"Episodes: {n_episodes}\n"
        f"Num Flights: {num_flights}\n\n"
        f"Conflicts:\n"
        f"  Mean: {np.mean(metrics['conflicts']):.2f}\n"
        f"  Conflict-free: {conflict_free}/{n_episodes} ({100*conflict_free/n_episodes:.1f}%)\n\n"
        f"Targets Reached:\n"
        f"  Mean: {mean_targets:.2f}/{num_flights}\n"
        f"  Max: {np.max(metrics['targets_reached']):.0f}\n\n"
        f"Restricted Intrusions:\n"
        f"  Mean: {np.mean(metrics['restricted_intrusions']):.2f}\n"
        f"  Intrusion-free: {restricted_free}/{n_episodes} ({100*restricted_free/n_episodes:.1f}%)\n\n"
        f"Drift:\n"
        f"  Mean: {np.mean(metrics['total_drift']):.2f} rad"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Evaluation Summary — Conflicts: {np.mean(metrics['conflicts']):.2f}/ep | "
                 f"Restricted: {np.mean(metrics['restricted_intrusions']):.2f}/ep",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation plot to {save_path}")
    plt.show()


# ────────────────────────── CHECKPOINT COMPARISON ──────────────────────────

def _eval_checkpoint_worker(ckpt_path, step_num, n_episodes, num_flights):
    """Worker to evaluate a single checkpoint (used by ProcessPoolExecutor)."""
    metrics = _eval_episodes_worker(ckpt_path, list(range(n_episodes)), num_flights, deploy_all=True, random_heading=True)
    mean_c = np.mean(metrics["conflicts"])
    mean_r = np.mean(metrics["restricted_intrusions"])
    mean_t = np.mean(metrics["targets_reached"])
    cf = sum(1 for c in metrics["conflicts"] if c == 0)
    cf_pct = 100 * cf / n_episodes
    return step_num, mean_c, mean_r, mean_t, cf_pct


def compare_checkpoints(checkpoint_dir="results/checkpoints/",
                        n_episodes=20, num_flights=5,
                        save_path="results/plots/checkpoint_comparison.png",
                        workers=1):
    """Compare performance across training checkpoints."""
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")
    ], key=lambda f: int(f.split("_")[-2]))

    steps = []
    mean_conflicts = []
    mean_restricted = []
    mean_targets = []
    conflict_free_pct = []

    if workers <= 1:
        # Sequential fallback
        for ckpt in checkpoints:
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            step_num = int(ckpt.split("_")[-2])
            print(f"Evaluating {ckpt} ({step_num} steps)...")
            metrics = _eval_episodes_worker(ckpt_path, list(range(n_episodes)), num_flights, deploy_all=True)
            steps.append(step_num)
            mean_conflicts.append(np.mean(metrics["conflicts"]))
            mean_restricted.append(np.mean(metrics["restricted_intrusions"]))
            mean_targets.append(np.mean(metrics["targets_reached"]))
            cf = sum(1 for c in metrics["conflicts"] if c == 0)
            conflict_free_pct.append(100 * cf / n_episodes)
    else:
        print(f"Evaluating {len(checkpoints)} checkpoints across {workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for ckpt in checkpoints:
                ckpt_path = os.path.join(checkpoint_dir, ckpt)
                step_num = int(ckpt.split("_")[-2])
                futures.append(executor.submit(_eval_checkpoint_worker, ckpt_path, step_num, n_episodes, num_flights))
            for future in futures:
                step_num, mc, mr, mt, cfp = future.result()
                steps.append(step_num)
                mean_conflicts.append(mc)
                mean_restricted.append(mr)
                mean_targets.append(mt)
                conflict_free_pct.append(cfp)
        # Sort by step number since parallel results may arrive out of order
        order = np.argsort(steps)
        steps = [steps[i] for i in order]
        mean_conflicts = [mean_conflicts[i] for i in order]
        mean_restricted = [mean_restricted[i] for i in order]
        mean_targets = [mean_targets[i] for i in order]
        conflict_free_pct = [conflict_free_pct[i] for i in order]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    ax = axes[0]
    ax.plot(steps, mean_conflicts, "o-", color="#F44336", linewidth=2, markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Conflicts / Episode")
    ax.set_title("Conflict Rate vs Training")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(steps, mean_restricted, "o-", color="#8E24AA", linewidth=2, markersize=6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Restricted Intrusions / Ep")
    ax.set_title("Restricted Area Violations vs Training")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, mean_targets, "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax.axhline(num_flights, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Targets Reached")
    ax.set_title("Target Completion vs Training")
    ax.set_ylim(0, num_flights + 1)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-flights", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Number of parallel worker processes for evaluation")
    parser.add_argument("--no-random-heading", action="store_true", 
                        help="Evaluate on perfectly straight initial headings instead of randomized ones.")
    args = parser.parse_args()
    
    random_heading_val = not args.no_random_heading

    # Construct the full paths based on the run-dir
    model_path = os.path.join(args.run_dir, args.model_name)
    eval_log_path = os.path.join(args.run_dir, "eval_logs", "evaluations.npz")
    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")

    if args.command == "training":
        plot_training_curves(eval_log_path=eval_log_path, 
                             save_path=os.path.join(args.run_dir, "plots", "training_curves.png"))

    elif args.command == "trajectory":
        plot_trajectories(model_path, args.num_flights, deploy_all=True,
                          save_path=os.path.join(args.run_dir, "plots", "trajectories.png"),
                          random_heading=random_heading_val)

    elif args.command == "evaluate":
        plot_evaluation(model_path, args.episodes, args.num_flights,
                        save_path=os.path.join(args.run_dir, "plots", "evaluation.png"),
                        workers=args.workers, random_heading=random_heading_val)

    elif args.command == "compare":
        compare_checkpoints(checkpoint_dir=checkpoint_dir, 
                            n_episodes=args.episodes, 
                            num_flights=args.num_flights,
                            save_path=os.path.join(args.run_dir, "plots", "checkpoint_comparison.png"),
                            workers=args.workers)

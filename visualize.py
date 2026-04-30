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
    python visualize.py evaluate --run-dir results/nospawninra --workers 8 --episodes 1000 --no-random-heading
    python visualize.py training --run-dir results/expanded_obs_matrix_4
    python visualize.py trajectory --run-dir results/nospawninra
    python visualize.py compare --run-dir results/alert_shared_reward_ALL_AGENTS
    python visualize.py evaluate --run-dir results/minimal_reward_ALL_AGENTS
    python visualize.py training --run-dir results/test_03drift_40conflict
    python visualize.py trajectory --run-dir results/minimal_reward_ALL_AGENTS
    python visualize.py evaluate --run-dir results/05drift_08conflict_1.5target_02proximity_ALL_AGENTS --episodes 100 --workers 8
    python visualize.py compare --run-dir results/05drift_06conflict_01target_02proximity_ALL_AGENTS --workers 8
    python visualize.py training --run-dir results/<run> --workers 8
    python visualize.py trajectory --run-dir results/basic_policy --workers 8

    python visualize.py evaluate --run-dir results/4_intruders_unlocked_physics --no-random-heading
    python visualize.py evaluate --run-dir results/minimal_reward_ALL_AGENTS --no-random-heading --workers 8
    python visualize.py evaluate --run-dir results/jan_3 --no-random-heading --workers 24 --episodes 1000
    
    python visualize.py compare --run-dir results/jan_3 --no-random-heading

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
    
    # Restricted airspace flags (already in [0, 1] range)
    # obs[5*n+3] and obs[5*n+4] - no normalization needed
    
   # Ownship: 5n to 5n+4
    # Restricted Block: 5n+5 to 5n+9
    res_idx = 5 * n + 5
    if res_idx + 4 < len(obs):
        # obs[res_idx] is binary 'is_in' -> no norm needed
        
        # Distance (res_idx + 1)
        obs[res_idx+1] = (obs[res_idx+1] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
        
        # sin/cos (res_idx + 2, + 3) -> already [-1, 1], no norm needed
        
        # Approach Rate (res_idx + 4) -> Normalize by max speed (e.g. 250 m/s)
        obs[res_idx+4] = obs[res_idx+4] / 250.0

    return np.clip(obs, -1.0, 1.0).astype(np.float32)


# ────────────────────────── TRAINING CURVES ──────────────────────────

def plot_training_curves(eval_log_path="results/eval_logs/evaluations.npz",
                         save_path="results/plots/training_curves.png"):
    """Plot reward and episode length curves from EvalCallback logs (academic styling)."""
    data = np.load(eval_log_path)
    timesteps = data["timesteps"]
    results = data["results"]      # (n_evals, n_episodes)
    ep_lengths = data["ep_lengths"]

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)
    mean_length = ep_lengths.mean(axis=1)
    std_length = ep_lengths.std(axis=1)

    # Set publication-quality style
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # Helper function to style axes
    def style_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

    # Reward curve
    ax = axes[0]
    ax.plot(timesteps, mean_reward, color="#1976D2", linewidth=2, label="Mean reward", zorder=2)
    ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                    alpha=0.15, color="#1976D2", zorder=1)
    ax.set_xlabel('Training Steps', fontweight='normal')
    ax.set_ylabel('Evaluation Reward', fontweight='normal')
    ax.set_title('(a) Training Reward Curve', fontweight='bold', loc='left')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fancybox=False)
    style_axis(ax)

    # Episode length curve
    ax = axes[1]
    ax.plot(timesteps, mean_length, color="#388E3C", linewidth=2, label="Mean episode length", zorder=2)
    ax.fill_between(timesteps, mean_length - std_length, mean_length + std_length,
                    alpha=0.15, color="#388E3C", zorder=1)
    ax.set_xlabel('Training Steps', fontweight='normal')
    ax.set_ylabel('Episode Length (steps)', fontweight='normal')
    ax.set_title('(b) Episode Length Curve', fontweight='bold', loc='left')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='black', fancybox=False)
    style_axis(ax)

    fig.suptitle('ATC Training Progress', fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.97, wspace=0.30)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"Saved academic-quality training curves to {save_path}")
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
                      save_path="results/plots/trajectories.png", random_heading=False):
    """Plot aircraft trajectories for one episode (academic styling)."""
    model = SAC.load(model_path)
    trajectories, targets, airspace, restricted_airspace, total_conflicts, total_restricted_intrusions = record_episode(
        model, num_flights, deploy_all
    )

    # Set publication-quality style
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')

    # Professional color palette for aircraft
    aircraft_colors = ['#1976D2', '#388E3C', '#D32F2F', '#F57C00', '#6A1B9A',
                      '#00796B', '#0097A7', '#7B1FA2', '#C2185B', '#E64A19']

    # Draw airspace boundary
    airspace_x = [p[0]/1000 for p in airspace]
    airspace_y = [p[1]/1000 for p in airspace]
    ax.plot(airspace_x, airspace_y, color='#424242', linestyle='--', alpha=0.4, linewidth=1.2, label="Airspace boundary")

    # Draw restricted airspace boundary
    if restricted_airspace:
        restricted_x = [p[0]/1000 for p in restricted_airspace]
        restricted_y = [p[1]/1000 for p in restricted_airspace]
        ax.plot(restricted_x, restricted_y, color='#D32F2F', alpha=0.7, linewidth=2.0, label="Restricted airspace", linestyle='-')

    for i in range(num_flights):
        traj = trajectories[i]
        x = np.array(traj["x"]) / 1000  # Convert to km
        y = np.array(traj["y"]) / 1000
        conflicts = traj["conflicts"]
        restricted_intrusions = traj["restricted_intrusions"]

        base_color = aircraft_colors[i % len(aircraft_colors)]

        # Plot trajectory with color indicating conflicts and restricted airspace intrusions
        for j in range(len(x) - 1):
            if conflicts[j]:
                color = "#B71C1C"
                linewidth = 2.5
                zorder = 3
            elif restricted_intrusions[j]:
                color = "#F9A825"
                linewidth = 2.0
                zorder = 2
            else:
                color = base_color
                linewidth = 1.5
                zorder = 1
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], color=color, linewidth=linewidth, zorder=zorder)

        # Start position (circle)
        ax.plot(x[0], y[0], 'o', color=base_color, markersize=8, zorder=5, markeredgecolor='white', markeredgewidth=0.5)
        ax.annotate(f"A{i}", (x[0], y[0]), fontsize=8, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 10),
                    textcoords="offset points", color='#212121')

        # Target position (star)
        tx, ty = targets[i][0]/1000, targets[i][1]/1000
        ax.plot(tx, ty, '*', color=base_color, markersize=16, zorder=5, markeredgecolor='white', markeredgewidth=0.5)

    # Legend with custom patches
    start_patch = mpatches.Patch(color="#757575", label="● Start position")
    target_patch = mpatches.Patch(color="#757575", label="★ Target")
    conflict_patch = mpatches.Patch(color="#B71C1C", label="— Conflict (severity high)")
    restricted_patch = mpatches.Patch(color="#F9A825", label="— Restricted intrusion")
    normal_patch = mpatches.Patch(color="#757575", label="— Normal flight path")
    ax.legend(handles=[start_patch, target_patch, normal_patch, conflict_patch, restricted_patch], 
              loc="upper left", framealpha=0.95, edgecolor='#424242', fancybox=False)

    mode = "Multi-agent" if deploy_all else "Single-agent"
    summary_text = f"{mode} · {num_flights} aircraft · {total_conflicts} conflict(s) · {total_restricted_intrusions} restricted intrusion(s)"
    ax.set_title(f"Aircraft Trajectories\n{summary_text}", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel("X coordinate (km)", fontweight='normal')
    ax.set_ylabel("Y coordinate (km)", fontweight='normal')
    ax.set_aspect("equal")
    
    # Clean axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"Saved academic-quality trajectory plot to {save_path}")
    plt.show()


# ────────────────────────── EVALUATION METRICS ──────────────────────────

def _eval_episodes_worker(model_path, episode_indices, num_flights, deploy_all, random_heading=False):
    """Worker function that runs a batch of episodes (used by ProcessPoolExecutor)."""
    model = SAC.load(model_path)
    env = Environment(num_flights=num_flights, random_init_heading=random_heading)

    local_metrics = {
        "conflicts": [],
        "targets_reached": [],
        "episode_length": [],
        "total_drift": [],
        "restricted_intrusions": [],
    }

    for ep in episode_indices:
        raw_obs_list = env.reset(num_flights)
        done = False
        ep_conflicts = 0
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
        local_metrics["targets_reached"].append(len(env.done))
        local_metrics["episode_length"].append(step)
        local_metrics["total_drift"].append(ep_drift)
        local_metrics["restricted_intrusions"].append(ep_restricted_intrusions)

    env.close()
    return local_metrics


def run_evaluation(model_path, n_episodes=30, num_flights=5, deploy_all=True, workers=1, random_heading=False):
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
                    save_path="results/plots/evaluation.png", workers=1, random_heading=False):
    """Run evaluation and plot summary metrics (academic styling)."""
    print(f"Running {n_episodes} episodes across {workers} worker(s)...")
    metrics = run_evaluation(model_path, n_episodes, num_flights, deploy_all=True, workers=workers, random_heading=random_heading)

    # Set publication-quality style
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 13
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('white')

    # Color palette for academic publication
    colors = {
        'conflicts': '#D32F2F',
        'targets': '#388E3C',
        'episode': '#1976D2',
        'drift': '#F57C00',
        'intrusions': '#6A1B9A',
        'mean_line': '#424242'
    }

    # Helper function to style axes
    def style_axis(ax, show_grid=True):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

    # Conflicts per episode
    ax = axes[0, 0]
    ax.bar(range(len(metrics["conflicts"])), metrics["conflicts"], 
           color=colors['conflicts'], alpha=0.75, edgecolor='none', width=0.8)
    mean_conflicts = np.mean(metrics['conflicts'])
    ax.axhline(mean_conflicts, color=colors['mean_line'], linestyle='-', linewidth=1.2, alpha=0.7)
    ax.text(len(metrics['conflicts'])*0.98, mean_conflicts, f'μ={mean_conflicts:.2f}', 
            ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_xlabel('Episode', fontweight='normal')
    ax.set_ylabel('Number of Conflicts', fontweight='normal')
    ax.set_title('(a) Conflicts per Episode', fontweight='bold', loc='left')
    ax.set_ylim(bottom=0)
    style_axis(ax)

    # Targets reached
    ax = axes[0, 1]
    ax.bar(range(len(metrics["targets_reached"])), metrics["targets_reached"],
           color=colors['targets'], alpha=0.75, edgecolor='none', width=0.8)
    mean_targets = np.mean(metrics['targets_reached'])
    ax.axhline(mean_targets, color=colors['mean_line'], linestyle='-', linewidth=1.2, alpha=0.7,
               label=f'Mean = {mean_targets:.2f}')
    ax.axhline(num_flights, color='#757575', linestyle='--', linewidth=1.0, alpha=0.6,
               label=f'Maximum = {num_flights}')
    ax.set_xlabel('Episode', fontweight='normal')
    ax.set_ylabel('Targets Reached', fontweight='normal')
    ax.set_title('(b) Targets Reached per Episode', fontweight='bold', loc='left')
    ax.set_ylim(0, num_flights + 0.5)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fancybox=False)
    style_axis(ax)

    # Restricted airspace intrusions per episode
    ax = axes[0, 2]
    ax.bar(range(len(metrics["restricted_intrusions"])), metrics["restricted_intrusions"], 
           color=colors['intrusions'], alpha=0.75, edgecolor='none', width=0.8)
    mean_intrusions = np.mean(metrics['restricted_intrusions'])
    ax.axhline(mean_intrusions, color=colors['mean_line'], linestyle='-', linewidth=1.2, alpha=0.7)
    ax.text(len(metrics['restricted_intrusions'])*0.98, mean_intrusions, f'μ={mean_intrusions:.2f}', 
            ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_xlabel('Episode', fontweight='normal')
    ax.set_ylabel('Restricted Zone Intrusions', fontweight='normal')
    ax.set_title('(c) Restricted Airspace Intrusions', fontweight='bold', loc='left')
    ax.set_ylim(bottom=0)
    style_axis(ax)

    # Episode length distribution
    ax = axes[1, 0]
    n, bins, patches = ax.hist(metrics["episode_length"], bins=15, color=colors['episode'], 
                                alpha=0.75, edgecolor='black', linewidth=0.8)
    ax.axvline(np.mean(metrics['episode_length']), color=colors['mean_line'], 
               linestyle='-', linewidth=1.2, alpha=0.7)
    ax.set_xlabel('Episode Length (steps)', fontweight='normal')
    ax.set_ylabel('Frequency', fontweight='normal')
    ax.set_title('(d) Episode Length Distribution', fontweight='bold', loc='left')
    style_axis(ax)

    # Cumulative drift per episode
    ax = axes[1, 1]
    ax.bar(range(len(metrics["total_drift"])), metrics["total_drift"], 
           color=colors['drift'], alpha=0.75, edgecolor='none', width=0.8)
    mean_drift = np.mean(metrics['total_drift'])
    ax.axhline(mean_drift, color=colors['mean_line'], linestyle='-', linewidth=1.2, alpha=0.7)
    ax.text(len(metrics['total_drift'])*0.98, mean_drift, f'μ={mean_drift:.2f}', 
            ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_xlabel('Episode', fontweight='normal')
    ax.set_ylabel('Cumulative Drift (radians)', fontweight='normal')
    ax.set_title('(e) Cumulative Drift per Episode', fontweight='bold', loc='left')
    ax.set_ylim(bottom=0)
    style_axis(ax)

    # Summary statistics panel
    ax = axes[1, 2]
    ax.axis('off')
    
    conflict_free = sum(1 for c in metrics["conflicts"] if c == 0)
    intrusion_free = sum(1 for i in metrics["restricted_intrusions"] if i == 0)
    
    summary_data = [
        ('Conflicts', f"{np.mean(metrics['conflicts']):.3f}", f"{100*conflict_free/n_episodes:.1f}%"),
        ('Targets Reached', f"{mean_targets:.3f}", f"{100*mean_targets/num_flights:.1f}%"),
        ('Restricted Intrusions', f"{mean_intrusions:.3f}", f"{100*intrusion_free/n_episodes:.1f}%"),
        ('Episode Length', f"{np.mean(metrics['episode_length']):.1f}", f"steps"),
        ('Drift', f"{mean_drift:.3f}", f"rad"),
    ]
    
    table_text = "Performance Metrics\n" + "="*45 + "\n"
    table_text += f"{'Metric':<22} {'Mean':<15} {'Rate':<10}\n"
    table_text += "-"*45 + "\n"
    for metric, value, rate in summary_data:
        table_text += f"{metric:<22} {value:<15} {rate:<10}\n"
    table_text += "="*45 + f"\n\nEpisodes: {n_episodes}"
    table_text += f"\nAircraft: {num_flights}"
    
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace', fontweight='normal',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', 
                     edgecolor='#424242', alpha=0.95, linewidth=1.0))

    fig.suptitle('ATC Conflict Resolution: Evaluation Results', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.97, hspace=0.35, wspace=0.30)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"Saved academic-quality evaluation plot to {save_path}")
    plt.show()


# ────────────────────────── CHECKPOINT COMPARISON ──────────────────────────

def _eval_checkpoint_worker(ckpt_path, step_num, n_episodes, num_flights):
    """Worker to evaluate a single checkpoint (used by ProcessPoolExecutor)."""
    metrics = _eval_episodes_worker(ckpt_path, list(range(n_episodes)), num_flights, deploy_all=True)
    mean_c = np.mean(metrics["conflicts"])
    mean_t = np.mean(metrics["targets_reached"])
    cf = sum(1 for c in metrics["conflicts"] if c == 0)
    cf_pct = 100 * cf / n_episodes
    return step_num, mean_c, mean_t, cf_pct


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
                step_num, mc, mt, cfp = future.result()
                steps.append(step_num)
                mean_conflicts.append(mc)
                mean_targets.append(mt)
                conflict_free_pct.append(cfp)
        # Sort by step number since parallel results may arrive out of order
        order = np.argsort(steps)
        steps = [steps[i] for i in order]
        mean_conflicts = [mean_conflicts[i] for i in order]
        mean_targets = [mean_targets[i] for i in order]
        conflict_free_pct = [conflict_free_pct[i] for i in order]

    # Set publication-quality style
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')

    # Helper function to style axes
    def style_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)

    # Conflict rate
    ax = axes[0]
    ax.plot(steps, mean_conflicts, "o-", color="#D32F2F", linewidth=2, markersize=6, zorder=2)
    ax.set_xlabel("Training Steps", fontweight='normal')
    ax.set_ylabel("Mean Conflicts per Episode", fontweight='normal')
    ax.set_title("(a) Conflict Rate vs Training", fontweight='bold', loc='left')
    style_axis(ax)

    # Target completion
    ax = axes[1]
    ax.plot(steps, mean_targets, "o-", color="#388E3C", linewidth=2, markersize=6, zorder=2)
    ax.axhline(num_flights, color='#424242', linestyle='--', linewidth=1.0, alpha=0.6, label=f"Maximum = {num_flights}", zorder=1)
    ax.set_xlabel("Training Steps", fontweight='normal')
    ax.set_ylabel("Mean Targets Reached", fontweight='normal')
    ax.set_title("(b) Target Completion vs Training", fontweight='bold', loc='left')
    ax.set_ylim(0, num_flights + 0.5)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fancybox=False)
    style_axis(ax)

    # Safety rate
    ax = axes[2]
    ax.plot(steps, conflict_free_pct, "o-", color="#1976D2", linewidth=2, markersize=6, zorder=2)
    ax.set_xlabel("Training Steps", fontweight='normal')
    ax.set_ylabel("Conflict-Free Episodes (%)", fontweight='normal')
    ax.set_title("(c) Safety Rate vs Training", fontweight='bold', loc='left')
    ax.set_ylim(-5, 105)
    ax.axhline(100, color='#757575', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
    style_axis(ax)

    fig.suptitle('ATC Training Progress: Checkpoint Comparison', fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.97, wspace=0.32)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"Saved academic-quality checkpoint comparison plot to {save_path}")
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

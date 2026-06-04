import os
import argparse
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from stable_baselines3 import SAC
from concurrent.futures import ProcessPoolExecutor

from atcenv.env import Environment, NUMBER_INTRUDERS_STATE
from visualize import normalize_obs


"""
Hypotheses evaluation script for the ATC conflict resolution model.

Usage:
    python evaluate_hypotheses.py <mode> [options]

Commands (Modes):
    airspace-sweep         Run airspace area sweep (Hypothesis B)
    density-sweep          Run traffic density sweep (Hypothesis D)
    uncertainty-ablation   Run environmental uncertainty ablation (Hypothesis C)
    heatmap                Generate spatial density heatmap (spaghetti plot)
    reward-progress        Plot incremental reward shaping progress (Hypothesis E)
    all                    Run all available hypothesis evaluations
    plot-airspace          Plot airspace sweep from a CSV file
    plot-density           Plot density sweep from a CSV file
    plot-uncertainty       Plot uncertainty ablation from a CSV file
    plot-airspace-compare  Plot airspace sweep comparing SAC and MVP
    plot-density-compare   Plot density sweep comparing SAC and MVP

Options:
    --run-dir              Path to run directory containing best_model/best_model.zip
    --episodes             Number of episodes to run per condition (default: 100)
    --incremental-dir      Tensorboard log dir for stepwise training (Hypothesis E)
    --baseline-dir         Tensorboard log dir for baseline training (Hypothesis E)
    --baseline-model       Path to baseline model dir (e.g. Adam's 30-dim obs model)
    --policy               Policy to use: 'sac' or 'mvp' (default: sac)
    --save-csv             Save evaluation metrics to CSV files
    --csv-path             Path to the CSV file to plot from (for plot-* modes)

Examples:
    python evaluate_hypotheses.py all --run-dir results/my_run --episodes 50 --save-csv
    python evaluate_hypotheses.py density-sweep --run-dir results/my_run --policy mvp --save-csv
    python evaluate_hypotheses.py plot-airspace --csv-path results/my_run/hypotheses_plots/airspace_sweep.csv
    python evaluate_hypotheses.py reward-progress --incremental-dir results/inc_logs --baseline-dir results/base_logs
"""

ACTION_FREQUENCY = 5

# --- Normalization constants ---
INTRUDER_DIST_NORM = 50000.0
INTRUDER_POS_NORM = 13000.0
TARGET_DIST_NORM = 200000.0

import atcenv.mvp_resolver as _mvp_improved
import atcenv.mvp_resolver_v0 as _mvp_original

_MVP_VERSION = "improved"

def mvp_actions_for_env(env):
    if _MVP_VERSION == "original":
        return _mvp_original.mvp_actions_for_env(env)
    return _mvp_improved.mvp_actions_for_env(env)


def normalize_obs_standard(raw_obs):
    """
    Standard Normalizer for 30-dim observation.
    Layout: [intruder_data (20)] [ownship_state (5)] [restricted_airspace (5)]
    """
    obs = np.array(raw_obs, dtype=np.float32)
    n = 4  # NUMBER_INTRUDERS_STATE

    # 1. Intruder Data (Indices 0 to 19)
    obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
    obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM
    obs[4*n:5*n] = obs[4*n:5*n] / np.pi

    # 2. Ownship State (Indices 20 to 24)
    # 20: Airspeed, 21: Optimal, 22: Target Dist
    obs[20] = (obs[20] - 230.0) / 30.0
    obs[21] = (obs[21] - 230.0) / 30.0
    obs[22] = (obs[22] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)
    # 23: Sin Drift, 24: Cos Drift (Already [-1, 1])

    # 3. Restricted Airspace Block (Indices 25 to 29)
    # 25: in_restricted (flag, no norm)
    # 26: Distance
    obs[26] = (obs[26] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    # 27: Sin Bearing, 28: Cos Bearing (Already [-1, 1])
    # 29: Approach Rate
    obs[29] = obs[29] / 250.0

    return np.clip(obs, -1.0, 1.0).astype(np.float32)

def run_episode(model, env_kwargs, num_flights=10, norm_fn=None, policy="sac"):
    """
    Measures drift once per decision step to prevent inflated metrics.
    """
    if norm_fn is None:
        norm_fn = normalize_obs_standard
        
    env_kwargs['random_init_heading'] = False
    env = Environment(num_flights=num_flights, **env_kwargs)
    raw_obs_list = env.reset(num_flights)

    done = False
    decision_steps = 0
    unique_conflicts = set()
    unique_intrusions = set()
    cumulative_drift = 0.0
    had_conflict = False
    had_intrusion = False

    while not done:
        # 1. Decide actions for all active flights
        current_actions = {}
        active_indices = [i for i in range(num_flights) if i not in env.done]

        # Ensure there are enough observations 
        if len(raw_obs_list) != len(active_indices):
            break

        if policy == "mvp":
            mvp_actions = mvp_actions_for_env(env)
            for idx, agent_num in enumerate(active_indices):
                current_actions[agent_num] = mvp_actions[idx]
        else:
            for idx, agent_num in enumerate(active_indices):
                # Use standard normalizer
                obs = norm_fn(raw_obs_list[idx])
                action, _ = model.predict(obs, deterministic=True)
                current_actions[agent_num] = action

        # 2. Step the environment ACTION_FREQUENCY times
        for sub_step in range(ACTION_FREQUENCY):
            active_now = [i for i in range(num_flights) if i not in env.done]
            if not active_now:
                done = True
                break

            env_actions = np.zeros((len(active_now), 2), dtype=np.float32)

            # Apply the SAME action for all ACTION_FREQUENCY steps (matches training wrapper)
            for idx, agent_num in enumerate(active_now):
                env_actions[idx] = current_actions.get(agent_num, [0.0, 0.0])

            raw_obs_list, rewards, done_t, done_e, info = env.step(env_actions)

            # Record safety violations that happen during any sub-step
            #if env.conflicts:
            #    had_conflict = True
            #if env.restricted_airspace_intrusions:
            #    had_intrusion = True

            #for f_idx in env.conflicts:
            #    unique_conflicts.add(f_idx)
            #for f_idx in env.restricted_airspace_intrusions:
            #    unique_intrusions.add(f_idx)

            if done_t or done_e:
                done = True
                break

        if len(env.conflicts) > 0:
            had_conflict = True
        if len(env.restricted_airspace_intrusions) > 0:
            had_intrusion = True
        
        # Record drift once per decision cycle
        for f in env.flights:
            if hasattr(f, 'drift'):
                cumulative_drift += abs(f.drift)

        decision_steps += 1

    targets_reached = len(env.done)
    env.close()
    # Return episode-level fail-safe flags
    return had_conflict, had_intrusion, targets_reached, cumulative_drift / max(1, decision_steps)

def eval_worker(model_path, env_kwargs, num_flights, episodes, policy="sac"):
    """Worker function to run a full parameter condition internally so it isn't bottlenecked."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings in workers
    from stable_baselines3 import SAC
    model = SAC.load(model_path) if policy == "sac" else None
    
    conflict_episodes = []
    intrusion_episodes = []
    drifts = []
    for _ in range(episodes):
        had_conflict, had_intrusion, _, d = run_episode(model, env_kwargs, num_flights, norm_fn=normalize_obs_standard, policy=policy)
        conflict_episodes.append(had_conflict)
        intrusion_episodes.append(had_intrusion)
        drifts.append(d)
    return conflict_episodes, intrusion_episodes, drifts

def plot_airspace_sweep(csv_path, out_dir):
    import pandas as pd
    import re
    df = pd.read_csv(csv_path)
    
    ratios = []
    cf_fail_rates = []
    if_fail_rates = []
    cf_ci = []
    if_ci = []
    
    cols = df.columns
    ratio_strs = set()
    for c in cols:
        m = re.match(r'Ratio_([0-9.]+)_Conflicts', c)
        if m:
            ratio_strs.add(m.group(1))
            
    sorted_ratio_strs = sorted(list(ratio_strs), key=float)
    
    for r_str in sorted_ratio_strs:
        ratios.append(float(r_str))
        c_col = f"Ratio_{r_str}_Conflicts"
        i_col = f"Ratio_{r_str}_Intrusions"
        c_data = df[c_col].dropna().values
        i_data = df[i_col].dropna().values
        N_c = len(c_data)
        N_i = len(i_data)
        cf_fail_rates.append(np.mean(c_data) * 100)
        if_fail_rates.append(np.mean(i_data) * 100)
        cf_ci.append(1.96 * (np.std(c_data, ddof=1) * 100) / np.sqrt(N_c) if N_c > 1 else 0)
        if_ci.append(1.96 * (np.std(i_data, ddof=1) * 100) / np.sqrt(N_i) if N_i > 1 else 0)

    cf_fail_rates = np.array(cf_fail_rates)
    if_fail_rates = np.array(if_fail_rates)
    cf_ci = np.array(cf_ci)
    if_ci = np.array(if_ci)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Restricted Airspace Ratio')
    ax1.set_ylabel('Episodes w/ Conflict (%)', color='tab:blue')
    ax1.plot(ratios, cf_fail_rates, color='tab:blue', marker='o', linewidth=1.5, label='Conflict Episode Rate')
    ax1.fill_between(ratios, np.maximum(0, cf_fail_rates - cf_ci), cf_fail_rates + cf_ci, color='tab:blue', alpha=0.15)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Episodes w/ Intrusion (%)', color='tab:orange')
    ax2.plot(ratios, if_fail_rates, color='tab:orange', marker='s', linewidth=1.5, linestyle='--', label='Intrusion Episode Rate')
    ax2.fill_between(ratios, np.maximum(0, if_fail_rates - if_ci), if_fail_rates + if_ci, color='tab:orange', alpha=0.15)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.axvline(x=0.20, color='gray', linestyle=':', label='Training Environment Area (20%)')

    ylim_max = max(np.max(cf_fail_rates + cf_ci), np.max(if_fail_rates + if_ci)) * 1.1 + 2
    if ylim_max < 20: ylim_max = 20
    ax1.set_ylim([-2, ylim_max])
    ax2.set_ylim([-2, ylim_max])

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "airspace_sweep.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved airspace_sweep.png")


def plot_airspace_compare(csv_sac, csv_mvp, out_dir):
    import pandas as pd
    import re
    
    # helper to process
    def get_data(csv_path, metric):
        df = pd.read_csv(csv_path)
        ratios, fail_rates, ci = [], [], []
        cols = df.columns
        ratio_strs = set()
        for c in cols:
            m = re.match(r'Ratio_([0-9.]+)_' + metric, c)
            if m: ratio_strs.add(m.group(1))
        for r_str in sorted(list(ratio_strs), key=float):
            ratios.append(float(r_str))
            data = df[f"Ratio_{r_str}_{metric}"].dropna().values
            N = len(data)
            fail_rates.append(np.mean(data) * 100)
            ci.append(1.96 * (np.std(data, ddof=1) * 100) / np.sqrt(N) if N > 1 else 0)
        return ratios, np.array(fail_rates), np.array(ci)

    for metric, fname_suffix, ylabel in [("Conflicts", "conflicts", "Episodes w/ Conflict (%)"), 
                                         ("Intrusions", "intrusions", "Episodes w/ Intrusion (%)")]:
        r_sac, f_sac, ci_sac = get_data(csv_sac, metric)
        r_mvp, f_mvp, ci_mvp = get_data(csv_mvp, metric)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Restricted Airspace Ratio')
        ax1.set_ylabel(ylabel)
        
        # Plot MVP first (orange, dashed, square)
        ax1.plot(r_mvp, f_mvp, color='tab:orange', marker='s', linewidth=1.5, linestyle='--', label='MVP', markersize=7, markerfacecolor='none')
        ax1.fill_between(r_mvp, np.maximum(0, f_mvp - ci_mvp), f_mvp + ci_mvp, color='tab:orange', alpha=0.15)
        
        # Plot SAC (blue, solid, circle)
        ax1.plot(r_sac, f_sac, color='tab:blue', marker='o', linewidth=1.5, label='SAC', markersize=5)
        ax1.fill_between(r_sac, np.maximum(0, f_sac - ci_sac), f_sac + ci_sac, color='tab:blue', alpha=0.15)
        
        ax1.axvline(x=0.20, color='gray', linestyle=':', label='Training Environment Area (20%)')
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left')
        
        fig.tight_layout()
        filename = f"airspace_sweep_compare_{fname_suffix}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")

def plot_density_compare(csv_sac, csv_mvp, out_dir):
    import pandas as pd
    import re
    
    def get_data(csv_path, metric):
        df = pd.read_csv(csv_path)
        densities, fail_rates, ci = [], [], []
        cols = df.columns
        dens_strs = set()
        for c in cols:
            m = re.match(r'Density_([0-9]+)_' + metric, c)
            if m: dens_strs.add(m.group(1))
        for d_str in sorted([int(d) for d in dens_strs]):
            densities.append(d_str)
            data = df[f"Density_{d_str}_{metric}"].dropna().values
            N = len(data)
            fail_rates.append(np.mean(data) * 100)
            ci.append(1.96 * (np.std(data, ddof=1) * 100) / np.sqrt(N) if N > 1 else 0)
        return densities, np.array(fail_rates), np.array(ci)

    for metric, fname_suffix, ylabel in [("Conflicts", "conflicts", "Episodes w/ Conflict (%)"), 
                                         ("Intrusions", "intrusions", "Episodes w/ Intrusion (%)")]:
        r_sac, f_sac, ci_sac = get_data(csv_sac, metric)
        r_mvp, f_mvp, ci_mvp = get_data(csv_mvp, metric)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Traffic Density (Number of Aircraft)')
        ax1.set_ylabel(ylabel)
        
        ax1.plot(r_mvp, f_mvp, color='tab:orange', marker='s', linewidth=1.5, linestyle='--', label='MVP', markersize=7, markerfacecolor='none')
        ax1.fill_between(r_mvp, np.maximum(0, f_mvp - ci_mvp), f_mvp + ci_mvp, color='tab:orange', alpha=0.15)
        
        ax1.plot(r_sac, f_sac, color='tab:blue', marker='o', linewidth=1.5, label='SAC', markersize=5)
        ax1.fill_between(r_sac, np.maximum(0, f_sac - ci_sac), f_sac + ci_sac, color='tab:blue', alpha=0.15)
        
        ax1.axvline(x=10, color='gray', linestyle=':', label='Training Environment Area (10 Flights)')
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left')
        
        fig.tight_layout()
        filename = f"density_sweep_compare_{fname_suffix}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")

def sweep_airspace(model_path, out_dir, episodes, default_flights=10, save_csv=False, policy="sac"):
    print("Running Airspace Area Sweep (Parallelized)...")
    ratios = np.arange(0.10, 0.525, 0.025)
    
    all_conflict_episodes = {}
    all_intrusion_episodes = {}

    with ProcessPoolExecutor() as executor:
        futures = {ratio: executor.submit(eval_worker, model_path, {'restricted_area_ratio': ratio}, default_flights, episodes, policy) for ratio in ratios}

        for ratio, future in tqdm(futures.items(), desc="Airspace Sweep"):
            conflict_episodes, intrusion_episodes, _ = future.result()
            all_conflict_episodes[ratio] = conflict_episodes
            all_intrusion_episodes[ratio] = intrusion_episodes

    if save_csv:
        import pandas as pd
        data = {}
        for ratio in ratios:
            data[f"Ratio_{ratio:.3f}_Conflicts"] = all_conflict_episodes[ratio]
            data[f"Ratio_{ratio:.3f}_Intrusions"] = all_intrusion_episodes[ratio]
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"airspace_sweep_{policy}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved airspace sweep metrics to {csv_path}")
        plot_airspace_sweep(csv_path, out_dir)
    else:
        # Save a temporary CSV to plot from
        import pandas as pd
        data = {}
        for ratio in ratios:
            data[f"Ratio_{ratio:.3f}_Conflicts"] = all_conflict_episodes[ratio]
            data[f"Ratio_{ratio:.3f}_Intrusions"] = all_intrusion_episodes[ratio]
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"temp_airspace_sweep_{policy}.csv")
        df.to_csv(csv_path, index=False)
        plot_airspace_sweep(csv_path, out_dir)
        os.remove(csv_path)

def plot_density_sweep(csv_path, out_dir):
    import pandas as pd
    import re
    df = pd.read_csv(csv_path)
    
    densities = []
    cf_fail_rates = []
    if_fail_rates = []
    cf_ci = []
    if_ci = []
    
    cols = df.columns
    dens_strs = set()
    for c in cols:
        m = re.match(r'Density_([0-9]+)_Conflicts', c)
        if m:
            dens_strs.add(m.group(1))
            
    sorted_dens = sorted([int(d) for d in dens_strs])
    
    for d in sorted_dens:
        densities.append(d)
        c_col = f"Density_{d}_Conflicts"
        i_col = f"Density_{d}_Intrusions"
        c_data = df[c_col].dropna().values
        i_data = df[i_col].dropna().values
        N_c = len(c_data)
        N_i = len(i_data)
        cf_fail_rates.append(np.mean(c_data) * 100)
        if_fail_rates.append(np.mean(i_data) * 100)
        cf_ci.append(1.96 * (np.std(c_data, ddof=1) * 100) / np.sqrt(N_c) if N_c > 1 else 0)
        if_ci.append(1.96 * (np.std(i_data, ddof=1) * 100) / np.sqrt(N_i) if N_i > 1 else 0)

    cf_fail_rates = np.array(cf_fail_rates)
    if_fail_rates = np.array(if_fail_rates)
    cf_ci = np.array(cf_ci)
    if_ci = np.array(if_ci)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(densities, cf_fail_rates, color='tab:blue', marker='o', linewidth=1.5, label='Conflict Episode Rate')
    ax.fill_between(densities, np.maximum(0, cf_fail_rates - cf_ci), cf_fail_rates + cf_ci, color='tab:blue', alpha=0.15)

    ax.plot(densities, if_fail_rates, color='tab:orange', marker='s', linewidth=1.5, label='Intrusion Episode Rate')
    ax.fill_between(densities, np.maximum(0, if_fail_rates - if_ci), if_fail_rates + if_ci, color='tab:orange', alpha=0.15)

    ax.axhline(y=20, color='red', linestyle='--', label='Limit Threshold')

    ax.set_xlabel('Traffic Density (Number of Aircraft)')
    ax.set_ylabel('Error Rate (% of episodes)')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.autoscale(enable=True, axis='y')
    ylim = ax.get_ylim()
    if ylim[1] < 25: ax.set_ylim([-2, 25])

    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_sweep.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved density_sweep.png")


def sweep_density(model_path, out_dir, episodes, save_csv=False, policy="sac"):
    print("Running Density Sweep (Parallelized)...")
    densities = np.arange(10, 31, 1) 
    
    all_conflict_episodes = {}
    all_intrusion_episodes = {}

    with ProcessPoolExecutor() as executor:
        # Pass a smaller distance_init_buffer (2.0) and enable relaxation to allow aircraft to fit at high densities
        futures = {d: executor.submit(eval_worker, model_path, {'distance_init_buffer': 2.0, 'enable_spawn_relaxation': True}, d, episodes, policy) for d in densities}

        for d, future in tqdm(futures.items(), desc="Density Sweep"):
            conflict_episodes, intrusion_episodes, drift_arr = future.result()
            all_conflict_episodes[d] = conflict_episodes
            all_intrusion_episodes[d] = intrusion_episodes

    if save_csv:
        import pandas as pd
        data = {}
        for d in densities:
            data[f"Density_{d}_Conflicts"] = all_conflict_episodes[d]
            data[f"Density_{d}_Intrusions"] = all_intrusion_episodes[d]
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"density_sweep_{policy}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved density sweep metrics to {csv_path}")
        plot_density_sweep(csv_path, out_dir)
    else:
        import pandas as pd
        data = {}
        for d in densities:
            data[f"Density_{d}_Conflicts"] = all_conflict_episodes[d]
            data[f"Density_{d}_Intrusions"] = all_intrusion_episodes[d]
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"temp_density_sweep_{policy}.csv")
        df.to_csv(csv_path, index=False)
        plot_density_sweep(csv_path, out_dir)
        os.remove(csv_path)

def plot_uncertainties_sweep(csv_path, out_dir):
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    names = []
    cf_rates = []
    if_rates = []
    cf_ci = []
    if_ci = []
    
    expected_names = ["Base (Ideal)", "Scramble Only", "Wind Only", "Delay Only", "All Uncertainties"]
    
    for name in expected_names:
        c_col = f"Condition_{name}_Conflicts"
        i_col = f"Condition_{name}_Intrusions"
        if c_col in df.columns and i_col in df.columns:
            names.append(name)
            c_data = df[c_col].dropna().values
            i_data = df[i_col].dropna().values
            N_c = len(c_data)
            N_i = len(i_data)
            cf_rates.append(np.mean(c_data) * 100)
            if_rates.append(np.mean(i_data) * 100)
            cf_ci.append(1.96 * (np.std(c_data, ddof=1) * 100) / np.sqrt(N_c) if N_c > 1 else 0)
            if_ci.append(1.96 * (np.std(i_data, ddof=1) * 100) / np.sqrt(N_i) if N_i > 1 else 0)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    cf_lower_error = np.minimum(cf_rates, cf_ci)
    if_lower_error = np.minimum(if_rates, if_ci)
    rects1 = ax.bar(x - width/2, cf_rates, width, yerr=[cf_lower_error, cf_ci], capsize=4, label='Conflict Episode Rate', color='#aec7e8', edgecolor='#1f77b4')
    rects2 = ax.bar(x + width/2, if_rates, width, yerr=[if_lower_error, if_ci], capsize=4, label='Intrusion Episode Rate', color='#ffbb78', edgecolor='#ff7f0e')

    ax.set_ylabel('Episodes with Errors (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    ax.autoscale(enable=True, axis='y')
    ylim = ax.get_ylim()
    if ylim[1] < 25: ax.set_ylim([-2, 25])

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_ablation.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved uncertainty_ablation.png")

def sweep_uncertainties(model_path, out_dir, episodes, default_flights=10, save_csv=False, policy="sac"):
    print("Running Uncertainty Ablation Sweep (Parallelized)...")
    experiments = [
        ("Base (Ideal)", {}),
        ("Scramble Only", {'enable_position_uncertainty': True}),
        ("Wind Only", {'enable_wind': True}),
        ("Delay Only", {'enable_delay': True}),
        ("All Uncertainties", {'enable_position_uncertainty': True, 'enable_wind': True, 'enable_delay': True})
    ]
    
    all_conflict_episodes = {}
    all_intrusion_episodes = {}

    with ProcessPoolExecutor() as executor:
        futures = {name: executor.submit(eval_worker, model_path, config, default_flights, episodes, policy) for name, config in experiments}

        for name, future in tqdm(futures.items(), desc="Uncertainty Ablation"):
            cfs_list, ifs_list, _ = future.result()
            all_conflict_episodes[name] = cfs_list
            all_intrusion_episodes[name] = ifs_list

    if save_csv:
        import pandas as pd
        data = {}
        for name, _ in experiments:
            data[f"Condition_{name}_Conflicts"] = all_conflict_episodes[name]
            data[f"Condition_{name}_Intrusions"] = all_intrusion_episodes[name]
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"uncertainty_ablation_{policy}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved uncertainty ablation metrics to {csv_path}")
        plot_uncertainties_sweep(csv_path, out_dir)
    else:
        import pandas as pd
        data = {}
        for name, _ in experiments:
            data[f"Condition_{name}_Conflicts"] = all_conflict_episodes[name]
            data[f"Condition_{name}_Intrusions"] = all_intrusion_episodes[name]
        df = pd.DataFrame(data)
        csv_path = os.path.join(out_dir, f"temp_uncertainty_ablation_{policy}.csv")
        df.to_csv(csv_path, index=False)
        plot_uncertainties_sweep(csv_path, out_dir)
        os.remove(csv_path)

def generate_heatmap(model, out_dir, episodes, default_flights=10, policy="sac"):
    """
    Generate Heatmap with fixed geometry and target-facing initialization.
    Aircraft start facing their targets to match training behavior.
    """
    print(f"Generating Heatmap ({episodes} episodes on fixed airspace)...")
    env = Environment(num_flights=default_flights, random_init_heading=False)
    
    # Initialize fixed geometry - lock airspace
    env.reset(default_flights)
    fix_air, fix_res = env.airspace, env.restricted_airspace
    
    all_traj = []
    for _ in tqdm(range(episodes), desc="Heatmap Episodes"):
        # Reuse fixed airspace for this episode
        env.airspace = fix_air
        env.restricted_airspace = fix_res
        env.flights = []
        env.done = set()
        env.i = 0
        env.conflicts = set()
        env.restricted_airspace_intrusions = set()
        
        # Generate flights that start facing their targets (random_init_heading=False)
        from atcenv.definitions import Flight
        tol = env.distance_init_buffer * env.tol
        min_distance = env.distance_init_buffer * env.min_distance
        while len(env.flights) < default_flights:
            candidate = Flight.random(env.airspace, env.min_speed, env.max_speed, tol, 
                                     random_init_heading=False, restricted_airspace=env.restricted_airspace)
            valid = True
            for f in env.flights:
                if math.hypot(candidate.position.x - f.position.x, candidate.position.y - f.position.y) < min_distance:
                    valid = False
                    break
            if valid:
                env.flights.append(candidate)
        
        raw_obs_list = env.observation()
        # Initialize trajectories with starting positions
        traj = {i: {'x': [f.position.x], 'y': [f.position.y]} for i, f in enumerate(env.flights)}
        done = False
        
        while not done:
            active = [i for i in range(default_flights) if i not in env.done]
            if not active:
                break
            
            actions = {}
            if policy == "mvp":
                mvp_actions = mvp_actions_for_env(env)
                for idx, agent_num in enumerate(active):
                    actions[agent_num] = mvp_actions[idx]
            else:
                for idx, agent_num in enumerate(active):
                    obs = normalize_obs_standard(raw_obs_list[idx])
                    act, _ = model.predict(obs, deterministic=True)
                    actions[agent_num] = act

            for sub_step in range(ACTION_FREQUENCY):
                now = [i for i in range(default_flights) if i not in env.done]
                if not now:
                    break
                    
                env_act = np.zeros((len(now), 2), dtype=np.float32)
                
                # Apply the SAME action for all ACTION_FREQUENCY steps (matches training wrapper)
                for idx, agent_num in enumerate(now):
                    env_act[idx] = actions.get(agent_num, [0, 0])
                
                raw_obs_list, _, done_t, done_e, _ = env.step(env_act)
                
                # Record positions of planes still flying or just reached target
                for idx, f in enumerate(env.flights):
                    if idx in now:
                        traj[idx]['x'].append(f.position.x)
                        traj[idx]['y'].append(f.position.y)
                        
                if done_t or done_e:
                    done = True
                    break

        all_traj.append(traj)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    
    # Draw all trajectories as spaghetti lines
    for t_set in all_traj:
        for i in t_set:
            if len(t_set[i]['x']) > 1:
                ax.plot(t_set[i]['x'], t_set[i]['y'], color='yellow', alpha=0.15, linewidth=0.7)

    # Draw fixed airspace boundaries
    x, y = fix_air.polygon.exterior.xy
    ax.plot(x, y, color='white', linewidth=2, label='Outer Boundary')
    x, y = fix_res.polygon.exterior.xy
    ax.plot(x, y, color='cyan', linewidth=2, linestyle='--', label='Restricted Airspace')
    
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    env.close()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_heatmap.png"), dpi=300)
    plt.close()
    print(f"Saved trajectory_heatmap.png to {out_dir}")

def plot_reward_progress(incremental_dir, baseline_dir, out_dir):
    print("Running Reward Progression Plot...")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard is not installed. Pip install tensorboard to run this plot.")
        return
        
    def get_data(log_dir, tag):
        events = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if "tfevents" in file:
                    ea = EventAccumulator(os.path.join(root, file))
                    ea.Reload()
                    if tag in ea.Tags()['scalars']:
                        for e in ea.Scalars(tag):
                            events.append((e.step, e.value))
        events.sort(key=lambda x: x[0])
        if not events: return [], []
        steps, values = zip(*events)
        return steps, values

    inc_steps, inc_vals = get_data(incremental_dir, "rollout/ep_rew_mean")
    base_steps, base_vals = get_data(baseline_dir, "rollout/ep_rew_mean")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if inc_steps:
        ax.plot(inc_steps, inc_vals, label='Incremental Reward Shaping (3 Stages)', linewidth=2)
    if base_steps:
        ax.plot(base_steps, base_vals, label='Single Session (Baseline)', linewidth=2)
        
    ax.axvline(x=4000000, color='gray', linestyle=':', label='Stage 2: Conflict Penalty Added')
    ax.axvline(x=8000000, color='gray', linestyle='--', label='Stage 3: Intrusion Penalty Added')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Reward')
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "incremental_progress.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved incremental_progress.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['airspace-sweep', 'density-sweep', 'uncertainty-ablation', 'reward-progress', 'heatmap', 'all', 'plot-airspace', 'plot-density', 'plot-uncertainty', 'plot-airspace-compare', 'plot-density-compare'])
    parser.add_argument('--run-dir', required=False, default="results/thisonLite", help="Path to run directory containing best_model/best_model.zip")
    parser.add_argument('--episodes', default=100, type=int, help="Number of episodes to run per condition")
    parser.add_argument('--incremental-dir', default=None, help="Tensorboard log dir for stepwise training (Hypothesis E)")
    parser.add_argument('--baseline-dir', default=None, help="Tensorboard log dir for baseline training (Hypothesis E)")
    parser.add_argument('--baseline-model', default=None, help="Path to baseline model dir (e.g. Adam's 30-dim obs model)")
    parser.add_argument('--save-csv', action='store_true', help="Save evaluation metrics to CSV files.")
    parser.add_argument('--csv-path', default=None, help="Path to the CSV file to plot from (for plot-* modes)")
    parser.add_argument('--csv-sac', default=None, help="Path to the SAC CSV file for comparison modes")
    parser.add_argument('--csv-mvp', default=None, help="Path to the MVP CSV file for comparison modes")
    parser.add_argument('--policy', default='sac', choices=['sac', 'mvp'], help="Policy to evaluate (sac or mvp)")
    args = parser.parse_args()
    
    # Set academic plot defaults
    matplotlib.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    
    out_dir = "results/hypotheses_plots"
    if args.run_dir:
        out_dir = os.path.join(args.run_dir, "hypotheses_plots")
    elif args.csv_path:
        out_dir = os.path.dirname(args.csv_path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Artifacts will be stored in {out_dir}")

    if args.mode == 'plot-airspace':
        if not args.csv_path: print("Error: --csv-path is required")
        else: plot_airspace_sweep(args.csv_path, out_dir)
        exit(0)
    elif args.mode == 'plot-density':
        if not args.csv_path: print("Error: --csv-path is required")
        else: plot_density_sweep(args.csv_path, out_dir)
        exit(0)
    elif args.mode == 'plot-uncertainty':
        if not args.csv_path: print("Error: --csv-path is required")
        else: plot_uncertainties_sweep(args.csv_path, out_dir)
        exit(0)

    if args.mode in ['reward-progress', 'all']:
        if args.incremental_dir or args.baseline_dir:
            plot_reward_progress(args.incremental_dir, args.baseline_dir, out_dir)
        elif args.mode == 'reward-progress':
            print("Error: Must provide --incremental-dir and --baseline-dir for this plot.")
            

    if args.mode == 'plot-airspace-compare':
        if not args.csv_sac or not args.csv_mvp:
            print("Error: plot-airspace-compare requires --csv-sac and --csv-mvp")
            exit(1)
        plot_airspace_compare(args.csv_sac, args.csv_mvp, out_dir)
        exit(0)
    elif args.mode == 'plot-density-compare':
        if not args.csv_sac or not args.csv_mvp:
            print("Error: plot-density-compare requires --csv-sac and --csv-mvp")
            exit(1)
        plot_density_compare(args.csv_sac, args.csv_mvp, out_dir)
        exit(0)

    if args.mode in ['airspace-sweep', 'density-sweep', 'uncertainty-ablation', 'heatmap', 'all']:
        # Determine which model to use
        if args.baseline_model:
            run_dir = args.baseline_model
        elif args.run_dir:
            run_dir = args.run_dir
        else:
            print("Error: --run-dir or --baseline-model is required for evaluate sweeps.")
            exit(1)
            
        model_path = os.path.join(run_dir, "best_model", "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(run_dir, "sac_atc_final.zip")
            
        print(f"Loading model from {model_path}")
        model = SAC.load(model_path) if args.policy == "sac" else None
        
        if out_dir == "results/hypotheses_plots" and args.baseline_model:
            out_dir = os.path.join(run_dir, "hypotheses_plots")
            os.makedirs(out_dir, exist_ok=True)
        
        if args.mode in ['airspace-sweep', 'all']:
            sweep_airspace(model_path, out_dir, args.episodes, save_csv=args.save_csv, policy=args.policy)
        if args.mode in ['density-sweep', 'all']:
            sweep_density(model_path, out_dir, args.episodes, save_csv=args.save_csv, policy=args.policy)
        if args.mode in ['uncertainty-ablation', 'all']:
            sweep_uncertainties(model_path, out_dir, args.episodes, save_csv=args.save_csv, policy=args.policy)
        if args.mode in ['heatmap', 'all']:
            model = SAC.load(model_path) if args.policy == "sac" else None
            generate_heatmap(model, out_dir, args.episodes, policy=args.policy)
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
Usage:
    python evaluate_hypotheses.py <mode> --run-dir <path>

Commands (Modes):
    python evaluate_hypotheses.py airspace-sweep         # Run airspace area sweep (Hypothesis B)
    python evaluate_hypotheses.py density-sweep          # Run traffic density sweep (Hypothesis D)
    python evaluate_hypotheses.py uncertainty-ablation   # Run environmental uncertainty ablation (Hypothesis C)
    python evaluate_hypotheses.py heatmap                # Generate spatial density heatmap (spaghetti plot)
    python evaluate_hypotheses.py reward-progress        # Plot incremental reward shaping progress (Hypothesis E)
    python evaluate_hypotheses.py all                    # Run all available hypothesis evaluations

    To evaluate hypotheses on a specific model, use the --run-dir argument:
    
    python evaluate_hypotheses.py airspace-sweep --run-dir results/best_model_run
    python evaluate_hypotheses.py density-sweep --run-dir results/test_03drift_40conflict
    python evaluate_hypotheses.py heatmap --run-dir results/minimal_reward_ALL_AGENTS
    python evaluate_hypotheses.py all --run-dir results/expanded_obs_matrix_4

    You can specify the number of episodes to run per condition using the --episodes argument (default is 100):
    
    python evaluate_hypotheses.py uncertainty-ablation --run-dir results/best_model_run --episodes 50
    python evaluate_hypotheses.py density-sweep --run-dir results/basic_policy --episodes 200

    To evaluate using a specific baseline model (like Adam's 30-dim obs model), use --baseline-model instead of --run-dir:
    
    python evaluate_hypotheses.py all --baseline-model results/adams_baseline

    To plot the reward progression (Hypothesis E), you must provide the Tensorboard log directories instead of a run directory:
    
    python evaluate_hypotheses.py reward-progress --incremental-dir results/incremental_logs --baseline-dir results/baseline_logs

"""

ACTION_FREQUENCY = 15

# --- Normalization for Adam's 30-dim observation (no relative velocities) ---
INTRUDER_DIST_NORM = 50000.0
INTRUDER_POS_NORM = 13000.0
TARGET_DIST_NORM = 200000.0

def normalize_obs_lite(raw_obs):
    """Normalize a 38-dim observation down to a 30-dim observation (Adam's branch, no relative velocities)."""
    obs = np.array(raw_obs, dtype=np.float32)
    n = NUMBER_INTRUDERS_STATE
    if len(obs) == 38:
        # Downsample to 30 by removing rel_vx (20:24) and rel_vy (24:28)
        obs = np.concatenate((obs[:5*n], obs[7*n:]))

    obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
    obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM
    obs[4*n:5*n] = obs[4*n:5*n] / np.pi
    obs[5*n]     = (obs[5*n] - 230.0) / 30.0
    obs[5*n+1]   = (obs[5*n+1] - 230.0) / 30.0
    obs[5*n+2]   = (obs[5*n+2] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)
    # sin/cos drift at 5*n+3, 5*n+4 already in [-1,1]
    # restricted block at 5*n+5..5*n+9
    res_idx = 5 * n + 5
    if res_idx + 1 < len(obs):
        obs[res_idx+1] = (obs[res_idx+1] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    if res_idx + 4 < len(obs):
        obs[res_idx+4] = obs[res_idx+4] / 250.0
    return np.clip(obs, -1.0, 1.0).astype(np.float32)

def get_normalizer(model):
    """Auto-detect observation size and return the correct normalizer."""
    obs_size = model.observation_space.shape[0]
    if obs_size <= 30:
        return normalize_obs_lite
    return normalize_obs

# Global reference set by main, used by workers
_USE_LITE_NORM = False

def _get_norm_fn():
    if _USE_LITE_NORM:
        return normalize_obs_lite
    return normalize_obs

def run_episode(model, env_kwargs, num_flights=10, norm_fn=None):
    if norm_fn is None:
        norm_fn = _get_norm_fn()
    expected_obs_size = model.observation_space.shape[0]
    env_kwargs['random_init_heading'] = False
    env = Environment(num_flights=num_flights, **env_kwargs)
    raw_obs_list = env.reset(num_flights)

    done = False
    step = 0
    unique_conflicts = set()
    unique_intrusions = set()
    cumulative_drift = 0.0
    
    while not done:
        current_actions = {}
        agent_idx = 0
        for i in range(num_flights):
            if i not in env.done:
                obs = norm_fn(raw_obs_list[agent_idx])
                action, _ = model.predict(obs, deterministic=True)
                current_actions[i] = action
                agent_idx += 1

        for _ in range(ACTION_FREQUENCY):
            active_indices = [i for i in range(num_flights) if i not in env.done]
            actions = np.zeros((len(active_indices), 2), dtype=np.float32)
            for idx, agent_num in enumerate(active_indices):
                actions[idx] = current_actions.get(agent_num, np.array([0.0, 0.0], dtype=np.float32))

            raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
            
            for f_idx in env.conflicts:
                unique_conflicts.add(f_idx)
            for f_idx in env.restricted_airspace_intrusions:
                unique_intrusions.add(f_idx)
                
            for f in env.flights:
                cumulative_drift += abs(f.drift)
                
            if done_t or done_e:
                done = True
                break

        step += 1

    targets_reached = len(env.done)
    env.close()
    return len(unique_conflicts), len(unique_intrusions), targets_reached, cumulative_drift / max(1, step)

def eval_worker(model_path, env_kwargs, num_flights, episodes):
    """Worker function to run a full parameter condition internally so it isn't bottlenecked."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings in workers
    from stable_baselines3 import SAC
    model = SAC.load(model_path)
    norm_fn = get_normalizer(model)
    
    cfs_list = []; ifs_list = []; drifts = []
    for _ in range(episodes):
        cf, inf, _, d = run_episode(model, env_kwargs, num_flights, norm_fn=norm_fn)
        cfs_list.append((cf / num_flights) * 100)
        ifs_list.append((inf / num_flights) * 100)
        drifts.append(d)
    return cfs_list, ifs_list, drifts

def sweep_airspace(model_path, out_dir, episodes, default_flights=10):
    print("Running Airspace Area Sweep (Parallelized)...")
    ratios = np.arange(0.10, 0.525, 0.025)
    cf_rates = []; if_rates = []
    cf_stds = []; if_stds = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for ratio in ratios:
            futures.append(executor.submit(eval_worker, model_path, {'restricted_area_ratio': ratio}, default_flights, episodes))
            
        for future in tqdm(futures, desc="Airspace Sweep"):
            cfs_list, ifs_list, _ = future.result()
            cf_rates.append(np.mean(cfs_list))
            if_rates.append(np.mean(ifs_list))
            cf_stds.append(np.std(cfs_list))
            if_stds.append(np.std(ifs_list))
            
    cf_rates = np.array(cf_rates); if_rates = np.array(if_rates)
    cf_stds = np.array(cf_stds); if_stds = np.array(if_stds)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Restricted Airspace Ratio (Area %)')
    ax1.set_ylabel('Flights w/ Conflict (%)', color='tab:blue')
    ax1.plot(ratios, cf_rates, color='tab:blue', marker='o', linewidth=1.5, label='Conflict Rate')
    ax1.fill_between(ratios, np.maximum(0, cf_rates - cf_stds), cf_rates + cf_stds, color='tab:blue', alpha=0.15)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Flights w/ Intrusion (%)', color='tab:orange')
    ax2.plot(ratios, if_rates, color='tab:orange', marker='s', linewidth=1.5, linestyle='--', label='Intrusion Rate')
    ax2.fill_between(ratios, np.maximum(0, if_rates - if_stds), if_rates + if_stds, color='tab:orange', alpha=0.15)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    ax1.axvline(x=0.20, color='gray', linestyle=':', label='Training Environment Area (20%)')
    
    ylim_max = max(np.max(cf_rates + cf_stds), np.max(if_rates + if_stds)) * 1.1 + 2
    if ylim_max < 20: ylim_max = 20
    ax1.set_ylim([-2, ylim_max])
    ax2.set_ylim([-2, ylim_max])
    
    #plt.title("Hypothesis B: Robustness Against Enlarged Restricted Airspace", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "airspace_sweep.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved airspace_sweep.png")

def sweep_density(model_path, out_dir, episodes):
    print("Running Density Sweep (Parallelized)...")
    densities = np.arange(10, 31, 1) # from 10 to 30
    cf_rates = []; if_rates = []
    cf_stds = []; if_stds = []
    drifts = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for num_flights in densities:
            futures.append(executor.submit(eval_worker, model_path, {}, num_flights, episodes))
            
        for i, future in enumerate(tqdm(futures, desc="Density Sweep")):
            cfs_list, ifs_list, drift_arr = future.result()
            
            c = np.mean(cfs_list)
            i_pct = np.mean(ifs_list)
            cf_rates.append(c)
            if_rates.append(i_pct)
            cf_stds.append(np.std(cfs_list))
            if_stds.append(np.std(ifs_list))
            drifts.append(drift_arr)
            
    cf_rates = np.array(cf_rates); if_rates = np.array(if_rates)
    cf_stds = np.array(cf_stds); if_stds = np.array(if_stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(densities, cf_rates, color='tab:blue', marker='o', linewidth=1.5, label='Conflict Rate')
    ax.fill_between(densities, np.maximum(0, cf_rates - cf_stds), cf_rates + cf_stds, color='tab:blue', alpha=0.15)
    
    ax.plot(densities, if_rates, color='tab:orange', marker='s', linewidth=1.5, label='Intrusion Rate')
    ax.fill_between(densities, np.maximum(0, if_rates - if_stds), if_rates + if_stds, color='tab:orange', alpha=0.15)
    
    ax.axhline(y=20, color='red', linestyle='--', label='Limit Threshold')
    
    ax.set_xlabel('Traffic Density (Number of Aircraft)')
    ax.set_ylabel('Error Rate (% of flights)')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.autoscale(enable=True, axis='y')
    ylim = ax.get_ylim()
    if ylim[1] < 25: ax.set_ylim([-2, 25])
    
    ax.legend(loc='upper left')
    #plt.title("Hypothesis D: Operational Limit vs Traffic Density", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_sweep.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved density_sweep.png")

def sweep_uncertainties(model_path, out_dir, episodes, default_flights=10):
    print("Running Uncertainty Ablation Sweep (Parallelized)...")
    experiments = [
        ("Base (Ideal)", {}),
        ("Scramble Only", {'enable_position_uncertainty': True}),
        ("Wind Only", {'enable_wind': True}),
        ("Delay Only", {'enable_delay': True}),
        ("All Uncertainties", {'enable_position_uncertainty': True, 'enable_wind': True, 'enable_delay': True})
    ]
    
    cf_rates = []; if_rates = []
    cf_stds = []; if_stds = []
    names = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for name, config in experiments:
            futures.append(executor.submit(eval_worker, model_path, config, default_flights, episodes))
            names.append(name)
            
        for future in tqdm(futures, desc="Uncertainty Ablation"):
            cfs_list, ifs_list, _ = future.result()
            cf_rates.append(np.mean(cfs_list))
            if_rates.append(np.mean(ifs_list))
            cf_stds.append(np.std(cfs_list))
            if_stds.append(np.std(ifs_list))
        
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    cf_lower_error = np.minimum(cf_rates, cf_stds)
    if_lower_error = np.minimum(if_rates, if_stds)
    rects1 = ax.bar(x - width/2, cf_rates, width, yerr=[cf_lower_error, cf_stds], capsize=4, label='Conflict Rate', color='#aec7e8', edgecolor='#1f77b4')
    rects2 = ax.bar(x + width/2, if_rates, width, yerr=[if_lower_error, if_stds], capsize=4, label='Intrusion Rate', color='#ffbb78', edgecolor='#ff7f0e')

    ax.set_ylabel('Flights with Errors (%)')
    #ax.set_title('Hypothesis C: Ablation Study of Environmental Uncertainties')
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

def generate_heatmap(model, out_dir, episodes, default_flights=1000):
    print("Generating Spatial Density Heatmap (Spaghetti Plot)...")
    env_kwargs = {'restricted_area_ratio': 0.20, 'fixed_geometries': True, 'random_init_heading': False}
    
    # Store trajectories per episode per flight
    all_trajectories = []
    
    env = Environment(num_flights=default_flights, **env_kwargs)
    
    for _ in tqdm(range(episodes)):
        raw_obs_list = env.reset(default_flights)
        done = False
        
        episode_trajectories = {i: {'x': [], 'y': []} for i in range(default_flights)}
        
        while not done:
            current_actions = {}
            agent_idx = 0
            for i in range(default_flights):
                if i not in env.done:
                    norm_fn = get_normalizer(model)
                    obs = norm_fn(raw_obs_list[agent_idx])
                    action, _ = model.predict(obs, deterministic=True)
                    current_actions[i] = action
                    agent_idx += 1

            for _ in range(ACTION_FREQUENCY):
                active_indices = [i for i in range(default_flights) if i not in env.done]
                actions = np.zeros((len(active_indices), 2), dtype=np.float32)
                for idx, agent_num in enumerate(active_indices):
                    actions[idx] = current_actions.get(agent_num, np.array([0.0, 0.0], dtype=np.float32))

                raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
                
                # Record the line segments of exactly where each flight was
                for idx, f in enumerate(env.flights):
                    if idx not in env.done:
                        episode_trajectories[idx]['x'].append(f.position.x)
                        episode_trajectories[idx]['y'].append(f.position.y)
                    
                if done_t or done_e:
                    done = True
                    break

        all_trajectories.append(episode_trajectories)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Airspace boundaries using Shapely plotting
    if env.airspace:
        x, y = env.airspace.polygon.exterior.xy
        ax.plot(x, y, color='white', linewidth=2, label='Outer Boundary')
    if env.restricted_airspace:
        x, y = env.restricted_airspace.polygon.exterior.xy
        ax.plot(x, y, color='cyan', linewidth=2, linestyle='--', label='Restricted Airspace')

    # Draw all the trajectories as highly transparent lines
    for ep_trajs in all_trajectories:
        for f_idx, traj in ep_trajs.items():
            if len(traj['x']) > 1:
                ax.plot(traj['x'], traj['y'], color='yellow', alpha=0.3, linewidth=1.5)

    ax.set_aspect('equal')
    ax.set_facecolor('black')
    #ax.set_title(f"Algorithm Pathing Map ({episodes} episodes over Fixed Geometry)", pad=20)
    ax.legend(loc='upper right')
    
    env.close()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved trajectory_heatmap.png")

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
    #plt.title("Hypothesis E: Incremental Reward Shaping Progression", pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "incremental_progress.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved incremental_progress.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['airspace-sweep', 'density-sweep', 'uncertainty-ablation', 'reward-progress', 'heatmap', 'all'])
    parser.add_argument('--run-dir', required=False, help="Path to run directory containing best_model/best_model.zip")
    parser.add_argument('--episodes', default=100, type=int, help="Number of episodes to run per condition")
    parser.add_argument('--incremental-dir', default=None, help="Tensorboard log dir for stepwise training (Hypothesis E)")
    parser.add_argument('--baseline-dir', default=None, help="Tensorboard log dir for baseline training (Hypothesis E)")
    parser.add_argument('--baseline-model', default=None, help="Path to baseline model dir (e.g. Adam's 30-dim obs model)")
    args = parser.parse_args()
    
    # Set academic plot defaults
    matplotlib.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    
    out_dir = "results/hypotheses_plots"
    if args.run_dir:
        out_dir = os.path.join(args.run_dir, "hypotheses_plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Artifacts will be stored in {out_dir}")

    if args.mode in ['reward-progress', 'all']:
        if args.incremental_dir or args.baseline_dir:
            plot_reward_progress(args.incremental_dir, args.baseline_dir, out_dir)
        elif args.mode == 'reward-progress':
            print("Error: Must provide --incremental-dir and --baseline-dir for this plot.")
            
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
        model = SAC.load(model_path)
        
        # Auto-detect and set global norm flag for workers
        obs_size = model.observation_space.shape[0]
        if obs_size <= 30:
            _USE_LITE_NORM = True
            print(f"Detected baseline model (obs_size={obs_size}), using lite normalizer")
        else:
            print(f"Detected full model (obs_size={obs_size}), using standard normalizer")
        
        if out_dir == "results/hypotheses_plots" and args.baseline_model:
            out_dir = os.path.join(run_dir, "hypotheses_plots")
            os.makedirs(out_dir, exist_ok=True)
        
        if args.mode in ['airspace-sweep', 'all']:
            sweep_airspace(model_path, out_dir, args.episodes)
        if args.mode in ['density-sweep', 'all']:
            sweep_density(model_path, out_dir, args.episodes)
        if args.mode in ['uncertainty-ablation', 'all']:
            sweep_uncertainties(model_path, out_dir, args.episodes)
        if args.mode in ['heatmap', 'all']:
            model = SAC.load(model_path)
            generate_heatmap(model, out_dir, args.episodes)

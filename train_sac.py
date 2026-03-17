"""
SAC training script for the atcenv air traffic conflict resolution environment.

Algorithm choice based on Paper 6 (Badea et al.): SAC with continuous
heading + speed actions, 2 closest intruders in observation.

# USE THE NUMBER OF CORES YOUR CPU HAS
Usage:
    python train_sac.py --timesteps 500000 --num-flights 10 --num-envs 8 
    
    python train_sac.py --timesteps 200000 --num-flights 10 --run-name "test2_03drift_40conflict_ALL_AGENTS" --train-all

    python train_sac.py --timesteps 200000 --num-flights 10 --num-envs 8 --run-name "test_03drift_40conflict_ALL_AGENTS" --train-all

    python train_sac.py --run-name run_1_baseline
"""
import argparse
import os
import numpy as np
import torch
from datetime import datetime
print("Is PyTorch using GPU?", torch.cuda.is_available())
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import Vector Environment wrappers for multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from atcenv.sb3_wrapper import (
    ATCEnvWrapper, ACTION_FREQUENCY, OBS_SIZE,
    INTRUDER_DIST_NORM, INTRUDER_POS_NORM, TARGET_DIST_NORM,
)


def save_config(args, run_dir):
    """Save all training settings to a text file for experiment tracking."""
    save_path = os.path.join(run_dir, "training_config.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    import inspect
    from atcenv.env import Environment
    reward_source = inspect.getsource(Environment.reward)

    with open(save_path, "w") as f:
        f.write(f"Training Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run name: {args.run_name if args.run_name else 'timestamp-based'}\n\n")

        f.write(f"--- Training Parameters ---\n")
        f.write(f"Mode:            {'ALL AGENTS (Deploy-all)' if getattr(args, 'train_all', False) else 'SINGLE ACTOR (Baseline)'}\n")
        f.write(f"Timesteps:       {args.timesteps}\n")
        f.write(f"Num flights:     {args.num_flights}\n")
        f.write(f"Num envs:        {args.num_envs}\n")
        f.write(f"Algorithm:       SAC (MlpPolicy)\n")
        f.write(f"Learning rate:   1e-3\n")
        f.write(f"Buffer size:     100,000\n")
        f.write(f"Batch size:      256\n")
        f.write(f"Gamma:           0.99\n")
        f.write(f"Tau:             0.005\n")
        f.write(f"Train freq:      8\n")
        f.write(f"Gradient steps:  8\n")
        f.write(f"Entropy coeff:   auto\n\n")

        f.write(f"--- Observation Normalization ---\n")
        f.write(f"Intruder distance center: {INTRUDER_DIST_NORM} m\n")
        f.write(f"Intruder distance scale:  {INTRUDER_DIST_NORM * 0.3} m\n")
        f.write(f"Intruder position scale:  {INTRUDER_POS_NORM} m\n")
        f.write(f"Target distance scale:    {TARGET_DIST_NORM} m\n")
        f.write(f"Speed center:             230.0 m/s\n")
        f.write(f"Speed scale:              30.0 m/s\n")
        f.write(f"Observation size:         {OBS_SIZE}\n")
        f.write(f"Action frequency:         {ACTION_FREQUENCY}\n\n")

        f.write(f"--- Reward Function ---\n")
        for line in reward_source.split('\n'):
            stripped = line.strip()
            if any(kw in stripped for kw in ['drift', 'conflict', 'target', 'tot_reward']):
                if not stripped.startswith('#'):
                    f.write(f"  {stripped}\n")
        f.write(f"\n")

        f.write(f"--- Heading / Speed Limits ---\n")
        f.write(f"Max heading change:  22.5 deg per action\n")
        f.write(f"Speed change:        (max_speed - min_speed) / 10 per action\n")

    print(f"Config saved to {save_path}")


def make_env(num_flights: int = 10, **kwargs):
    """Create a monitored environment instance."""
    env = ATCEnvWrapper(num_flights=num_flights, **kwargs)
    return Monitor(env)


def train(args):
    # Determine the run directory based on the run name or timestamp
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    run_dir = f"./results/{run_name}"
    print(f"Starting training run: {run_name}")
    print(f"All outputs will be saved to: {run_dir}")

    # Save training config for experiment tracking
    save_config(args, run_dir)

    print(f"Is PyTorch using GPU? {torch.cuda.is_available()}")
    
    # Window positions for distributed display
    positions = [(i % 4 * 400, i // 4 * 300) for i in range(8)]
    
    if args.train_all:
        print(f"Deploying shared policy over ALL {args.num_flights} active flights simultaneously.")
        
        if args.num_envs > 1:
            print(f"Parallelizing Shared Policy across {args.num_envs} CPU Cores (Effective Batch: {args.num_envs * args.num_flights} experiences/step)")
            from atcenv.multi_agent_wrapper import SharedPolicyVecEnv, SubprocMultiAgentVecEnv
            from stable_baselines3.common.vec_env import VecMonitor
            
            env_fns = [lambda i=i: SharedPolicyVecEnv(num_flights=args.num_flights, window_pos=positions[i]) for i in range(args.num_envs)]
            base_env = SubprocMultiAgentVecEnv(env_fns, num_flights=args.num_flights)
            train_env = VecMonitor(base_env)
            
            # Eval environment only needs 1 core
            eval_env = VecMonitor(SharedPolicyVecEnv(num_flights=args.num_flights))
            
            effective_envs = args.num_envs * args.num_flights
        else:
            print(f"Running Shared Policy on 1 CPU Core (Effective Batch: {args.num_flights} experiences/step)")
            from atcenv.multi_agent_wrapper import SharedPolicyVecEnv
            from stable_baselines3.common.vec_env import VecMonitor
            
            # SharedPolicyVecEnv inherently behaves as a VecEnv of size N
            base_env = SharedPolicyVecEnv(num_flights=args.num_flights)
            train_env = VecMonitor(base_env)
            eval_env = VecMonitor(SharedPolicyVecEnv(num_flights=args.num_flights))
            
            effective_envs = args.num_flights
    else:
        print(f"Using single-actor baseline on {args.num_envs} CPU cores.")
        env_fns = [lambda i=i: make_env(num_flights=args.num_flights, window_pos=positions[i]) for i in range(args.num_envs)]
        train_env = SubprocVecEnv(env_fns)
        eval_env = DummyVecEnv([lambda: make_env(num_flights=args.num_flights)])
        effective_envs = args.num_envs

    # 4. Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,             # reference uses 1e-3
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        
        train_freq=8,
        gradient_steps=8,
        
        ent_coef="auto",
        verbose=1,
        tensorboard_log=f"{run_dir}/tensorboard/",
    )

    eval_freq_adjusted = max(5000 // effective_envs, 1)
    save_freq_adjusted = max(10000 // effective_envs, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{run_dir}/best_model/",
        log_path=f"{run_dir}/eval_logs/",
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path=f"{run_dir}/checkpoints/",
        name_prefix="sac_atc",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(f"{run_dir}/sac_atc_final")
    print(f"Training complete. Model saved to {run_dir}/sac_atc_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on ATC environment")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-flights", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=4) 
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for the results folder. If not provided, a timestamp is used.")
    parser.add_argument("--train-all", action="store_true",
                        help="Train with ALL agents simultaneously (Parameter Sharing) instead of a single actor")
    
    args = parser.parse_args()
    train(args)
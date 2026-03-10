"""
SAC training script for the atcenv air traffic conflict resolution environment.

Algorithm choice based on Paper 6 (Badea et al.): SAC with continuous
heading + speed actions, 2 closest intruders in observation.

# USE THE NUMBER OF CORES YOUR CPU HAS
Usage:
    python train_sac.py --timesteps 500000 --num-flights 10 --num-envs 6 
"""
import argparse
import numpy as np
import torch
print("Is PyTorch using GPU?", torch.cuda.is_available())
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import Vector Environment wrappers for multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from atcenv.sb3_wrapper import ATCEnvWrapper


def make_env(num_flights: int = 10, **kwargs):
    """Create a monitored environment instance."""
    env = ATCEnvWrapper(num_flights=num_flights, **kwargs)
    return Monitor(env)


def train(args):
    print(f"Is PyTorch using GPU? {torch.cuda.is_available()}")
    
    # 1. Create a function that creates environments (required for multiprocessing)
    def make_env_fn():
        return make_env(num_flights=args.num_flights)

    # 2. Create MULTIPLE parallel environments for training (Massive speedup)
    # This runs N environments on N separate CPU cores simultaneously
    train_env = SubprocVecEnv([make_env_fn for _ in range(args.num_envs)])

    # 3. Create a single evaluation environment
    # Note: EvalCallback requires eval_env to be wrapped in DummyVecEnv if train_env is vectorized
    eval_env = DummyVecEnv([make_env_fn])

    # 4. Create SAC model
    model = SAC(
        "MlpPolicy",
        train_env,
        device="cpu",                 # <-- UNCOMMENT THIS LINE if you want to test if CPU is faster than GPU for your small network
        learning_rate=1e-3,             # reference uses 1e-3
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        
        # SPEEDUP TWEAK: Take 8 steps in the envs, then do 8 back-to-back GPU updates.
        # This stops the CPU and GPU from getting stuck in traffic talking to each other.
        train_freq=8,
        gradient_steps=8,
        
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./results/tensorboard/",
    )

    # Note on Callbacks with Vector Environments: 
    # Because we take `args.num_envs` steps at a time, we must divide the frequency 
    # by `args.num_envs` so it still evaluates at the exact same absolute timesteps.
    eval_freq_adjusted = max(5000 // args.num_envs, 1)
    save_freq_adjusted = max(10000 // args.num_envs, 1)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./results/best_model/",
        log_path="./results/eval_logs/",
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path="./results/checkpoints/",
        name_prefix="sac_atc",
    )

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save("./results/sac_atc_final")
    print("Training complete. Model saved to ./results/sac_atc_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on ATC environment")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-flights", type=int, default=10)
    # Added argument to control how many CPU cores to use for environment generation
    parser.add_argument("--num-envs", type=int, default=4) 
    
    args = parser.parse_args()
    train(args)
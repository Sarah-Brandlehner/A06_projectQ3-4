"""
SAC training script for the atcenv air traffic conflict resolution environment.

Algorithm choice based on Paper 6 (Badea et al.): SAC with continuous
heading + speed actions, 2 closest intruders in observation.

Usage:
    python train_sac.py --timesteps 50000 --num-flights 10
"""
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from atcenv.sb3_wrapper import ATCEnvWrapper


def make_env(num_flights: int = 10, **kwargs):
    """Create a monitored environment instance."""
    env = ATCEnvWrapper(num_flights=num_flights, **kwargs)
    return Monitor(env)

a=1
def train(args):
    # Create environments
    train_env = make_env(num_flights=args.num_flights)
    eval_env = make_env(num_flights=args.num_flights)

    # Create SAC model — vanilla MlpPolicy like the reference
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,             # reference uses 1e-3
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./results/tensorboard/",
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./results/best_model/",
        log_path="./results/eval_logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
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
    args = parser.parse_args()
    train(args)

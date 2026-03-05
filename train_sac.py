"""
SAC training script for the atcenv air traffic conflict resolution environment.

Algorithm choice based on Paper 6 (Badea et al.): SAC with continuous
heading + speed actions, 2 closest intruders in observation.

Usage:
    python train_sac.py --timesteps 500000 --num-flights 10
"""
import argparse
import os
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from atcenv.sb3_wrapper import (
    ATCEnvWrapper, ACTION_FREQUENCY, OBS_SIZE,
    INTRUDER_DIST_NORM, INTRUDER_POS_NORM, TARGET_DIST_NORM,
)


def save_config(args, save_path="./results/training_config.txt"):
    """Save all training settings to a text file for experiment tracking."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Read reward constants from env.py source
    import inspect
    from atcenv.env import Environment
    reward_source = inspect.getsource(Environment.reward)

    with open(save_path, "w") as f:
        f.write(f"Training Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"--- Training Parameters ---\n")
        f.write(f"Timesteps:       {args.timesteps}\n")
        f.write(f"Num flights:     {args.num_flights}\n")
        f.write(f"Algorithm:       SAC (MlpPolicy)\n")
        f.write(f"Learning rate:   1e-3\n")
        f.write(f"Buffer size:     100,000\n")
        f.write(f"Batch size:      256\n")
        f.write(f"Gamma:           0.99\n")
        f.write(f"Tau:             0.005\n")
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
        # Extract key lines from the reward source
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
    # Save training config for experiment tracking
    save_config(args)

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

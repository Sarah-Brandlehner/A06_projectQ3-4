"""
SAC training script for the atcenv air traffic conflict resolution environment.

Algorithm choice based on Paper 6 (Badea et al.): SAC with continuous
heading + speed actions, 2 closest intruders in observation.

# USE THE NUMBER OF CORES YOUR CPU HAS
Usage:
    
    python train_sac.py --timesteps 300000 --num-flights 10 --num-envs 8 --train-all --run-name "4_intruders_unlocked_physics"

    python train_sac.py --timesteps 2000000 --num-flights 10 --num-envs 8 --train-all --run-name "minimal_reward_ALL_AGENTS"
    python train_sac.py --timesteps 3000000 --num-flights 10 --num-envs 8 --train-all --run-name "4_intruders_unlocked_physics"
    python train_sac.py --timesteps 1000000 --num-flights 10 --num-envs 8 --train-all --run-name "fine_5_steps_airspaces_v2" --load "results/fine_5_steps_airspaces_v2/best_model/best_model.zip"
"""
import argparse
import os
import numpy as np
import torch
from datetime import datetime
print("Is PyTorch using GPU?", torch.cuda.is_available())
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from collections import deque
from stable_baselines3.common.monitor import Monitor

# Import Vector Environment wrappers for multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from atcenv.sb3_wrapper import (
    ATCEnvWrapper, ACTION_FREQUENCY, OBS_SIZE,
    INTRUDER_DIST_NORM, INTRUDER_POS_NORM, TARGET_DIST_NORM,
)


# ─────────────────── REWARD BREAKDOWN CALLBACK ───────────────────

class RewardBreakdownCallback(BaseCallback):
    """
    Logs per-component reward means to TensorBoard every `log_freq` steps.
    Reads reward_drift / reward_conflict / reward_target / reward_proximity
    from the info dicts populated by SharedPolicyVecEnv.
    """

    def __init__(self, log_freq: int = 500, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._component_names = ["drift", "conflict", "alert", "target"]
        self._buffers = {name: deque(maxlen=window) for name in self._component_names}

    def _on_step(self) -> bool:
        # Collect reward components from the most recent infos
        infos = self.locals.get("infos", [])
        for info in infos:
            for name in self._component_names:
                key = f"reward_{name}"
                if key in info:
                    self._buffers[name].append(info[key])

        # Log rolling averages periodically
        if self.n_calls % self.log_freq == 0:
            for name in self._component_names:
                buf = self._buffers[name]
                if len(buf) > 0:
                    import numpy as _np
                    mean_val = _np.mean(buf)
                    self.logger.record(f"reward_components/{name}", mean_val)
        return True


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
            if any(kw in stripped for kw in ['drift', 'conflict', 'proximity', 'target', 'tot_reward']):
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
    
    if args.train_all:
        print(f"Deploying shared policy over ALL {args.num_flights} active flights simultaneously.")
        
        if args.num_envs > 1:
            print(f"Parallelizing Shared Policy across {args.num_envs} CPU Cores (Effective Batch: {args.num_envs * args.num_flights} experiences/step)")
            from atcenv.multi_agent_wrapper import SharedPolicyVecEnv, SubprocMultiAgentVecEnv
            from stable_baselines3.common.vec_env import VecMonitor
            
            def make_shared_env_fn():
                return SharedPolicyVecEnv(num_flights=args.num_flights)

            base_env = SubprocMultiAgentVecEnv([make_shared_env_fn for _ in range(args.num_envs)], num_flights=args.num_flights)
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
        def make_env_fn():
            return make_env(num_flights=args.num_flights)

        train_env = SubprocVecEnv([make_env_fn for _ in range(args.num_envs)])
        eval_env = DummyVecEnv([make_env_fn])
        effective_envs = args.num_envs

    # 4. Create or Load SAC model
    if args.load:
        print(f"Loading pre-trained model from {args.load}...")
        model = SAC.load(
            args.load,
            env=train_env,
            custom_objects={
                "learning_rate": 1e-5,
                "buffer_size": 1_000_000, 
                "batch_size": 1024,
                "ent_coef": 0.05,
            }
        )
        model.tensorboard_log = f"{run_dir}/tensorboard/"
        policy_kwargs = dict(net_arch=[512, 512, 512]) # Default in SB3 is [256, 256]

        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,             # 1e-3
            buffer_size=1_000_000,          # 100_000 (increased for multi-agent)
            batch_size=1024,                # 256 (increased for multi-agent)
            tau=0.005,                      # 0.005
            gamma=0.99,                     # 0.99
            learning_starts=5000,           # 1000
            
            train_freq=8,
            gradient_steps=8,
            
            ent_coef=0.05,                  # "auto" (fixed lower to force exploitation of good paths)
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

    reward_breakdown_cb = RewardBreakdownCallback(log_freq=500, window=200)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, reward_breakdown_cb],
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
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a pre-trained model .zip file to resume from.")
    
    args = parser.parse_args()
    train(args)
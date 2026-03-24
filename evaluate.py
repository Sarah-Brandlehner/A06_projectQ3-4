"""
Evaluation script for the ATC conflict resolution model.

Supports two modes:
  --deploy-all : Apply the trained policy to ALL agents (parameter sharing)
  (default)    : Single-actor evaluation (only agent 0 controlled)

Usage:
    python evaluate.py --model results/best_model/best_model.zip --episodes 10 --num-flights 5
    python evaluate.py --model results/best_model/best_model.zip --episodes 10 --num-flights 5 --deploy-all
    python evaluate.py --model results/shared_reward_4/best_model/best_model.zip --episodes 50 --num-flights 10 --deploy-all
"""
import argparse
import numpy as np
from stable_baselines3 import SAC

from atcenv.env import Environment, NUMBER_INTRUDERS_STATE
from atcenv.sb3_wrapper import (
    ATCEnvWrapper, ACTION_FREQUENCY, OBS_SIZE,
    INTRUDER_DIST_NORM, INTRUDER_POS_NORM, TARGET_DIST_NORM,
)


def normalize_obs(raw_obs):
    """Normalize a single agent's raw observation (same as wrapper)."""
    obs = np.array(raw_obs, dtype=np.float32)
    n = NUMBER_INTRUDERS_STATE

    # Intruder distances
    obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
    obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)

    # Relative dx, dy
    obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
    obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM

    # Track differences
    obs[4*n:5*n] = obs[4*n:5*n] / np.pi

    # Airspeeds
    obs[5*n]     = (obs[5*n] - 230.0) / 30.0
    obs[5*n+1]   = (obs[5*n+1] - 230.0) / 30.0

    # Distance to target
    obs[5*n+2]   = (obs[5*n+2] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)

    # Restricted airspace flags (already in [0, 1] range)
    # obs[5*n+3] and obs[5*n+4] - no normalization needed
    
    # Closest 4 vertices of restricted airspace (distance, dx, dy for each)
    # Normalize distances and positions similar to intruder data
    vertex_start = 5*n + 5
    for v in range(4):  # 4 vertices
        vertex_idx = vertex_start + v * 3
        if vertex_idx < len(obs):
            # Distance
            obs[vertex_idx] = (obs[vertex_idx] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
        if vertex_idx + 1 < len(obs):
            # dx
            obs[vertex_idx + 1] = obs[vertex_idx + 1] / INTRUDER_POS_NORM
        if vertex_idx + 2 < len(obs):
            # dy
            obs[vertex_idx + 2] = obs[vertex_idx + 2] / INTRUDER_POS_NORM

    return np.clip(obs, -1.0, 1.0).astype(np.float32)


def evaluate_single_actor(model_path: str, n_episodes: int = 10, num_flights: int = 5):
    """Standard single-actor evaluation (only agent 0 controlled)."""
    model = SAC.load(model_path)
    env = ATCEnvWrapper(num_flights=num_flights, training=False)

    total_conflicts = 0
    total_restricted_intrusions = 0
    total_targets_reached = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_conflicts = 0
        ep_restricted_intrusions = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_conflicts += len(env._env.conflicts)
            ep_restricted_intrusions += len(env._env.restricted_airspace_intrusions)
            done = terminated or truncated

        targets = len(env._env.done)
        total_conflicts += ep_conflicts
        total_restricted_intrusions += ep_restricted_intrusions
        total_targets_reached += targets
        print(f"  Episode {ep+1}: conflicts={ep_conflicts}, restricted_intrusions={ep_restricted_intrusions}, targets_reached={targets}/{num_flights}")

    env.close()

    print(f"\n=== Single-Actor Results ({n_episodes} episodes) ===")
    print(f"Avg conflicts/episode:  {total_conflicts / n_episodes:.1f}")
    print(f"Avg restricted intrusions/episode: {total_restricted_intrusions / n_episodes:.1f}")
    print(f"Avg targets reached:    {total_targets_reached / n_episodes:.1f} / {num_flights}")


def evaluate_all_agents(model_path: str, n_episodes: int = 10, num_flights: int = 5):
    """
    Deploy the trained policy to ALL agents (parameter sharing).
    Each agent gets its own observation, the same model predicts its action.
    """
    model = SAC.load(model_path)
    env = Environment(num_flights=num_flights)

    total_conflicts = 0
    total_restricted_intrusions = 0
    total_targets_reached = 0

    for ep in range(n_episodes):
        raw_obs_list = env.reset(num_flights)
        done = False
        ep_conflicts = 0
        ep_restricted_intrusions = 0

        while not done:
            n_active = len(env.flights) - len(env.done)

            # Get action for EACH active agent from the same model
            actions = np.zeros((n_active, 2), dtype=np.float32)
            agent_idx = 0
            for i in range(len(env.flights)):
                if i not in env.done:
                    obs = normalize_obs(raw_obs_list[agent_idx])
                    action, _ = model.predict(obs, deterministic=True)
                    actions[agent_idx] = action
                    agent_idx += 1

            # Step the sim ACTION_FREQUENCY times
            for step_i in range(ACTION_FREQUENCY):
                raw_obs_list, rewards, done_t, done_e, info = env.step(actions)
                if done_t or done_e:
                    done = True
                    break
                # After first step, maintain heading
                actions = np.zeros((len(env.flights) - len(env.done), 2), dtype=np.float32)

            ep_conflicts += len(env.conflicts)
            ep_restricted_intrusions += len(env.restricted_airspace_intrusions)
            if done_t or done_e:
                done = True

        targets = len(env.done)
        total_conflicts += ep_conflicts
        total_restricted_intrusions += ep_restricted_intrusions
        total_targets_reached += targets
        print(f"  Episode {ep+1}: conflicts={ep_conflicts}, restricted_intrusions={ep_restricted_intrusions}, targets_reached={targets}/{num_flights}")

    env.close()

    print(f"\n=== All-Agent Results ({n_episodes} episodes) ===")
    print(f"Avg conflicts/episode:  {total_conflicts / n_episodes:.1f}")
    print(f"Avg restricted intrusions/episode: {total_restricted_intrusions / n_episodes:.1f}")
    print(f"Avg targets reached:    {total_targets_reached / n_episodes:.1f} / {num_flights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ATC model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-flights", type=int, default=5)
    parser.add_argument("--deploy-all", action="store_true",
                        help="Apply the model to ALL agents (parameter sharing)")
    args = parser.parse_args()

    if args.deploy_all:
        print(f"Evaluating with ALL {args.num_flights} agents controlled by model...")
        evaluate_all_agents(args.model, args.episodes, num_flights=args.num_flights)
    else:
        print(f"Evaluating with single-actor (agent 0 only)...")
        evaluate_single_actor(args.model, args.episodes, num_flights=args.num_flights)

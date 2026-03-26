"""
SB3-compatible Gymnasium wrapper for the atcenv Environment.

Architecture follows bluesky-gym reference:
- Controls ONLY ONE aircraft (index 0) — the "actor"
- Other aircraft fly straight toward their waypoints (uncontrolled)
- Steps the simulation multiple times per RL action (ACTION_FREQUENCY)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from atcenv.env import Environment, NUMBER_INTRUDERS_STATE

# Number of sim steps per RL action (reference uses 5-10)
ACTION_FREQUENCY = 5

# Observation size: 7 * NUMBER_INTRUDERS_STATE + 10
# (7 relative state vectors per intruder + 5 ownship values + 2 restricted airspace flags + 3 closest vertex values)
OBS_SIZE = 7 * NUMBER_INTRUDERS_STATE + 10

# Normalization constants (matched to reference bluesky-gym ranges)
INTRUDER_DIST_NORM = 50000.0   # ~27 NM — intruder distances
INTRUDER_POS_NORM = 13000.0    # ~7 NM — relative dx/dy positions (ref: /13000)
TARGET_DIST_NORM = 200000.0    # ~108 NM — target can be far
SPEED_NORM = 300.0             # m/s — aircraft speed normalization


class ATCEnvWrapper(gym.Env):
    """
    Single-actor wrapper: one RL policy controls aircraft 0.
    Other aircraft maintain their initial heading toward waypoints.

    Action: [heading_change, speed_change] in [-1, 1]
      - heading_change scaled by ±22.5 degrees
      - speed_change scaled by ±6.67 kts
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_flights=10, training=True, **env_kwargs):
        super().__init__()
        self.num_flights = num_flights
        self.training = training
        self.env_kwargs = env_kwargs

        # Underlying multi-agent environment
        self._env = Environment(num_flights=self.num_flights, **env_kwargs)

        # SB3 requires fixed observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        # Action: [heading_change, speed_change] both in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def _normalize_obs(self, raw_obs):
        """
        Normalize raw observation vector to approx [-1, 1] range.

        Raw obs layout (with N = NUMBER_INTRUDERS_STATE):
          [0:N]     distances to N closest intruders (meters)
          [N:2N]    predicted distances to N closest intruders (meters)
          [2N:3N]   relative dx to N closest intruders (meters)
          [3N:4N]   relative dy to N closest intruders (meters)
          [4N:5N]   track differences to N closest intruders (radians)
          [5N:6N]   relative vx to N closest intruders (m/s)
          [6N:7N]   relative vy to N closest intruders (m/s)
          [7N]      current airspeed (m/s)
          [7N+1]    optimal airspeed (m/s)
          [7N+2]    distance to target (meters)
          [7N+3]    sin(drift angle)
          [7N+4]    cos(drift angle)
        """
        obs = np.array(raw_obs, dtype=np.float32)

        n = NUMBER_INTRUDERS_STATE

        # Intruder distances: center around typical separation, scale tightly
        obs[0:n]     = (obs[0:n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
        obs[n:2*n]   = (obs[n:2*n] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)

        # Relative dx, dy: normalize by 13km
        obs[2*n:3*n] = obs[2*n:3*n] / INTRUDER_POS_NORM
        obs[3*n:4*n] = obs[3*n:4*n] / INTRUDER_POS_NORM

        # Track differences: already in [-pi, pi], normalize to [-1, 1]
        obs[4*n:5*n] = obs[4*n:5*n] / np.pi

        # Relative velocities: normalize by max speed
        obs[5*n:6*n] = obs[5*n:6*n] / SPEED_NORM
        obs[6*n:7*n] = obs[6*n:7*n] / SPEED_NORM

        # Airspeeds: center around typical speed (~230 m/s, which is ~450 kt)
        obs[7*n]     = (obs[7*n] - 230.0) / 30.0
        obs[7*n+1]   = (obs[7*n+1] - 230.0) / 30.0

        # Distance to target: normalize by larger range (targets can be far)
        obs[7*n+2]   = (obs[7*n+2] - TARGET_DIST_NORM * 0.5) / (TARGET_DIST_NORM * 0.5)

        # sin/cos drift: already in [-1, 1]
        
        # Restricted airspace flags (already in 0 to 1) - indices 7*n+5, 7*n+6
        
        # Closest 1 point of restricted airspace (dist, dx, dy)
        point_start = 7*n + 7
        if point_start < len(obs):
            obs[point_start] = (obs[point_start] - INTRUDER_DIST_NORM) / (INTRUDER_DIST_NORM * 0.3)
            obs[point_start+1] = obs[point_start+1] / INTRUDER_POS_NORM
            obs[point_start+2] = obs[point_start+2] / INTRUDER_POS_NORM

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset and return the actor's (agent 0) observation."""
        raw_obs_list = self._env.reset(self.num_flights)
        obs = self._normalize_obs(raw_obs_list[0])
        return obs, {}

    def step(self, action):
        """
        Single-actor step:
        1. Build action array: actor (index 0) gets the RL action,
           all other aircraft get [0, 0] (maintain heading/speed).
        2. Step the sim ACTION_FREQUENCY times for meaningful advancement.
        3. Return actor's observation and reward.
        """
        n_agents = len(self._env.flights) - len(self._env.done)

        # Build action array: only actor (index 0) is controlled
        actions = np.zeros((n_agents, 2), dtype=np.float32)
        actions[0] = action  # Only aircraft 0 is controlled by the RL agent

        # Step simulation ACTION_FREQUENCY times, accumulating rewards
        accumulated_reward = 0.0
        for _ in range(ACTION_FREQUENCY):
            raw_obs_list, rewards, done_t, done_e, info = self._env.step(actions)
            # Accumulate actor's reward across all sub-steps
            if len(rewards) > 0:
                accumulated_reward += float(rewards[0])
            if done_t or done_e:
                break

        # Actor's accumulated reward across all sub-steps
        actor_reward = accumulated_reward

        # Termination
        terminated = done_e  # all aircraft reached target
        truncated = done_t   # max episode length

        # Actor's observation
        if raw_obs_list:
            obs = self._normalize_obs(raw_obs_list[0])
        else:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)

        return obs, actor_reward, terminated, truncated, info

    def close(self):
        self._env.close()

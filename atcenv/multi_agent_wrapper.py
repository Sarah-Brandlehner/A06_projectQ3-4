"""
Gymnasium wrapper to train with ALL agents simultaneously using parameter sharing.

Instead of controlling just aircraft 0 (like sb3_wrapper.py does), this wrapper
vectorizes the environment internally. Every step, it:
1. Takes N actions (one for each active aircraft)
2. Steps the simulation ACTION_FREQUENCY times
3. Returns N observations, N rewards, and N done flags

Stable Baselines 3 automatically treats this as a batch of size N, learning from
everything each aircraft does using a single shared neural network.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atcenv.env import Environment, NUMBER_INTRUDERS_STATE
from atcenv.sb3_wrapper import (
    ACTION_FREQUENCY, OBS_SIZE,
    INTRUDER_DIST_NORM, INTRUDER_POS_NORM, TARGET_DIST_NORM
)


class MultiAgentATCWrapper(gym.Env):
    """
    Multi-actor wrapper for parameter sharing.
    Internally behaves like a vectorized environment of varying size.
    BUT because SB3's VecEnv interface requires a FIXED number of environments,
    we have to pad the observations/actions to always be `num_flights` size,
    and simply zero out the padding for 'done' agents.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_flights=10, training=True, **env_kwargs):
        super().__init__()
        self.num_flights = num_flights
        self.training = training
        self.env_kwargs = env_kwargs

        # Underlying multi-agent environment
        self._env = Environment(num_flights=self.num_flights, **env_kwargs)



        raise NotImplementedError("SB3 architecture requires this to subclass VecEnv to work correctly with parameter sharing, which is being handled in the simplified version below.")


import stable_baselines3.common.vec_env as vec_env

class SharedPolicyVecEnv(vec_env.VecEnv):
    """
    A custom VecEnv that wraps a SINGLE ATC environment, but tells SB3
    that it is actually `num_flights` separate parallel environments.

    This forces SB3 to apply its single-agent neural network individually 
    to every single aircraft in the simulation (Parameter Sharing).
    """

    def __init__(self, num_flights=10, **env_kwargs):
        self.num_flights = num_flights
        self._env = Environment(num_flights=num_flights, **env_kwargs)
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        super().__init__(self.num_flights, self.observation_space, self.action_space)

        # Buffers for SB3
        self.buf_obs = np.zeros((self.num_flights, OBS_SIZE), dtype=np.float32)
        self.buf_dones = np.zeros(self.num_flights, dtype=bool)
        self.buf_rews = np.zeros(self.num_flights, dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_flights)]

        # Keep track of rewards accumulated over ACTION_FREQUENCY sub-steps
        self.accumulated_rewards = np.zeros(self.num_flights, dtype=np.float32)

        # Per-component reward accumulators for tracking
        self._component_names = ["drift", "conflict", "alert", "target"]
        self._accumulated_components = {
            name: np.zeros(self.num_flights, dtype=np.float32)
            for name in self._component_names
        }

    def _normalize_obs(self, raw_obs):
        """Normalize a single observation vector using sb3_wrapper constants."""
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

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def step_async(self, actions):
        """
        SB3 calls this first. Actions shape is (num_flights, 2).
        For aircraft that are 'done', the RL agent will still output an action, 
        but we must ignore it and force action [0,0].
        """
        self.current_actions = np.zeros((self.num_flights, 2), dtype=np.float32)
        
        for i in range(self.num_flights):
            if not self.buf_dones[i]:
                self.current_actions[i] = actions[i]

    def step_wait(self):
        """SB3 calls this second. We execute the actions in the env here."""
        
        self.accumulated_rewards.fill(0.0)
        for name in self._component_names:
            self._accumulated_components[name].fill(0.0)
        
        # We must track actual environment dones separately, 
        # because SB3 expects an env to instantly reset when done.
        episode_terminated = False
        episode_truncated = False

        # Step simulation ACTION_FREQUENCY times
        for step_i in range(ACTION_FREQUENCY):
            # Build actions for the underlying env (only for *actually active* agents)
            # Must rebuild every inner step because if an agent finishes, the active list shifts!
            active_indices = [i for i in range(self.num_flights) if i not in self._env.done]
            env_actions = np.zeros((len(active_indices), 2), dtype=np.float32)
            for idx, agent_num in enumerate(active_indices):
                env_actions[idx] = self.current_actions[agent_num]

            raw_obs_list, rewards, done_t, done_e, info = self._env.step(env_actions)
            
            # Map rewards back to absolute indexing
            if len(rewards) > 0:
                for idx, agent_num in enumerate(active_indices):
                    self.accumulated_rewards[agent_num] += float(rewards[agent_num])

            # Accumulate per-component rewards
            components = self._env.reward_components()
            for name in self._component_names:
                for idx, agent_num in enumerate(active_indices):
                    self._accumulated_components[name][agent_num] += float(components[name][agent_num])

            if done_t or done_e:
                episode_terminated = done_e
                episode_truncated = done_t
                break


        # Assign accumulated rewards to the buffer
        self.buf_rews[:] = self.accumulated_rewards

        # Write per-component rewards into info dicts
        for i in range(self.num_flights):
            for name in self._component_names:
                self.buf_infos[i][f"reward_{name}"] = float(self._accumulated_components[name][i])

        # Calculate who is still active AFTER the ACTION_FREQUENCY steps
        active_indices_now = [i for i in range(self.num_flights) if i not in self._env.done]

        if episode_terminated or episode_truncated:
            # 1. Grab terminal observations
            for agent_num in range(self.num_flights):
                if agent_num in active_indices_now and raw_obs_list:
                    idx = active_indices_now.index(agent_num)
                    term_obs = self._normalize_obs(raw_obs_list[idx])
                else:
                    term_obs = self.buf_obs[agent_num]
                self.buf_infos[agent_num]["terminal_observation"] = term_obs
                    
            # 2. Reset the environment internally
            raw_obs_list = self._env.reset(self.num_flights)
            
            # 3. We MUST return True for `done` this step so SB3 knows the episode ended.
            # We construct a return array where done is True for EVERY agent.
            return_dones = np.ones(self.num_flights, dtype=bool)
            
            # 4. We MUST return the NEW reset observations for the next episode.
            for i in range(self.num_flights):
                self.buf_obs[i] = self._normalize_obs(raw_obs_list[i])
            
            # 5. Internally clear buf_dones for the NEXT step.
            self.buf_dones.fill(False)
            
            return self.buf_obs.copy(), self.buf_rews.copy(), return_dones, list(self.buf_infos)

        else:
            # Episode continues. Check if individual agents finished in this step.
            for i in range(self.num_flights):
                was_done = self.buf_dones[i]
                is_done_now = (i in self._env.done)
                
                if is_done_now and not was_done:
                    # Agent just finished! This is a terminal state for *this* agent.
                    self.buf_dones[i] = True
                    self.buf_obs[i] = np.zeros(OBS_SIZE, dtype=np.float32)
                    self.buf_infos[i]["terminal_observation"] = np.zeros(OBS_SIZE, dtype=np.float32)

            # 4. Fill buf_obs with current active observations
            for idx, agent_num in enumerate(active_indices_now):
                if raw_obs_list:
                    self.buf_obs[agent_num] = self._normalize_obs(raw_obs_list[idx])
            
            # For agents that are already done, keep outputting zeros and 'done=True'
            for i in range(self.num_flights):
                if i in self._env.done:
                    self.buf_dones[i] = True
                    self.buf_rews[i] = 0.0  # no more rewards
                    self.buf_obs[i] = np.zeros(OBS_SIZE, dtype=np.float32)

            # Return the buffers as SB3 expects
            # For ongoing episodes, we just return buf_dones as is
            return self.buf_obs.copy(), self.buf_rews.copy(), self.buf_dones.copy(), list(self.buf_infos)


    def reset(self):
        """SB3 calls this to start. Must return initial states."""
        raw_obs_list = self._env.reset(self.num_flights)
        
        self.buf_dones.fill(False)
        self.buf_rews.fill(0.0)
        
        for i in range(self.num_flights):
            self.buf_obs[i] = self._normalize_obs(raw_obs_list[i])
            self.buf_infos[i] = {}
            
        return self.buf_obs.copy()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._env, attr_name) for _ in range(self.num_flights)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_flights
    
    def close(self):
        self._env.close()


# =========================================================================
# MULTIPROCESSING SUPPORT FOR PARAMETER SHARING
# =========================================================================

import multiprocessing as mp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper

def _worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process for SubprocMultiAgentVecEnv.
    Runs one SharedPolicyVecEnv (which internally handles `num_flights` agents).
    """
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocMultiAgentVecEnv(VecEnv):
    """
    A custom wrapper that parallelizes `SharedPolicyVecEnv` across CPU cores.
    
    SB3 cannot normally parallelize VecEnvs (it expects standard Envs inside 
    SubprocVecEnv). This class intercepts the (num_flights, obs_dim) outputs from 
    each Core, and flattens them into a single massive 1D tuple for SB3:
    `total_envs = num_cores * num_flights`
    """
    def __init__(self, env_fns, num_flights):
        self.waiting = False
        self.closed = False
        self.n_envs_per_worker = num_flights
        n_workers = len(env_fns)
        
        # Total effective environments = workers * flights
        self.total_envs = n_workers * num_flights

        # Start workers
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_workers)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # using spawn or fork based on default OS settings
            process = mp.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        
        super().__init__(self.total_envs, observation_space, action_space)

    def step_async(self, actions):
        # actions is shape (total_envs, action_dim)
        # We must split it into (n_workers, num_flights, action_dim)
        actions_split = np.split(actions, len(self.remotes))
        for remote, action in zip(self.remotes, actions_split):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        
        # each `obs` is (num_flights, obs_dim) -> concatenate to (total_envs, obs_dim)
        flat_obs = np.concatenate(obs, axis=0)
        flat_rews = np.concatenate(rews, axis=0)
        flat_dones = np.concatenate(dones, axis=0)
        
        # infos is tuple of lists of dicts
        flat_infos = []
        for info_list in infos:
            flat_infos.extend(info_list)
            
        return flat_obs, flat_rews, flat_dones, flat_infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.concatenate(obs, axis=0)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        return [None] * self.total_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.total_envs


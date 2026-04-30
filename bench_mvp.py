"""
Fast headless benchmark harness for the MVP resolver.

- Patches Environment.render to a no-op (no pygame window).
- Runs N episodes with fixed seeds for reproducibility.
- Reports conflict-free %, intrusion-free %, mean targets, mean drift,
  and wiggle proxies (heading reversals, mean |heading action|).

Usage:
    python bench_mvp.py                              # default: 100 eps, 10 flights, improved resolver
    python bench_mvp.py --resolver original          # use the baseline v0 resolver
    python bench_mvp.py --episodes 60 --num-flights 8 --workers 6
    python bench_mvp.py --tag myrun --resolver improved
"""
import argparse
import os
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Patch render before anything else imports env
import atcenv.env as _env_mod
_env_mod.Environment.render = lambda self: None

from atcenv.env import Environment
from atcenv.sb3_wrapper import ACTION_FREQUENCY


def _get_resolver(version):
    if version == "original":
        from atcenv.mvp_resolver_v0 import mvp_actions_for_env
    else:
        from atcenv.mvp_resolver import mvp_actions_for_env
    return mvp_actions_for_env


def _run_one(args):
    seed, num_flights, resolver_version = args
    random.seed(seed)
    np.random.seed(seed)

    mvp_actions_for_env = _get_resolver(resolver_version)

    env = Environment(num_flights=num_flights, random_init_heading=True)
    env.render = lambda: None

    env.reset(num_flights)
    done = False
    ep_conflicts = 0
    ep_intrusions = 0
    ep_drift = 0.0
    step = 0

    prev_action_sign = np.zeros(num_flights, dtype=np.int8)
    reversals = np.zeros(num_flights, dtype=np.int32)
    abs_action_sum = 0.0
    abs_action_count = 0

    while not done:
        actions = mvp_actions_for_env(env)
        active_idx = [i for i in range(len(env.flights)) if i not in env.done]
        for slot, i in enumerate(active_idx):
            a = float(actions[slot, 0])
            sign = 1 if a > 0.05 else (-1 if a < -0.05 else 0)
            if sign != 0 and prev_action_sign[i] != 0 and sign != prev_action_sign[i]:
                reversals[i] += 1
            if sign != 0:
                prev_action_sign[i] = sign
            abs_action_sum += abs(a)
            abs_action_count += 1

        for _ in range(ACTION_FREQUENCY):
            _, _, done_t, done_e, _ = env.step(actions)
            if done_t or done_e:
                done = True
                break
            if actions.shape[0] != len(env.flights) - len(env.done):
                actions = mvp_actions_for_env(env)

        ep_conflicts += len(env.conflicts)
        ep_intrusions += len(env.restricted_airspace_intrusions)
        for i, f in enumerate(env.flights):
            if i not in env.done:
                ep_drift += abs(f.drift)
        step += 1

    targets_reached = len(env.done)
    env.close()

    return {
        "conflicts": ep_conflicts,
        "intrusions": ep_intrusions,
        "targets_reached": targets_reached,
        "drift": ep_drift,
        "steps": step,
        "reversals": int(reversals.sum()),
        "mean_abs_action": (abs_action_sum / abs_action_count) if abs_action_count else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-flights", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seed-base", type=int, default=20260430)
    p.add_argument("--tag", type=str, default="MVP")
    p.add_argument("--resolver", type=str, default="improved", choices=["improved", "original"],
                   help="'improved' = v4 (path-bias + bug fixes), 'original' = v0 baseline")
    args = p.parse_args()

    seeds = [args.seed_base + ep for ep in range(args.episodes)]
    tasks = [(s, args.num_flights, args.resolver) for s in seeds]

    if args.workers <= 1:
        results = [_run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_run_one, tasks))

    n = args.episodes
    nf = args.num_flights
    conflict_free = sum(1 for r in results if r["conflicts"] == 0)
    intrusion_free = sum(1 for r in results if r["intrusions"] == 0)
    mean_targets = np.mean([r["targets_reached"] for r in results])
    mean_drift = np.mean([r["drift"] for r in results])
    mean_conflicts = np.mean([r["conflicts"] for r in results])
    mean_intrusions = np.mean([r["intrusions"] for r in results])
    mean_reversals = np.mean([r["reversals"] for r in results])
    mean_abs_a = np.mean([r["mean_abs_action"] for r in results])

    print(f"\n=== {args.tag}  [{args.resolver}]  ({n} eps × {nf} flights) ===")
    print(f"  conflict-free      : {conflict_free}/{n}  ({100*conflict_free/n:.1f}%)")
    print(f"  intrusion-free     : {intrusion_free}/{n}  ({100*intrusion_free/n:.1f}%)")
    print(f"  mean targets       : {mean_targets:.2f} / {nf}")
    print(f"  mean drift         : {mean_drift:.2f}")
    print(f"  mean conflicts     : {mean_conflicts:.2f}")
    print(f"  mean intrusions    : {mean_intrusions:.2f}")
    print(f"  heading reversals  : {mean_reversals:.1f}  (per ep, all agents)  -- wiggle proxy")
    print(f"  mean |heading a|   : {mean_abs_a:.3f}")


if __name__ == "__main__":
    main()

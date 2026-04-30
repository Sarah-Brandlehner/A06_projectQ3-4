"""
MVP Resolver — original / baseline version (v0).

This is the resolver as it existed before the wiggle-reduction work.
Kept for comparison; use mvp_resolver.py for the improved version.

Key constants:
  ACTION_FREQUENCY (5)   : env.step is called this many times per RL decision.
  min_dist  (11112 m)    : protected zone, 6 nm.
  lookahead (240 s)      : only resolve conflicts predicted within this CPA window.
  burden_share (0.5)     : assume both aircraft maneuver, take half each.
  restricted weight      : MVP also pushes away from the restricted airspace polygon.

Known issues (fixed in mvp_resolver.py):
  * When inside the restricted polygon the radial push points deeper inside
    rather than toward the nearest exit.
  * No pre-emptive path-bend around the polygon at long range.
"""
import math
import numpy as np

ACTION_FREQUENCY = 5
HEADING_SCALE_RAD = math.radians(22.5)
SPEED_DELTA_PER_TICK = (500 - 400) * 0.51444 / 3


def mvp_resolver(
    flight,
    intruders,
    min_dist=11112,
    lookahead=240.0,
    burden_share=0.5,
    restricted_airspace=None,
    restricted_buffer=11112,
    restricted_weight=2.0,
):
    # 1. ATTRACTIVE FORCE (toward target)
    dx_t = flight.target.x - flight.position.x
    dy_t = flight.target.y - flight.position.y
    dist_t = math.hypot(dx_t, dy_t)
    if dist_t < 1e-6:
        return [0.0, 0.0]

    v_goal_x = (dx_t / dist_t) * flight.optimal_airspeed
    v_goal_y = (dy_t / dist_t) * flight.optimal_airspeed

    # 2. REPULSIVE FORCES from intruders (predicted CPA)
    v_repulse_x = 0.0
    v_repulse_y = 0.0
    in_active_conflict = False

    v1x, v1y = flight.components
    own_id = id(flight)
    proximity_zone = 2.0 * min_dist
    closest_intruder_dist = float("inf")
    for intruder in intruders:
        rx = flight.position.x - intruder.position.x
        ry = flight.position.y - intruder.position.y
        cur_dist = math.hypot(rx, ry)
        if cur_dist < closest_intruder_dist:
            closest_intruder_dist = cur_dist

        if cur_dist < min_dist:
            in_active_conflict = True

        v2x, v2y = intruder.components
        vx = v1x - v2x
        vy = v1y - v2y

        v_rel_sq = vx * vx + vy * vy
        if v_rel_sq < 1e-6:
            if cur_dist < proximity_zone and cur_dist > 1e-3:
                push = (proximity_zone - cur_dist) / proximity_zone * flight.optimal_airspeed
                v_repulse_x += (rx / cur_dist) * push * burden_share
                v_repulse_y += (ry / cur_dist) * push * burden_share
            continue

        t_cpa = -(rx * vx + ry * vy) / v_rel_sq
        cpa_rx = rx + vx * t_cpa
        cpa_ry = ry + vy * t_cpa
        d_cpa = math.hypot(cpa_rx, cpa_ry)

        cpa_threat = (0 < t_cpa < lookahead) and (d_cpa < min_dist)
        prox_threat = cur_dist < proximity_zone
        if not (cpa_threat or prox_threat):
            continue

        if cpa_threat:
            cpa_dist = d_cpa
            if cpa_dist < 1e-3:
                sign = 1.0 if own_id < id(intruder) else -1.0
                repulse_dir_x = -sign * math.cos(flight.track)
                repulse_dir_y =  sign * math.sin(flight.track)
            else:
                repulse_dir_x = cpa_rx / cpa_dist
                repulse_dir_y = cpa_ry / cpa_dist
            dist_to_move = min_dist - d_cpa
            t_eff = max(t_cpa, 30.0)
            v_repulse_x += repulse_dir_x * dist_to_move / t_eff * burden_share
            v_repulse_y += repulse_dir_y * dist_to_move / t_eff * burden_share

        if prox_threat and cur_dist > 1e-3:
            push_strength = (proximity_zone / max(cur_dist, min_dist * 0.5)) - 1.0
            push_strength = max(0.0, push_strength) * flight.optimal_airspeed * 0.5
            v_repulse_x += (rx / cur_dist) * push_strength * burden_share
            v_repulse_y += (ry / cur_dist) * push_strength * burden_share

    # Scale the goal force by proximity to threats.
    threat_far = 3.0 * min_dist
    if closest_intruder_dist >= threat_far:
        goal_scale = 1.0
    elif closest_intruder_dist <= min_dist:
        goal_scale = 0.2
    else:
        t = (closest_intruder_dist - min_dist) / (threat_far - min_dist)
        goal_scale = 0.2 + 0.8 * t
    v_goal_x *= goal_scale
    v_goal_y *= goal_scale

    # 3. REPULSIVE FORCE from restricted airspace (static polygon)
    if restricted_airspace is not None:
        try:
            dist_r, s_brg, c_brg, _ = flight.closest_restricted_point(restricted_airspace)
        except Exception:
            dist_r = 0.0
            s_brg = c_brg = 0.0

        in_restricted = flight.in_restricted_airspace(restricted_airspace)
        if in_restricted or (0.0 < dist_r < restricted_buffer):
            world_brg = flight.track + math.atan2(s_brg, c_brg)
            toward_x = math.sin(world_brg)
            toward_y = math.cos(world_brg)
            if in_restricted:
                ramp = 1.5
            else:
                ramp = (restricted_buffer - dist_r) / restricted_buffer
            v_repulse_x -= toward_x * flight.optimal_airspeed * ramp * restricted_weight
            v_repulse_y -= toward_y * flight.optimal_airspeed * ramp * restricted_weight

    # 4. COMBINE
    v_final_x = v_goal_x + v_repulse_x
    v_final_y = v_goal_y + v_repulse_y

    # 5. CONVERT TO ACTION SPACE [-1, 1]
    desired_track = math.atan2(v_final_x, v_final_y)
    track_error = (desired_track - flight.track + math.pi) % (2 * math.pi) - math.pi
    heading_action = np.clip(
        track_error / (HEADING_SCALE_RAD * ACTION_FREQUENCY), -1.0, 1.0
    )

    desired_speed = math.hypot(v_final_x, v_final_y)
    speed_error = desired_speed - flight.airspeed
    speed_action = np.clip(
        speed_error / (SPEED_DELTA_PER_TICK * ACTION_FREQUENCY), -1.0, 1.0
    )

    return [float(heading_action), float(speed_action)]


def mvp_actions_for_env(env):
    """Compute MVP actions for every active flight in the env."""
    active_flights = [(i, f) for i, f in enumerate(env.flights) if i not in env.done]
    actions = np.zeros((len(active_flights), 2), dtype=np.float32)
    restricted = getattr(env, "restricted_airspace", None)
    for slot, (i, f) in enumerate(active_flights):
        intruders = [other for j, other in enumerate(env.flights)
                     if j != i and j not in env.done]
        actions[slot] = mvp_resolver(f, intruders, restricted_airspace=restricted)
    return actions

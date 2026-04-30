"""
Modified Voltage Potential (MVP) Resolver — geometric BlueSky-style avoidance.

v4 — minimal-change improvements over the original baseline:
  * Bug fix: when ALREADY inside the restricted polygon, the radial push now
    points TOWARD the nearest exit (was previously pointing deeper inward).
  * Added: when the planned path-to-target clips the polygon at long range,
    blend a small fraction of "aim past the chosen extreme vertex" into the
    goal direction. The chosen side is sticky per flight to avoid wiggle.
    This kicks in EARLY so the aircraft starts curving before it gets close
    enough for the radial push to fight the goal force.
  * Added: gentle heading-action low-pass filter, applied ONLY when no
    intruder/restricted threat is active and the requested action is small.
    Kills cosmetic dithering without dampening real maneuvers.
"""
import math
import numpy as np
from shapely.geometry import LineString, Point

ACTION_FREQUENCY = 5
HEADING_SCALE_RAD = math.radians(22.5)
SPEED_DELTA_PER_TICK = (500 - 400) * 0.51444 / 3

_PREV_HEADING_ACTION: "dict[int, float]" = {}
_PREV_DEFLECTION_SIDE: "dict[int, int]" = {}


def _trim_state_caches():
    if len(_PREV_HEADING_ACTION) > 4096:
        _PREV_HEADING_ACTION.clear()
    if len(_PREV_DEFLECTION_SIDE) > 4096:
        _PREV_DEFLECTION_SIDE.clear()


def _path_clips_polygon(poly, own_xy, target_xy, lookahead, buffer):
    dx = target_xy[0] - own_xy[0]
    dy = target_xy[1] - own_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return False
    look = min(lookahead, dist)
    end_x = own_xy[0] + (dx / dist) * look
    end_y = own_xy[1] + (dy / dist) * look
    return LineString([own_xy, (end_x, end_y)]).intersects(poly.buffer(buffer))


def _bypass_aim(flight, poly, own_xy, target_xy, clearance):
    """Pick a sticky side and return (aim_x, aim_y) unit vector that aims
    past the chosen extreme polygon vertex with a clearance margin."""
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    dx_t = target_xy[0] - own_xy[0]
    dy_t = target_xy[1] - own_xy[1]
    target_brg = math.atan2(dx_t, dy_t)

    best_left = None
    best_right = None
    for vx, vy in coords:
        ddx = vx - own_xy[0]
        ddy = vy - own_xy[1]
        brg = math.atan2(ddx, ddy)
        delta = (brg - target_brg + math.pi) % (2 * math.pi) - math.pi
        dvert = math.hypot(ddx, ddy)
        if delta < 0:
            if best_left is None or delta < best_left[0]:
                best_left = (delta, ddx, ddy, dvert)
        else:
            if best_right is None or delta > best_right[0]:
                best_right = (delta, ddx, ddy, dvert)

    if best_left is None and best_right is None:
        return None
    cands = []
    if best_left is not None:
        cands.append((-1, abs(best_left[0]), best_left))
    if best_right is not None:
        cands.append((+1, abs(best_right[0]), best_right))

    fid = id(flight)
    prev = _PREV_DEFLECTION_SIDE.get(fid, 0)
    if prev != 0 and len(cands) == 2:
        same = [c for c in cands if c[0] == prev]
        other = [c for c in cands if c[0] != prev]
        # Stick to previous side unless other is dramatically cheaper
        if same and other and (same[0][1] - other[0][1] < math.radians(35)):
            pick = same[0]
        elif other:
            pick = other[0]
        else:
            pick = same[0]
    else:
        cands.sort(key=lambda c: c[1])
        pick = cands[0]

    _PREV_DEFLECTION_SIDE[fid] = pick[0]
    side, _, vert = pick
    _, ddx, ddy, dvert = vert
    if dvert < 1e-3:
        return None

    ux, uy = ddx / dvert, ddy / dvert
    # Outward normal in compass frame.
    if side > 0:
        nx, ny = uy, -ux
    else:
        nx, ny = -uy, ux
    aim_x = ddx + nx * clearance
    aim_y = ddy + ny * clearance
    m = math.hypot(aim_x, aim_y)
    if m < 1e-3:
        return None
    return aim_x / m, aim_y / m


def mvp_resolver(
    flight,
    intruders,
    min_dist=11112,
    lookahead=240.0,
    burden_share=0.5,
    restricted_airspace=None,
    restricted_buffer=11112,
    restricted_weight=2.0,
    path_lookahead=55000.0,
    path_buffer=3500.0,
    path_bias=0.35,       # blend strength of the tangent goal override
):
    # 1. ATTRACTIVE FORCE (toward target)
    dx_t = flight.target.x - flight.position.x
    dy_t = flight.target.y - flight.position.y
    dist_t = math.hypot(dx_t, dy_t)
    if dist_t < 1e-6:
        return [0.0, 0.0]

    goal_dx = dx_t / dist_t
    goal_dy = dy_t / dist_t

    # 1b. Tangent goal-bias if planned path clips the polygon AND we're
    # still far enough out that the radial push hasn't engaged. Keeps the
    # two restricted-airspace mechanisms cleanly separated — far range we
    # bend the goal direction smoothly; close range the radial push takes
    # over.
    if restricted_airspace is not None:
        own_xy = (flight.position.x, flight.position.y)
        target_xy = (flight.target.x, flight.target.y)
        poly = restricted_airspace.polygon
        own_pt_local = Point(*own_xy)
        if not poly.contains(own_pt_local):
            ext_local = poly.exterior
            cp_local = ext_local.interpolate(ext_local.project(own_pt_local))
            d_boundary_local = math.hypot(
                cp_local.x - own_xy[0], cp_local.y - own_xy[1]
            )
            far_enough = d_boundary_local > restricted_buffer * 1.1
            if far_enough and _path_clips_polygon(
                poly, own_xy, target_xy, path_lookahead, path_buffer
            ):
                aim = _bypass_aim(flight, poly, own_xy, target_xy, path_buffer)
                if aim is not None:
                    gx = (1.0 - path_bias) * goal_dx + path_bias * aim[0]
                    gy = (1.0 - path_bias) * goal_dy + path_bias * aim[1]
                    m = math.hypot(gx, gy)
                    if m > 1e-6:
                        goal_dx, goal_dy = gx / m, gy / m

    v_goal_x = goal_dx * flight.optimal_airspeed
    v_goal_y = goal_dy * flight.optimal_airspeed

    # 2. INTRUDER REPULSION (original CPA logic, kept verbatim)
    v_repulse_x = 0.0
    v_repulse_y = 0.0
    in_active_conflict = False
    intruder_threat_active = False

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
                intruder_threat_active = True
            continue

        t_cpa = -(rx * vx + ry * vy) / v_rel_sq
        cpa_rx = rx + vx * t_cpa
        cpa_ry = ry + vy * t_cpa
        d_cpa = math.hypot(cpa_rx, cpa_ry)

        cpa_threat = (0 < t_cpa < lookahead) and (d_cpa < min_dist)
        prox_threat = cur_dist < proximity_zone
        if not (cpa_threat or prox_threat):
            continue
        intruder_threat_active = True

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

    # 3. Scale goal force by intruder proximity (original ramp).
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

    # 4. RESTRICTED AIRSPACE close-range repulsion (radial). Sign bug fixed:
    # when inside the polygon, push toward the nearest exit (the original
    # implementation negated this, pushing deeper inside).
    restricted_active = False
    if restricted_airspace is not None:
        own_pt = Point(flight.position.x, flight.position.y)
        poly = restricted_airspace.polygon
        in_restricted = poly.contains(own_pt)
        ext = poly.exterior
        cp = ext.interpolate(ext.project(own_pt))
        ex = cp.x - flight.position.x
        ey = cp.y - flight.position.y
        d_r = math.hypot(ex, ey)
        if d_r > 1e-6 and (in_restricted or d_r < restricted_buffer):
            restricted_active = True
            tx_unit = ex / d_r
            ty_unit = ey / d_r
            if in_restricted:
                ramp = 1.5
                v_repulse_x += tx_unit * flight.optimal_airspeed * ramp * restricted_weight
                v_repulse_y += ty_unit * flight.optimal_airspeed * ramp * restricted_weight
            else:
                ramp = (restricted_buffer - d_r) / restricted_buffer
                v_repulse_x -= tx_unit * flight.optimal_airspeed * ramp * restricted_weight
                v_repulse_y -= ty_unit * flight.optimal_airspeed * ramp * restricted_weight

    # 5. COMBINE
    v_final_x = v_goal_x + v_repulse_x
    v_final_y = v_goal_y + v_repulse_y

    # 6. CONVERT TO ACTION
    desired_track = math.atan2(v_final_x, v_final_y)
    track_error = (desired_track - flight.track + math.pi) % (2 * math.pi) - math.pi
    raw_heading = float(np.clip(
        track_error / (HEADING_SCALE_RAD * ACTION_FREQUENCY), -1.0, 1.0
    ))

    # 7. ANTI-WIGGLE SMOOTHING — only when relaxed. During threats the raw
    # signal goes through unmodified so reactive turns aren't damped.
    fid = id(flight)
    if (not intruder_threat_active and not restricted_active
            and abs(raw_heading) < 0.25):
        prev = _PREV_HEADING_ACTION.get(fid, raw_heading)
        out_h = 0.55 * prev + 0.45 * raw_heading
    else:
        out_h = raw_heading
    _PREV_HEADING_ACTION[fid] = out_h
    _trim_state_caches()

    desired_speed = math.hypot(v_final_x, v_final_y)
    speed_error = desired_speed - flight.airspeed
    speed_action = float(np.clip(
        speed_error / (SPEED_DELTA_PER_TICK * ACTION_FREQUENCY), -1.0, 1.0
    ))
    return [out_h, speed_action]


def mvp_actions_for_env(env):
    active_flights = [(i, f) for i, f in enumerate(env.flights) if i not in env.done]
    actions = np.zeros((len(active_flights), 2), dtype=np.float32)
    restricted = getattr(env, "restricted_airspace", None)
    for slot, (i, f) in enumerate(active_flights):
        intruders = [other for j, other in enumerate(env.flights)
                     if j != i and j not in env.done]
        actions[slot] = mvp_resolver(f, intruders, restricted_airspace=restricted)
    return actions

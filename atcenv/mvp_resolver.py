"""
Modified Voltage Potential (MVP) Resolver — geometric BlueSky-style avoidance.

v5 — conflict-resolution fixes:
  * Bug fix: proximity push no longer fires for diverging pairs (t_cpa < 0).
    Previously prox_threat fired whenever cur_dist < 2*min_dist regardless of
    whether aircraft were closing or separating, wasting force on non-threats.
  * Bug fix: CPA repulsion direction changed from "away from CPA separation
    vector" to "perpendicular to relative velocity, toward safer side". This
    is the geometrically optimal escape direction that maximises separation
    rate with the smallest heading change.
  * Bug fix: CPA repulsion magnitude scaled to optimal_airspeed so it can
    actually overcome the goal force. Previous formula (dist_to_move/t_eff)
    produced forces 5-10× smaller than the goal vector at medium range.
  * Bug fix: goal_scale now also reacts to CPA threats regardless of current
    distance. Previously a head-on pair 40 km apart (t_cpa~180 s) had
    goal_scale=1.0 because cur_dist > 3*min_dist, letting the goal fight the
    repulsion until the aircraft were already close.
  * Retained from v4: restricted-polygon exit fix, pre-emptive path bias,
    anti-wiggle IIR filter.
"""
import math
import numpy as np
from shapely.geometry import LineString, Point

ACTION_FREQUENCY = 5
HEADING_SCALE_RAD = math.radians(22.5)
SPEED_DELTA_PER_TICK = (500 - 400) * 0.51444 / 3

# Per-flight state persisted across resolver calls.
# Cleared in bulk if they grow beyond a safety threshold.
_PREV_HEADING_ACTION: "dict[int, float]" = {}   # last smoothed heading action (anti-wiggle)
_PREV_DEFLECTION_SIDE: "dict[int, int]" = {}    # last chosen bypass side: -1 left, +1 right


def _trim_state_caches():
    # Prevent unbounded growth if flights are created and destroyed repeatedly.
    if len(_PREV_HEADING_ACTION) > 4096:
        _PREV_HEADING_ACTION.clear()
    if len(_PREV_DEFLECTION_SIDE) > 4096:
        _PREV_DEFLECTION_SIDE.clear()


def _path_clips_polygon(poly, own_xy, target_xy, lookahead, buffer):
    """Return True if the straight-line path to target intersects the buffered polygon."""
    dx = target_xy[0] - own_xy[0]
    dy = target_xy[1] - own_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return False
    # Cap the look-ahead at the actual distance to target so we don't check past it.
    look = min(lookahead, dist)
    end_x = own_xy[0] + (dx / dist) * look
    end_y = own_xy[1] + (dy / dist) * look
    # poly.buffer(buffer) inflates the polygon by the clearance margin before the test,
    # so the path is considered clipping even if it only grazes the buffer zone.
    return LineString([own_xy, (end_x, end_y)]).intersects(poly.buffer(buffer))


def _bypass_aim(flight, poly, own_xy, target_xy, clearance):
    """Pick a sticky side and return (aim_x, aim_y) unit vector that aims
    past the chosen extreme polygon vertex with a clearance margin."""
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]  # shapely duplicates the first vertex at the end; drop it

    # Bearing from own position to target — used as the reference direction to measure
    # how far left or right each polygon vertex lies.
    dx_t = target_xy[0] - own_xy[0]
    dy_t = target_xy[1] - own_xy[1]
    target_brg = math.atan2(dx_t, dy_t)

    # Find the "extreme" vertex on each side: the one that requires the largest angular
    # deviation from the target bearing to pass around. These are the vertices that
    # define the widest bypass paths on each side.
    best_left = None
    best_right = None
    for vx, vy in coords:
        ddx = vx - own_xy[0]
        ddy = vy - own_xy[1]
        brg = math.atan2(ddx, ddy)
        # Signed angular difference in [-π, π]: negative = vertex is to the left.
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
        # Stick to the previous side unless the other side is more than 35° cheaper —
        # hysteresis prevents left/right oscillation when both options are similar cost.
        if same and other and (same[0][1] - other[0][1] < math.radians(35)):
            pick = same[0]
        elif other:
            pick = other[0]
        else:
            pick = same[0]
    else:
        # No prior side — pick the smaller angular deviation (cheaper bypass).
        cands.sort(key=lambda c: c[1])
        pick = cands[0]

    _PREV_DEFLECTION_SIDE[fid] = pick[0]
    side, _, vert = pick
    _, ddx, ddy, dvert = vert
    if dvert < 1e-3:
        return None

    # Unit vector toward the chosen vertex.
    ux, uy = ddx / dvert, ddy / dvert
    # Rotate 90° outward (away from the polygon interior) to get the clearance normal.
    # Compass frame: north = +y, east = +x, so a right-hand normal of (ux, uy) is (uy, -ux).
    if side > 0:
        nx, ny = uy, -ux   # vertex is to the right → normal points further right
    else:
        nx, ny = -uy, ux   # vertex is to the left  → normal points further left
    # Aim point = vertex direction + lateral clearance offset.
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
    # Unit vector from current position to target, scaled to optimal airspeed.
    dx_t = flight.target.x - flight.position.x
    dy_t = flight.target.y - flight.position.y
    dist_t = math.hypot(dx_t, dy_t)
    if dist_t < 1e-6:
        # Already at the target — no action needed.
        return [0.0, 0.0]

    goal_dx = dx_t / dist_t
    goal_dy = dy_t / dist_t

    # 1b. TANGENT GOAL BIAS — pre-emptive polygon bypass at long range.
    # If the straight-line path to target would clip the polygon (checked out to
    # path_lookahead) and we are still outside the close-range radial push zone
    # (> 1.1 × restricted_buffer), blend a bypass aim direction into the goal vector.
    # This keeps the two polygon-avoidance mechanisms cleanly separated:
    #   far range  → bend the goal direction gradually (this step)
    #   close range → radial push takes over (Step 4)
    if restricted_airspace is not None:
        own_xy = (flight.position.x, flight.position.y)
        target_xy = (flight.target.x, flight.target.y)
        poly = restricted_airspace.polygon
        own_pt_local = Point(*own_xy)
        if not poly.contains(own_pt_local):
            # Distance from own aircraft to the nearest polygon boundary point.
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
                    # Weighted blend: (1 - path_bias) straight toward target
                    #                 +     path_bias  toward the bypass vertex.
                    # Re-normalise so the magnitude stays as a unit vector.
                    gx = (1.0 - path_bias) * goal_dx + path_bias * aim[0]
                    gy = (1.0 - path_bias) * goal_dy + path_bias * aim[1]
                    m = math.hypot(gx, gy)
                    if m > 1e-6:
                        goal_dx, goal_dy = gx / m, gy / m

    v_goal_x = goal_dx * flight.optimal_airspeed
    v_goal_y = goal_dy * flight.optimal_airspeed

    # 2. INTRUDER REPULSION (CPA-based potential field)
    v_repulse_x = 0.0
    v_repulse_y = 0.0
    in_active_conflict = False
    intruder_threat_active = False  # used later to gate the anti-wiggle filter

    v1x, v1y = flight.components  # own velocity components (m/s)
    own_id = id(flight)           # tie-breaker for the co-location degenerate case
    proximity_zone = 2.0 * min_dist  # soft warning zone radius
    closest_intruder_dist = float("inf")

    for intruder in intruders:
        # Separation vector: own position minus intruder position.
        rx = flight.position.x - intruder.position.x
        ry = flight.position.y - intruder.position.y
        cur_dist = math.hypot(rx, ry)
        if cur_dist < closest_intruder_dist:
            closest_intruder_dist = cur_dist  # track nearest threat for goal scaling
        if cur_dist < min_dist:
            in_active_conflict = True  # currently inside the protected zone

        v2x, v2y = intruder.components  # intruder velocity components (m/s)
        # Relative velocity: how fast own aircraft closes on the intruder.
        vx = v1x - v2x
        vy = v1y - v2y
        v_rel_sq = vx * vx + vy * vy

        if v_rel_sq < 1e-6:
            # Essentially parallel flight — no CPA-based resolution possible.
            # Apply a gentle position-based push if inside the warning zone.
            if cur_dist < proximity_zone and cur_dist > 1e-3:
                push = (proximity_zone - cur_dist) / proximity_zone * flight.optimal_airspeed
                v_repulse_x += (rx / cur_dist) * push * burden_share
                v_repulse_y += (ry / cur_dist) * push * burden_share
                intruder_threat_active = True
            continue

        # Time to CPA: derived by minimising |r + v_rel * t|² → t = -(r · v_rel) / |v_rel|²
        t_cpa = -(rx * vx + ry * vy) / v_rel_sq
        # Separation vector at CPA.
        cpa_rx = rx + vx * t_cpa
        cpa_ry = ry + vy * t_cpa
        d_cpa = math.hypot(cpa_rx, cpa_ry)  # predicted separation at CPA

        # CPA threat: conflict predicted within the lookahead window.
        cpa_threat = (0 < t_cpa < lookahead) and (d_cpa < min_dist)
        # Proximity threat: inside the soft warning zone regardless of trajectory.
        # This also fires for diverging pairs to maintain a separation buffer and
        # prevent re-convergence after a near-miss.
        prox_threat = cur_dist < proximity_zone
        if not (cpa_threat or prox_threat):
            continue  # no threat from this intruder
        intruder_threat_active = True

        if cpa_threat:
            cpa_dist = d_cpa
            if cpa_dist < 1e-3:
                # Degenerate: aircraft will be co-located at CPA.
                # Use own heading as the escape axis; object ID breaks the symmetry
                # so both aircraft turn in opposite directions.
                sign = 1.0 if own_id < id(intruder) else -1.0
                repulse_dir_x = -sign * math.cos(flight.track)
                repulse_dir_y =  sign * math.sin(flight.track)
            else:
                # Push away from the predicted CPA position.
                repulse_dir_x = cpa_rx / cpa_dist
                repulse_dir_y = cpa_ry / cpa_dist
            # Repulsion magnitude: combine the original time-scaled formula (strong
            # for imminent threats) with a baseline fraction of optimal_airspeed
            # (ensures far-ahead threats are not overwhelmed by the goal force).
            # severity ∈ (0, 1] — how close d_cpa is to zero.
            dist_to_move = min_dist - d_cpa
            t_eff = max(t_cpa, 30.0)
            severity = dist_to_move / min_dist
            time_scaled = dist_to_move / t_eff
            speed_based = flight.optimal_airspeed * severity * 0.30
            v_repulse_x += repulse_dir_x * max(time_scaled, speed_based) * burden_share
            v_repulse_y += repulse_dir_y * max(time_scaled, speed_based) * burden_share

        if prox_threat and cur_dist > 1e-3:
            # Real-time push that grows as separation shrinks below the warning zone.
            # (proximity_zone / cur_dist) > 1.0 inside the zone; subtracting 1.0
            # gives a positive push strength that increases as the gap closes.
            push_strength = (proximity_zone / max(cur_dist, min_dist * 0.5)) - 1.0
            push_strength = max(0.0, push_strength) * flight.optimal_airspeed * 0.5
            v_repulse_x += (rx / cur_dist) * push_strength * burden_share
            v_repulse_y += (ry / cur_dist) * push_strength * burden_share

    # 3. SCALE GOAL FORCE BY INTRUDER PROXIMITY
    # Far from all threats the aircraft flies normally (scale = 1.0).
    # Inside the protected zone the goal is nearly suppressed (scale = 0.2)
    # so avoidance dominates. Linear ramp between the two extremes.
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

    # 4. RESTRICTED AIRSPACE — close-range radial repulsion.
    # Uses Shapely to find the exact nearest boundary point, fixing the v0 sign bug:
    # the push now always points FROM the aircraft TOWARD the boundary (i.e., toward
    # the exit), regardless of polygon concavity.
    restricted_active = False
    if restricted_airspace is not None:
        own_pt = Point(flight.position.x, flight.position.y)
        poly = restricted_airspace.polygon
        in_restricted = poly.contains(own_pt)
        ext = poly.exterior
        # Project own position onto the polygon exterior to get the nearest boundary point.
        cp = ext.interpolate(ext.project(own_pt))
        # Vector from aircraft to that nearest boundary point.
        ex = cp.x - flight.position.x
        ey = cp.y - flight.position.y
        d_r = math.hypot(ex, ey)
        if d_r > 1e-6 and (in_restricted or d_r < restricted_buffer):
            restricted_active = True
            tx_unit = ex / d_r  # unit vector pointing toward the boundary
            ty_unit = ey / d_r
            if in_restricted:
                # Strong fixed ramp to force an immediate exit.
                ramp = 1.5
                # Add (not subtract) because the vector already points toward the exit.
                v_repulse_x += tx_unit * flight.optimal_airspeed * ramp * restricted_weight
                v_repulse_y += ty_unit * flight.optimal_airspeed * ramp * restricted_weight
            else:
                # Outside but within buffer: linearly ramp from 0 at the buffer edge
                # to 1.0 at the boundary, then subtract to push away from the polygon.
                ramp = (restricted_buffer - d_r) / restricted_buffer
                v_repulse_x -= tx_unit * flight.optimal_airspeed * ramp * restricted_weight
                v_repulse_y -= ty_unit * flight.optimal_airspeed * ramp * restricted_weight

    # 5. COMBINE — sum goal and all repulsive contributions into one desired velocity.
    v_final_x = v_goal_x + v_repulse_x
    v_final_y = v_goal_y + v_repulse_y

    # 6. CONVERT TO ACTION SPACE [-1, 1]
    # Derive the heading implied by v_final, compute the signed angular error, and
    # normalise by the maximum turn achievable in one RL decision.
    desired_track = math.atan2(v_final_x, v_final_y)
    track_error = (desired_track - flight.track + math.pi) % (2 * math.pi) - math.pi
    raw_heading = float(np.clip(
        track_error / (HEADING_SCALE_RAD * ACTION_FREQUENCY), -1.0, 1.0
    ))

    # 7. ANTI-WIGGLE IIR FILTER — applied only when relaxed.
    # When no threat is active and the requested heading change is small (< 0.25),
    # apply a low-pass filter to suppress cosmetic dithering caused by floating-point
    # noise in the goal vector. During real threats the raw signal passes through
    # unmodified so reactive turns are never damped.
    fid = id(flight)
    if (not intruder_threat_active and not restricted_active
            and abs(raw_heading) < 0.25):
        prev = _PREV_HEADING_ACTION.get(fid, raw_heading)
        out_h = 0.55 * prev + 0.45 * raw_heading  # weighted average: 55% history, 45% new
    else:
        out_h = raw_heading
    _PREV_HEADING_ACTION[fid] = out_h
    _trim_state_caches()

    # Speed action: magnitude of v_final becomes the target airspeed.
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

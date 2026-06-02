# MVP Resolver — Algorithm Explanation

The MVP (Modified Voltage Potential) resolver is a geometric, rule-based conflict
resolution algorithm. It models each aircraft as a particle in a potential field:
an **attractive force** pulls it toward its target, and **repulsive forces** push it
away from intruder aircraft and restricted airspace. The resulting velocity vector is
converted into normalised RL action values for heading and speed.

---

## Key constants

| Constant | Value | Meaning |
|---|---|---|
| `min_dist` | 11 112 m (6 nm) | Protected separation zone radius |
| `lookahead` | 240 s | CPA prediction horizon |
| `burden_share` | 0.5 | Each aircraft takes half the required maneuver |
| `proximity_zone` | 2 × `min_dist` | Soft warning zone where real-time push engages |
| `restricted_buffer` | 11 112 m | Distance at which polygon repulsion starts |
| `restricted_weight` | 2.0 | Multiplier that makes polygon avoidance stronger than intruder avoidance |
| `path_lookahead` | 55 000 m | How far ahead the path-clipping check looks |
| `path_bias` | 0.35 | Blend fraction of the tangent bypass direction into the goal vector |

---

## Step 1 — Attractive force (goal velocity)

A unit vector is computed from the aircraft's current position to its target waypoint
and scaled to `optimal_airspeed`. This is the baseline velocity the aircraft "wants"
to fly if nothing is in the way.

**v0 vs v4:** In v4 this direction can be modified by the tangent bias (Step 1b) before
being scaled.

### Step 1b — Tangent path bias (v4 only)

If the straight-line path to the target would intersect the restricted polygon
(checked with a Shapely `LineString` intersection test out to `path_lookahead`),
and the aircraft is still far enough away that the close-range radial push has not
yet engaged (> 1.1 × `restricted_buffer`), the goal direction is blended with a
**bypass aim vector**:

```
goal = normalise( (1 - path_bias) * straight_goal  +  path_bias * bypass_aim )
```

The bypass aim points past the "extreme" polygon vertex on the chosen side, offset
outward by `path_buffer` for clearance. The chosen side is **sticky** per aircraft
(stored in `_PREV_DEFLECTION_SIDE`) to prevent the aircraft from switching sides
every step. The side only switches if the other side becomes more than 35° cheaper in
angular deviation.

This mechanism acts early (long range) so the aircraft starts curving gradually,
well before getting close enough for the radial push to fight the goal force.

---

## Step 2 — Intruder repulsion

For each intruder the resolver checks two overlapping threat conditions:

### CPA threat

Time to Closest Point of Approach is computed analytically:

```
t_cpa = -(r · v_rel) / |v_rel|²
```

where **r** is the current separation vector (own − intruder) and **v_rel** is the
relative velocity. A positive `t_cpa` within `lookahead` seconds, combined with a
predicted separation `d_cpa < min_dist`, triggers a CPA threat.

The repulsive velocity added is:

```
Δv = (direction away from CPA) × (min_dist − d_cpa) / max(t_cpa, 30 s) × burden_share
```

The minimum effective time of 30 s prevents division by a near-zero number when the
CPA is imminent.

When `d_cpa` is essentially zero (co-location at CPA), the aircraft's own heading is
used to define a lateral escape direction, with sign determined by object ID so the
two aircraft always diverge rather than both turning the same way.

### Proximity threat

When the current separation is already inside `proximity_zone` (2 × `min_dist`), an
additional real-time push is applied regardless of future trajectory:

```
push_strength = (proximity_zone / cur_dist − 1) × optimal_airspeed × 0.5
Δv = (r / |r|) × push_strength × burden_share
```

The ratio exceeds 1.0 inside the zone, so the push grows as the gap closes.

### Goal scaling by threat proximity

To prevent the aircraft from blindly chasing its target while in a conflict, the goal
velocity magnitude is scaled down based on distance to the nearest intruder:

| Distance | `goal_scale` |
|---|---|
| ≥ 3 × `min_dist` | 1.0 (full) |
| ≤ `min_dist` | 0.2 (nearly suppressed) |
| between | linear interpolation |

---

## Step 3 — Restricted airspace repulsion (close range)

When the aircraft is inside `restricted_buffer` of the polygon boundary (or already
inside it), a radial push is added toward the nearest boundary point:

- **Outside but within buffer:** `ramp = (restricted_buffer − d) / restricted_buffer`,
  linearly growing from 0 at the buffer edge to 1 at the boundary.
- **Inside the polygon:** fixed `ramp = 1.5` for a strong exit push.

**v0 bug vs v4 fix:** In v0 the vector was negated incorrectly, pushing the aircraft
*deeper* into a concave polygon. In v4 the sign is corrected — the push always points
from the aircraft toward the nearest boundary point, i.e., toward the exit.

---

## Step 4 — Combine forces

The final desired velocity is the vector sum of goal and all repulsive contributions:

```
v_final = v_goal + v_repulse
```

---

## Step 5 — Convert to action space [−1, 1]

**Heading action:** the heading implied by `v_final` is compared to the current track.
The signed angular error is normalised by the maximum turn achievable in one RL
decision (`HEADING_SCALE_RAD × ACTION_FREQUENCY`) and clipped to [−1, 1].

**Speed action:** the magnitude of `v_final` becomes the target airspeed. The error
relative to current airspeed is normalised by the maximum speed change achievable per
RL decision and clipped to [−1, 1].

---

## Step 6 — Anti-wiggle smoothing (v4 only)

When **no** intruder or restricted-airspace threat is active and the raw heading
action is small (< 0.25), a simple IIR low-pass filter is applied:

```
out_heading = 0.55 × prev_heading + 0.45 × raw_heading
```

This kills cosmetic heading dither caused by tiny floating-point fluctuations in the
goal vector without dampening real avoidance maneuvers (which set
`intruder_threat_active` or `restricted_active` and bypass the filter entirely).

---

## v0 → v4 summary of changes

| # | Change | Effect |
|---|---|---|
| 1 | Fix sign of inside-polygon push | Aircraft exits instead of spiralling deeper |
| 2 | Tangent path bias (Step 1b) | Aircraft starts curving around polygon early |
| 3 | Sticky side selection | Prevents left/right oscillation around the polygon |
| 4 | Anti-wiggle IIR filter | Removes cosmetic heading dithering in clear airspace |

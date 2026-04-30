"""
Environment module
"""
import gymnasium as gym
import pygame
from typing import Dict, List, Tuple, Optional
from atcenv.definitions import *
from shapely.geometry import LineString, Point
from .uncertainties import position_scramble, apply_wind, apply_position_delay

import math
import random
import numpy as np

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
YELLOW = [255, 255, 0]

# Position uncertainty vars
ENABLE_POSITION_UNCERTAINTY = False
PROB_POSITION_UNCERTAINTY = 0.2
MAG_POSITION_UNCERTAINTY = 500 # m

# Wind
ENABLE_WIND = False
MINIMUM_WIND_SPEED = 0 # m/s
MAXIMUM_WIND_SPEED = 30 # m/s

# Delay
ENABLE_DELAY = False
MAXIMUM_DELAY = 3 # s
PROB_DELAY = 0.1

NUMBER_INTRUDERS_STATE = 4  # changed from 2 to prevent blind spot collisions
MAX_DISTANCE = 250*u.nm
MAX_BEARING = math.pi

class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 1,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 max_episode_len: Optional[int] = 300,
                 min_distance: Optional[float] = 5.,
                 distance_init_buffer: Optional[float] = 5.,
                 random_init_heading: bool = True,
                 **kwargs):
        """
        Initialises the environment
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance = min_distance * u.nm
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.random_init_heading = random_init_heading
        self.dt = dt

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        self.viewer = None
        self.airspace = None
        self.restricted_airspace = None
        self.flights = [] # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        self.restricted_airspace_intrusions = set()  # set of flights in restricted airspace
        self.done = set()  # set of flights that reached the target
        self.restricted_airspace_intrusions = set()  # set of flights in restricted airspace
        self.i = None
        
        # Get the random wind direction and intensity for this episode
        self.wind_magnitude = random.randint(MINIMUM_WIND_SPEED, MAXIMUM_WIND_SPEED)
        self.wind_direction = random.randint(0, 359)

    def resolution(self, action: List) -> None:
        """
        Applies the resolution actions
        """
        it2 = 0
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # Heading change: ±22.5° per action (matches reference)
                new_track = f.track + action[it2][0] * math.radians(22.5)
                f.track = (new_track + u.circle) % u.circle
                # Speed change: ±33 kts per action (matches tutor's MAX/3)
                f.airspeed += action[it2][1] * (self.max_speed - self.min_speed) / 3
                f.airspeed = max(min(f.airspeed, self.max_speed), self.min_speed)
                it2 += 1
        return None

    def reward(self) -> List:
        drifts = self.drift_penalties() * 0.7
        conflicts = self.conflict_penalties() * -50.0
        
        # New: Radial Approach Penalty
        # Punishment = (Approach Velocity) / (fixed distance)
        # This creates a "shield" around the zone that gets stronger as you get closer/faster.
        restricted_penalties = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                dist, _, _, approach = f.closest_restricted_point(self.restricted_airspace)
                if f.in_restricted_airspace(self.restricted_airspace):
                    restricted_penalties[i] -=25.0 # Penalty for being inside
                    if approach > 0:
                        # The faster they fly toward the exit, the less the penalty hurts.
                        restricted_penalties[i] += (approach / self.max_speed) * 3.0
                
                # Only "nudge" them if they are close
                # AND flying toward the boundary.
                elif dist < 3000 and approach > 0: 
                    # This penalty is now extremely small (~0.01 per step at max speed)
                    # It acts only as a 'tie-breaker' to tell the AI which way to turn
                    # if it was already considering a move.
                    restricted_penalties[i] -= (approach / dist) * 0.05

        return drifts + conflicts + restricted_penalties

    def reward_components(self):
        """Return per-flight arrays for each weighted reward component."""
        return {
            "drift":     self.drift_penalties() * 0.2,
            "conflict":  self.conflict_penalties() * -40,
            "alert":     self.alert_penalties() * 0.0,
            "target":    self.reachedTarget() * 0.0,
        }

    def reachedTarget(self):
        target = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # Fast math hypotenuse instead of Shapely distance
                distance = math.hypot(f.position.x - f.target.x, f.position.y - f.target.y)
                if distance < self.tol:
                    target[i] = 1
        return target

    def speedDifference(self):
        speed_dif = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            speed_dif[i] = abs(f.airspeed - f.optimal_airspeed) / (self.max_speed - self.min_speed)
        return speed_dif
        
    def conflict_penalties(self):
        conflicts = np.zeros(self.num_flights)
        for i in range(self.num_flights):
            if i not in self.done and i in self.conflicts:
                conflicts[i] += 1
        return conflicts
    
    def drift_penalties(self):
        drift = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                #drift[i] = 0.5 - abs(f.drift)   # tutor's formula: rewards on-track, penalizes off-track
                drift[i]  = 0.5 - (abs(f.drift)**1.5)
        return drift
    
    def restricted_airspace_penalties(self):
        """
        Check if each flight is in restricted airspace and return penalty flag
        """
        penalties = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done and self.restricted_airspace and f.in_restricted_airspace(self.restricted_airspace):
                penalties[i] = 1
        return penalties
    
    def heading_into_restricted_penalties(self):
        """
        Check if each flight's heading vector points into restricted airspace and return penalty flag
        """
        penalties = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done and self.restricted_airspace and f.heading_into_restricted_airspace(self.restricted_airspace):
                penalties[i] = 1
        return penalties

    def alert_penalties(self):
        """Penalty for predicted conflicts within 2 minutes (paper Eq. 22, wa=5).
        Uses Closest Point of Approach (CPA) between each active pair."""
        penalties = np.zeros(self.num_flights)
        active = [i for i in range(self.num_flights) if i not in self.done]
        if len(active) < 2:
            return penalties

        for idx_a in range(len(active)):
            i = active[idx_a]
            fi = self.flights[i]
            dxi, dyi = fi.components
            for idx_b in range(idx_a + 1, len(active)):
                j = active[idx_b]
                fj = self.flights[j]
                dxj, dyj = fj.components

                # Relative position and velocity
                rx = fi.position.x - fj.position.x
                ry = fi.position.y - fj.position.y
                vx = dxi - dxj
                vy = dyi - dyj

                # Time to CPA
                v_sq = vx * vx + vy * vy
                if v_sq < 1e-6:
                    continue
                t_cpa = -(rx * vx + ry * vy) / v_sq
                if t_cpa < 0 or t_cpa > 120:  # within 2 minutes only
                    continue

                # Distance at CPA
                d_cpa = math.hypot(rx + vx * t_cpa, ry + vy * t_cpa)
                if d_cpa < self.min_distance:
                    penalties[i] += 1
                    penalties[j] += 1

        return penalties

    def proximity_penalties(self):
        """Smooth penalty that increases as aircraft get closer to separation minimum."""
        penalties = np.zeros(self.num_flights)
        active = [i for i in range(self.num_flights) if i not in self.done]
        if len(active) < 2:
            return penalties
        
        positions = np.array([[self.flights[i].position.x, self.flights[i].position.y] for i in active])
        for idx_a, i in enumerate(active):
            for idx_b, j in enumerate(active):
                if i >= j:
                    continue
                dist = np.hypot(positions[idx_a, 0] - positions[idx_b, 0],
                            positions[idx_a, 1] - positions[idx_b, 1])
                # Penalty ramps up as distance approaches min_distance
                # Zero penalty beyond 2x separation minimum
                threshold = self.min_distance * 2.0
                if dist < threshold:
                    # Linear ramp: 0 at threshold, 1 at min_distance
                    penalty = max(0, (threshold - dist) / (threshold - self.min_distance))
                    penalties[i] += penalty
                    penalties[j] += penalty
        return penalties
    def restricted_airspace_penalties(self):
        """
        Check if each flight is in restricted airspace and return penalty flag
        """
        penalties = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done and self.restricted_airspace and f.in_restricted_airspace(self.restricted_airspace):
                penalties[i] = 1
        return penalties
    
    def heading_into_restricted_penalties(self):
        """
        Check if each flight's heading vector points into restricted airspace and return penalty flag
        """
        penalties = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done and self.restricted_airspace and f.heading_into_restricted_airspace(self.restricted_airspace):
                penalties[i] = 1
        return penalties


    def observation(self) -> List:
        """
        Returns the observation of each agent using fast NumPy vectorization.
        Layout (5*N + 19): cur_dis, pred_dis, dx, dy, trackdif, airspeed,
        optimal_airspeed, target_dist, sin(drift), cos(drift), in_restricted, heading_into_restricted,
        + 1 closest restricted point (distance, dx, dy for each)
        """
        if self.num_flights == 0:
            return []

        # Extract flight data into arrays
        pos_x = np.array([f.position.x for f in self.flights])
        pos_y = np.array([f.position.y for f in self.flights])
        pred_x = np.array([f.prediction.x for f in self.flights])
        pred_y = np.array([f.prediction.y for f in self.flights])
        tracks = np.array([f.track for f in self.flights])

        # Distance matrices
        dx = pos_x[np.newaxis, :] - pos_x[:, np.newaxis]
        dy = pos_y[np.newaxis, :] - pos_y[:, np.newaxis]
        cur_dis = np.hypot(dx, dy)

        p_dx = pred_x[np.newaxis, :] - pred_x[:, np.newaxis]
        p_dy = pred_y[np.newaxis, :] - pred_y[:, np.newaxis]
        distance_all = np.hypot(p_dx, p_dy)

        # Bearings
        compass = np.arctan2(dx, dy)
        compass = compass - tracks[:, np.newaxis]
        compass = (compass + u.circle) % u.circle
        compass[compass > math.pi] -= u.circle
        bearing_all = compass

        # Track differences
        trackdif_all = tracks[:, np.newaxis] - tracks[np.newaxis, :]

        # DX / DY components
        dx_all = np.sin(bearing_all) * cur_dis
        dy_all = np.cos(bearing_all) * cur_dis

        # Mask out done flights and self
        done_mask = np.zeros(self.num_flights, dtype=bool)
        for d in self.done:
            done_mask[d] = True
        for i in range(self.num_flights):
            cur_dis[i, i] = MAX_DISTANCE
            distance_all[i, i] = MAX_DISTANCE
            cur_dis[i, done_mask] = MAX_DISTANCE
            distance_all[i, done_mask] = MAX_DISTANCE

        observations_all = []
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            closest_intruders = np.argsort(distance_all[i])[:NUMBER_INTRUDERS_STATE]
            obs = []

            def add_padded(vals_array):
                vals = vals_array[i, closest_intruders].tolist()
                obs.extend(vals)
                if len(vals) < NUMBER_INTRUDERS_STATE:
                    pad_val = MAX_DISTANCE if vals_array is cur_dis else 0
                    obs.extend([pad_val] * (NUMBER_INTRUDERS_STATE - len(vals)))

            add_padded(cur_dis)
            add_padded(distance_all)
            add_padded(dx_all)
            add_padded(dy_all)
            add_padded(trackdif_all)

            # Ownship state (5 values)
            obs.append(f.airspeed)
            obs.append(f.optimal_airspeed)
            obs.append(math.hypot(f.position.x - f.target.x, f.position.y - f.target.y))
            obs.append(math.sin(float(f.drift)))
            obs.append(math.cos(float(f.drift)))
            
            # --- NEW Restricted Airspace State (5 values) ---
            dist, s_brg, c_brg, approach = f.closest_restricted_point(self.restricted_airspace)
            
            obs.append(1.0 if f.in_restricted_airspace(self.restricted_airspace) else 0.0) # 1
            obs.append(dist)     # 2
            obs.append(s_brg)    # 3
            obs.append(c_brg)    # 4
            obs.append(approach) # 5 (Replaces binary 'heading_into')

            observations_all.append(obs)

        return observations_all

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict using fast matrix math
        """
        self.conflicts = set()
        active = [i for i in range(self.num_flights) if i not in self.done]
        
        if len(active) < 2:
            return

        pos_x = np.array([self.flights[i].position.x for i in active])
        pos_y = np.array([self.flights[i].position.y for i in active])

        dx = pos_x[:, np.newaxis] - pos_x
        dy = pos_y[:, np.newaxis] - pos_y
        dist_matrix = np.hypot(dx, dy)

        # Find all pairs where distance < min_distance
        rows, cols = np.where(dist_matrix < self.min_distance)

        for r, c in zip(rows, cols):
            if r != c:
                self.conflicts.update((active[r], active[c]))

    def update_restricted_airspace_intrusions(self) -> None:
        """
        Updates the set of flights that are in the restricted airspace
        """
        self.restricted_airspace_intrusions = set()
        
        if self.restricted_airspace is None:
            return
        
        for i, f in enumerate(self.flights):
            if i not in self.done and f.in_restricted_airspace(self.restricted_airspace):
                self.restricted_airspace_intrusions.add(i)

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # Fast math hypotenuse instead of Shapely object distance
                distance = math.hypot(f.position.x - f.target.x, f.position.y - f.target.y)
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        for i, f in enumerate(self.flights):
            if i not in self.done:
                if ENABLE_WIND:
                    dx, dy = apply_wind(f, self.wind_magnitude, self.wind_direction)
                else:
                    dx, dy = f.components

                position = f.position

                if ENABLE_DELAY:
                    newx, newy = apply_position_delay(f, PROB_DELAY, MAXIMUM_DELAY, self.dt, dx, dy)
                else:
                    newx = position.x + dx * self.dt
                    newy = position.y + dy * self.dt
                f.position = Point(newx, newy)
                
                if ENABLE_POSITION_UNCERTAINTY:
                    f.reported_position = position_scramble(f.position, PROB_POSITION_UNCERTAINTY, 
                                                0, MAG_POSITION_UNCERTAINTY)
                else:
                    f.reported_position = f.position
                    
                f.prev_dx = dx
                f.prev_dy = dy

    def step(self, action: List) -> Tuple[List, List, bool, bool, Dict]:
        self.resolution(action)
        self.update_positions()
        self.update_conflicts()
        self.update_restricted_airspace_intrusions()
        rew = self.reward()          # reward BEFORE update_done so reachedTarget() works
        self.update_done()           # now mark agents as done
        obs = self.observation()     # obs reflects new done status
        self.i += 1
        self.checkSpeedDif()

        done_t = (self.i == self.max_episode_len) 
        done_e = (len(self.done) == self.num_flights)

        # self.render() # comment out for training    

        return obs, rew, done_t, done_e, {}

    def checkSpeedDif(self):
        self.average_speed_dif = 0
        speed_dif = np.array([])
        for i, f in enumerate(self.flights):
            speed_dif = np.append(speed_dif, abs(f.airspeed - f.optimal_airspeed))
        self.average_speed_dif = np.average(speed_dif)

    def reset(self, number_flights_training) -> List:
        self.airspace = Airspace.random(self.min_area, self.max_area)
        self.restricted_airspace = RestrictedAirspace.random(self.min_area, self.max_area)
        self.num_flights = number_flights_training
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol, random_init_heading=self.random_init_heading, restricted_airspace=self.restricted_airspace)
            for f in self.flights:
                # Replaced shapely distance with math.hypot for fast reset
                if math.hypot(candidate.position.x - f.position.x, candidate.position.y - f.position.y) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        self.i = 0
        self.conflicts = set()
        self.restricted_airspace_intrusions = set()
        self.done = set()

        minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
        self.world_bounds = (minx, miny, maxx, maxy)

        return self.observation()
    
    def render(self) -> None:
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen_width, self.screen_height = 600, 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Airspace")
            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.world_bounds = (minx, miny, maxx, maxy)

        def world_to_screen(x, y):
            minx, miny, maxx, maxy = self.world_bounds
            sx = int((x - minx) / (maxx - minx) * self.screen_width)
            sy = int(self.screen_height - (y - miny) / (maxy - miny) * self.screen_height)
            return sx, sy

        self.screen.fill(BLACK)

        # Draw main airspace
        sector_pts = [
            world_to_screen(x, y)
            for x, y in self.airspace.polygon.boundary.coords
        ]
        pygame.draw.lines(self.screen, WHITE, False, sector_pts, 1)

        # Draw restricted airspace
        if self.restricted_airspace:
            restricted_pts = [
                world_to_screen(x, y)
                for x, y in self.restricted_airspace.polygon.boundary.coords
            ]
            pygame.draw.lines(self.screen, GREEN, False, restricted_pts, 2)

        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            # Determine color: RED for conflict, YELLOW for restricted airspace, BLUE for normal
            if i in self.conflicts:
                color = RED
            elif i in self.restricted_airspace_intrusions:
                color = YELLOW
            else:
                color = BLUE

            cx, cy = world_to_screen(
                f.reported_position.x,
                f.reported_position.y
            )

            radius = int(
                (self.min_distance / 2.0)
                / (self.world_bounds[2] - self.world_bounds[0])
                * self.screen_width
            )

            pygame.draw.circle(self.screen, color, (cx, cy), radius, 1)

            plan = LineString([f.reported_position, f.target])
            plan_pts = [world_to_screen(x, y) for x, y in plan.coords]
            pygame.draw.lines(self.screen, color, False, plan_pts, 1)

            prediction = LineString([f.reported_position, f.prediction])
            pred_pts = [world_to_screen(x, y) for x, y in prediction.coords]
            pygame.draw.lines(self.screen, color, False, pred_pts, 4)

        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if hasattr(self, "screen"):
            pygame.display.quit()
            pygame.quit()
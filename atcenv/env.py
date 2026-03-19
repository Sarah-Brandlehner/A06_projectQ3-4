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

NUMBER_INTRUDERS_STATE = 2  # start with 2, can try 4 later as single change
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
        self.dt = dt

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        self.viewer = None
        self.airspace = None
        self.flights = [] # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        self.done = set()  # set of flights that reached the target
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
                # Speed change: ±6.67 kts per action (matches reference D_VELOCITY)
                f.airspeed += action[it2][1] * (self.max_speed - self.min_speed) / 10
                f.airspeed = max(min(f.airspeed, self.max_speed), self.min_speed)
                it2 += 1
        return None

    def reward(self) -> List:
        # Penalties per sub-step (accumulated across ACTION_FREQUENCY steps in wrapper)
        # Effective per RL step: drift ≈ -2.5, conflict = -10.0, target = +1.0
        drifts     = self.drift_penalties() * -0.5
        conflicts  = self.conflict_penalties() * -8
        target     = self.reachedTarget() * 1.5
        # proximity  = self.proximity_penalties() * -2.0 # added proximity penalty
        tot_reward = drifts + conflicts + target #+ proximity
        return tot_reward

    def reward_components(self):
        """Return per-flight arrays for each weighted reward component."""
        return {
            "drift":     self.drift_penalties() * -0.5,
            "conflict":  self.conflict_penalties() * -8,
            "target":    self.reachedTarget() * 1.5,
            "proximity": self.proximity_penalties() * -2.0,
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
                drift[i] = abs(f.drift)
        return drift

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


    def observation(self) -> List:
        """
        Returns the observation of each agent using fast NumPy vectorization.
        Layout (5*N + 5): cur_dis, pred_dis, dx, dy, trackdif, airspeed,
        optimal_airspeed, target_dist, sin(drift), cos(drift)
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

            # Ownship state
            obs.append(f.airspeed)
            obs.append(f.optimal_airspeed)
            obs.append(math.hypot(f.position.x - f.target.x, f.position.y - f.target.y))
            obs.append(math.sin(float(f.drift)))
            obs.append(math.cos(float(f.drift)))

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
        self.update_done()
        self.update_conflicts()
        rew = self.reward()
        obs = self.observation()
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
        self.num_flights = number_flights_training
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol)
            for f in self.flights:
                # Replaced shapely distance with math.hypot for fast reset
                if math.hypot(candidate.position.x - f.position.x, candidate.position.y - f.position.y) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        self.i = 0
        self.conflicts = set()
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

        sector_pts = [
            world_to_screen(x, y)
            for x, y in self.airspace.polygon.boundary.coords
        ]
        pygame.draw.lines(self.screen, WHITE, False, sector_pts, 1)

        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            color = RED if i in self.conflicts else BLUE

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
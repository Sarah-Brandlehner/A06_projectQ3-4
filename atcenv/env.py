"""
Environment module
"""
import gymnasium as gym
import pygame
from typing import Dict, List
from atcenv.definitions import *
from shapely.geometry import LineString
from .uncertainties import position_scramble, apply_wind, apply_position_delay

import math as m
# our own packages
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

NUMBER_INTRUDERS_STATE = 5
NUMBER_INTRUDERS_STATE = 2
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

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
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
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param action: list of resolution actions assigned to each flight
        :return:
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
        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

    def reward(self) -> List:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """

        # Simple reward following bluesky-gym reference:
        # 1. Drift penalty: abs(drift_rad) * -0.1  (reference: DRIFT_PENALTY = -0.1)
        # 2. Intrusion penalty: -5.0 per active conflict (strong to ensure avoidance)
        # 3. Target bonus: +1.0 for reaching the target
        drifts     = self.drift_penalties() * -0.1   # drift_penalties returns abs(drift_rad)
        conflicts  = self.conflict_penalties() * -5.0
        target     = self.reachedTarget() * 1.0

        tot_reward = drifts + conflicts + target

        return tot_reward

    def reachedTarget(self):
        """
        Returns a list with aircraft that just reached the target
        :return: boolean list - 1 if aircraft have reached the target
        """        
        target = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    target[i] = 1
                    
        return target

    def speedDifference(self):
        """
        Returns a list with the diferent betwee the current aircraft and its optimal speed
        :return: float of the speed difference
        """
        speed_dif = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            speed_dif[i] = abs(f.airspeed - f.optimal_airspeed) / (self.max_speed - self.min_speed)
                    
        return speed_dif
        
    def conflict_penalties(self):
        """
        Returns a list with aircraft that are in conflict,
        can be used for multiplication as individual reward
        component
        :return: boolean list for conflicts
        """
        
        conflicts = np.zeros(self.num_flights)
        for i in range(self.num_flights):
            if i not in self.done:
                if i in self.conflicts:
                    conflicts[i] += 1
                    
        return conflicts
    
    def drift_penalties(self):
        """
        Returns a list with the drift angle for all aircraft,
        can be used for multiplication as individual reward
        component
        :return: float of the drift angle
        """
        
        drift = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # Return abs(drift) in radians (positive value)
                # Reward function multiplies by -0.1 to match reference
                drift[i] = abs(f.drift)
        
        return drift
            


    def observation(self) -> List:
        """
        Returns the observation of each agent
        :return: observation of each agent
        """
        # observations (size = 2 * NUMBER_INTRUDERS_STATE + 5):
        # distance to closest #NUMBER_INTRUDERS_STATE intruders
        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        # current bearing
        # current speed
        # optimal airspeed
        # distance to target
        # bearing to target

        observations_all = []
        cur_dis     = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        distance_all = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        bearing_all = np.ones((self.num_flights, self.num_flights))*MAX_BEARING
        dx_all  = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        dy_all  = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE

        trackdif_all  = np.ones((self.num_flights, self.num_flights))*3.14
        for i in range(self.num_flights):
            if i not in self.done:
                for j in range(self.num_flights):
                    if j not in self.done and j != i:
                        # predicted used instead of position, so ownship can work in regard to future position and still
                        # avoid a future conflict
                        cur_dis[i][j] = self.flights[i].position.distance(self.flights[j].position)

                        distance_all[i][j] = self.flights[i].prediction.distance(self.flights[j].prediction)

                         # relative bearing
                        dx = self.flights[j].position.x - self.flights[i].position.x
                        dy = self.flights[j].position.y - self.flights[i].position.y
                        compass = math.atan2(dx, dy)                             
                        compass = compass - self.flights[i].track  
                        compass = (compass + u.circle) % u.circle       
                        if compass > math.pi:
                            compass = -(u.circle - compass)
                        elif compass < -math.pi:
                            compass = u.circle + compass
                        bearing_all[i][j] = compass

                        trackdif_all[i][j] = self.flights[i].track - self.flights[j].track

                        dx_all[i][j] = m.sin(float(bearing_all[i][j])) * cur_dis[i][j]
                        dy_all[i][j] = m.cos(float(bearing_all[i][j])) * cur_dis[i][j]

        for i, f in enumerate(self.flights):
            if i not in self.done:
                observations = []

                closest_intruders = np.argsort(distance_all[i])[:NUMBER_INTRUDERS_STATE]

                # distance to closest #NUMBER_INTRUDERS_STATE
                observations += np.take(cur_dis[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < NUMBER_INTRUDERS_STATE:
                    observations.append(MAX_DISTANCE)
                
                observations += np.take(distance_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 2*NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                # relative bearing #NUMBER_INTRUDERS_STATE
                observations += np.take(dx_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 3*NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                observations += np.take(dy_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 4*NUMBER_INTRUDERS_STATE:
                    observations.append(0)
                                
                observations += np.take(trackdif_all[i], closest_intruders).tolist()

                while len(observations) < 5*NUMBER_INTRUDERS_STATE:
                    observations.append(0)
                
                # current speed
                observations.append(f.airspeed)

                # optimal speed
                observations.append(f.optimal_airspeed)

                # distance to target
                observations.append(f.position.distance(f.target))

                # bearing to target
                observations.append(m.sin(float(f.drift)))
                observations.append(m.cos(float(f.drift)))

                observations_all.append(observations)
        # RDC: here you should implement your observation function
        ##########################################################
        return observations_all
        ##########################################################

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.min_distance:
                            self.conflicts.update((i, j))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                if ENABLE_WIND:
                    dx, dy = apply_wind(f, self.wind_magnitude, self.wind_direction)
                else:
                    dx, dy = f.components

                # get current position
                position = f.position

                # get new position and advance one time step
                if ENABLE_DELAY:
                    newx, newy = apply_position_delay(f, PROB_DELAY, MAXIMUM_DELAY, self.dt, dx, dy)
                else:
                    newx = position.x + dx * self.dt
                    newy = position.y + dy * self.dt
                f.position = Point(newx, newy)
                
                # Scramble the position
                if ENABLE_POSITION_UNCERTAINTY:
                    f.reported_position = position_scramble(f.position, PROB_POSITION_UNCERTAINTY, 
                                                0, MAG_POSITION_UNCERTAINTY)
                else:
                    f.reported_position = f.position
                    
                # Store the dx and dy as the previous dx and dy
                f.prev_dx = dx
                f.prev_dy = dy

    def step(self, action: List,) -> Tuple[List, List, bool, Dict]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """
        # apply resolution actions
        self.resolution(action)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # increase steps counter
        self.i += 1

        # store difference from optimal speed
        self.checkSpeedDif()

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done_t = (self.i == self.max_episode_len) 
        done_e = (len(self.done) == self.num_flights)


        #update screen
        self.render() 

        return obs, rew, done_t, done_e, {}

    def checkSpeedDif(self):
        self.average_speed_dif = 0
        speed_dif = np.array([])
        for i, f in enumerate(self.flights):
            speed_dif = np.append(speed_dif, abs(f.airspeed - f.optimal_airspeed))

        self.average_speed_dif = np.average(speed_dif)

    def reset(self, number_flights_training) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # during training, the number of flights will increase from  1 to 10
        self.num_flights = number_flights_training

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if candidate.position.distance(f.position) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = set()

        # reset screen
        minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
        self.world_bounds = (minx, miny, maxx, maxy)

        # return initial observation
        return self.observation()
    

    def render(self) -> None:
        """
        Renders the environment (pygame replacement for gym.envs.classic_control.rendering)
        """

        # -------------------------------------------------
        # initialize viewer (once)
        # -------------------------------------------------
        if not hasattr(self, "screen"):
            pygame.init()

            self.screen_width, self.screen_height = 600, 600
            self.screen = pygame.display.set_mode( (self.screen_width, self.screen_height) )
            pygame.display.set_caption("Airspace")

            # world bounds 
            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.world_bounds = (minx, miny, maxx, maxy)

        # -------------------------------------------------
        # helper: world → screen coordinates
        # -------------------------------------------------
        def world_to_screen(x, y):
            minx, miny, maxx, maxy = self.world_bounds

            sx = int((x - minx) / (maxx - minx) * self.screen_width)
            sy = int(self.screen_height - (y - miny) / (maxy - miny) * self.screen_height )

            return sx, sy

        # -------------------------------------------------
        # background
        # -------------------------------------------------
        self.screen.fill(BLACK)

        # -------------------------------------------------
        # airspace boundary
        # -------------------------------------------------
        sector_pts = [
            world_to_screen(x, y)
            for x, y in self.airspace.polygon.boundary.coords
        ]
        pygame.draw.lines(self.screen, WHITE, False, sector_pts, 1)

        # -------------------------------------------------
        # flights
        # -------------------------------------------------
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            color = RED if i in self.conflicts else BLUE

            # safety circle
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

            # flight plan
            plan = LineString([f.reported_position, f.target])
            plan_pts = [world_to_screen(x, y) for x, y in plan.coords]
            pygame.draw.lines(self.screen, color, False, plan_pts, 1)

            # prediction
            prediction = LineString([f.reported_position, f.prediction])
            pred_pts = [world_to_screen(x, y) for x, y in prediction.coords]
            pygame.draw.lines(self.screen, color, False, pred_pts, 4)

        # -------------------------------------------------
        # update display + keep window responsive
        # -------------------------------------------------
        pygame.display.flip()
        pygame.event.pump()


    def close(self):
        if hasattr(self, "screen"):
            pygame.display.quit()
            pygame.quit()

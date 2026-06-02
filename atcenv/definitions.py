"""
Definitions module
"""
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
from dataclasses import dataclass, field
import atcenv.units as u
import math
import random
from typing import Optional, Tuple


@dataclass
class Airspace:
    """
    Airspace class
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float):
        """
        Creates a random airspace sector with min_area < area <= max_area

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :return: random airspace
        """
        R = math.sqrt(max_area / math.pi)

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        p = [random_point_in_circle(R) for _ in range(5)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)
    
@dataclass
class RestrictedAirspace:
    """
    Restricted Airspace class - smaller airspace inside the main airspace
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float, scale_factor: float = 0.2):
        """
        Creates a random restricted airspace sector inside the main airspace
        
        :param max_area: maximum area of the main sector (in nm^2)
        :param min_area: minimum area of the main sector (in nm^2)
        :param scale_factor: scale factor for the restricted airspace (0-1, default 0.2 for 20% size)
        :return: random restricted airspace
        """
        # Scale down the radius for a smaller airspace
        R = math.sqrt(max_area / math.pi) * scale_factor

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        
        p = [random_point_in_circle(R) for _ in range(4)]
        polygon = Polygon(p).convex_hull

        # Scale down the minimum area requirement proportionally
        scaled_min_area = min_area * (scale_factor ** 2)
        
        while polygon.area < scaled_min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)

@dataclass
class RestrictedAirspace:
    """
    Restricted Airspace class - smaller airspace inside the main airspace
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float, scale_factor: float = 0.2):
        """
        Creates a random restricted airspace sector inside the main airspace
        
        :param max_area: maximum area of the main sector (in nm^2)
        :param min_area: minimum area of the main sector (in nm^2)
        :param scale_factor: scale factor for the restricted airspace (0-1, default 0.2 for 20% size)
        :return: random restricted airspace
        """
        # Scale down the radius for a smaller airspace
        R = math.sqrt(max_area / math.pi) * scale_factor

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        
        p = [random_point_in_circle(R) for _ in range(4)]
        polygon = Polygon(p).convex_hull

        # Scale down the minimum area requirement proportionally
        scaled_min_area = min_area * (scale_factor ** 2)
        
        while polygon.area < scaled_min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)


@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    target: Point
    optimal_airspeed: float
    random_init_heading: bool = True

    airspeed: float = field(init=False)
    track: float = field(init=False)
    
    prev_dx: float = field(init=False)
    prev_dy: float = field(init=False)
    
    reported_position: Point = None

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        if self.random_init_heading:
            self.track = random.uniform(0, 2 * math.pi)
        else:
            self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        
        # Initialise previous speeds
        self.prev_dx = self.airspeed * math.sin(self.track)
        self.prev_dy = self.airspeed * math.cos(self.track)
        
        # The reported position, not the actual one
        self.reported_position = self.position
        # The action delay still left before 
        self.action_delay = -999
        self.delayed_action = None

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target
        :return:
        """
        if self.reported_position is not None:
            dx = self.target.x - self.reported_position.x
            dy = self.target.y - self.reported_position.y
        else:
            dx = self.target.x - self.position.x
            dy = self.target.y - self.position.y
        compass = math.atan2(dx, dy)
        return (compass + u.circle) % u.circle

    @property
    def prediction(self, dt: Optional[float] = 15) -> Point:
        """
        Predicts the future position after dt seconds, maintaining the current speed and track
        :param dt: prediction look-ahead time (in seconds)
        :return:
        """
        dx, dy = self.components
        return Point([self.reported_position.x + dx * dt, self.reported_position.y + dy * dt])

    @property
    def components(self) -> Tuple:
        """
        X and Y Speed components (in kt)
        :return: speed components
        """
        dx = self.airspeed * math.sin(self.track)
        dy = self.airspeed * math.cos(self.track)
        return dx, dy

    @property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return self.reported_position.distance(self.target)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift

    def in_restricted_airspace(self, restricted_airspace: 'RestrictedAirspace') -> bool:
        """
        Check if the aircraft is currently in the restricted airspace
        
        :param restricted_airspace: RestrictedAirspace object
        :return: True if aircraft is in restricted airspace, False otherwise
        """
        if restricted_airspace is None:
            return False
        return restricted_airspace.polygon.contains(self.position)

    def heading_into_restricted_airspace(self, restricted_airspace: 'RestrictedAirspace', lookahead: float = 50000.0) -> bool:
        """
        Check if the aircraft's current heading line intersects with the restricted airspace
        
        :param restricted_airspace: RestrictedAirspace object
        :param lookahead: distance to look ahead along the current track (in meters), default 50km
        :return: True if heading line intersects restricted airspace, False otherwise
        """
        if restricted_airspace is None:
            return False
        
        # Create a line from current position extending in the direction of current track
        dx = math.sin(self.track) * lookahead
        dy = math.cos(self.track) * lookahead
        
        end_point = Point(self.position.x + dx, self.position.y + dy)
        heading_line = LineString([self.position, end_point])
        
        # Check if the heading line intersects with the restricted airspace boundary
        return heading_line.intersects(restricted_airspace.polygon)

    def closest_restricted_point(self, restricted_airspace: 'RestrictedAirspace'):
        """
        Get the distance and relative heading/proximity features to the closest point on the restricted airspace boundary.

        :param restricted_airspace: RestrictedAirspace object
        :return: (distance, sin_bearing, cos_bearing, approach_dot)
        """
        if restricted_airspace is None:
            return 0.0, 0.0, 0.0, 0.0

        point = Point(self.position.x, self.position.y)
        poly = restricted_airspace.polygon
        nearest = nearest_points(poly, point)
        closest_point = nearest[0]  # point on polygon

        dx = closest_point.x - self.position.x
        dy = closest_point.y - self.position.y
        dist = math.hypot(dx, dy)

        # Relative bearing from current heading to closest restricted vertex
        if dist > 1e-6:
            rel_brg = math.atan2(dx, dy) - self.track
            rel_brg = (rel_brg + math.pi) % (2 * math.pi) - math.pi
            s_brg = math.sin(rel_brg)
            c_brg = math.cos(rel_brg)
        else:
            s_brg, c_brg = 0.0, 1.0

        # Approach rate: projection of current velocity onto vector to restricted point
        v_dx, v_dy = self.components
        approach = 0.0
        if dist > 1e-6:
            approach = (v_dx * dx + v_dy * dy) / dist

        return dist, s_brg, c_brg, approach

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, tol: float = 0., random_init_heading: bool = True, restricted_airspace: Optional['RestrictedAirspace'] = None):
        """
        Creates a random flight

        :param airspace: airspace where the flight is located
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param tol: tolerance to consider that the target has been reached (in meters)
        :param random_init_heading: whether to use random initial heading
        :param restricted_airspace: optional RestrictedAirspace to avoid spawning in
        :return: random flight
        """
        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        # random position - avoid restricted airspace if provided
        position = random_point_in_polygon(airspace.polygon)
        while restricted_airspace is not None and restricted_airspace.polygon.contains(position):
            position = random_point_in_polygon(airspace.polygon)

        # random target
        boundary = airspace.polygon.boundary
        while True:
            d = random.uniform(0, airspace.polygon.boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > tol:
                break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)

        return cls(position, target, airspeed, random_init_heading=random_init_heading)
    def in_restricted_airspace(self, restricted_airspace: 'RestrictedAirspace') -> bool:
        """
        Check if the aircraft is currently in the restricted airspace
        
        :param restricted_airspace: RestrictedAirspace object
        :return: True if aircraft is in restricted airspace, False otherwise
        """
        if restricted_airspace is None:
            return False
        return restricted_airspace.polygon.contains(self.position)

    def heading_into_restricted_airspace(self, restricted_airspace: 'RestrictedAirspace', lookahead: float = 50000.0) -> bool:
        """
        Check if the aircraft's current heading line intersects with the restricted airspace
        
        :param restricted_airspace: RestrictedAirspace object
        :param lookahead: distance to look ahead along the current track (in meters), default 50km
        :return: True if heading line intersects restricted airspace, False otherwise
        """
        if restricted_airspace is None:
            return False
        
        # Create a line from current position extending in the direction of current track
        dx = math.sin(self.track) * lookahead
        dy = math.cos(self.track) * lookahead
        
        end_point = Point(self.position.x + dx, self.position.y + dy)
        heading_line = LineString([self.position, end_point])
        
        # Check if the heading line intersects with the restricted airspace boundary
        return heading_line.intersects(restricted_airspace.polygon)

    def closest_restricted_point(self, restricted_airspace: 'RestrictedAirspace'):
        """Finds the single closest point on the restricted boundary and returns (dist, sin_brg, cos_brg, approach_rate)"""
        if restricted_airspace is None:
            return 0.0, 0.0, 0.0, 0.0

        # Find the point on the exterior linear ring closest to aircraft position
        exterior = restricted_airspace.polygon.exterior
        closest_p = exterior.interpolate(exterior.project(self.position))

        dx, dy = closest_p.x - self.position.x, closest_p.y - self.position.y
        dist = math.hypot(dx, dy)

        # Relative bearing from current heading to the closest point
        if dist > 1e-6:
            rel_brg = math.atan2(dx, dy) - self.track
            rel_brg = (rel_brg + math.pi) % (2 * math.pi) - math.pi
            s_brg = math.sin(rel_brg)
            c_brg = math.cos(rel_brg)
            v_dx, v_dy = self.components
            approach = (v_dx*dx + v_dy*dy) / dist
        else:
            s_brg, c_brg = 0.0, 1.0
            approach = 0.0

        return dist, s_brg, c_brg, approach


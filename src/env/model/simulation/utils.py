import shapely.geometry
from shapely.geometry import Point, LineString
import math
import numpy as np

# Constants, which should not be changed.
TILE_SIZE_PX = 16  # The size of a single tile in pixels
TILE_SIZE_METERS = 2.0  # The size of a single tile in meters
ROAD_TILE_COUNT = 6  # The count of tiles, which a road is made of
ROAD_PIXEL_COUNT = ROAD_TILE_COUNT * TILE_SIZE_PX  # The count of pixels which a road is made of
ROAD_METER_COUNT = ROAD_TILE_COUNT * TILE_SIZE_METERS  # The count of meters which a road is made of

CAR_MAX_WIDTH = 2.2
CAR_MAX_LENGTH = 6.0


class SimVersions:
    VERSION_0 = 0  # Only 48 patches
    VERSION_1 = 1  # Patches match with the route lookahead. This means that there will be lookahead patches in front of the agent
    VERSION_2 = 2  # During training, the size of the road grid and number of traffic participants can be randomized
    VERSION_3 = 3  # Removed the pedestrian threshold. It is now a factor.
    VERSION_4 = 4  # Added more actions to the agent
    VERSION_5 = 5  # Changed right of way calculation. The can_fit check is now done by the individual cars. Cars will wait until they can fit. But they keep the right of way


def get_velocity_time_way_acceleration(s, a, t, v_0=0):
    """
    This method finds the distance traveled or displacement (s) of an object using its initial velocity (v_0), acceleration (a), and time (t) traveled.

    Depending on which values are given, it can be used to solve for s, v_0, a or t. The variable, which you are solving for, should be None.
    The other variables then need to be floats.

    :param s: Displacement
    :param a: Acceleration
    :param t: Time in seconds
    :param v_0: Initial velocity
    :return: Solution
    """
    # Online Calculator: https://www.calculatorsoup.com/calculators/physics/displacement_v_a_t.php
    if s is None:
        return 0.5 * a * math.pow(t, 2) + v_0 * t
    elif a is None:
        return (2 * (s - v_0 * t)) / (t ** 2)
    elif t is None:
        p = v_0 * 1 / float((0.5 * a))
        q = -s * 1 / float((0.5 * a))
        return -p / 2 + math.sqrt(math.pow((p / 2.0), 2) - q), -p / 2 - math.sqrt(math.pow((p / 2.0), 2) - q)
    else:
        return s / t - 1 / 2 * a * t


def get_normalized(value, min_value, max_value):
    """
    Returns the normalized value between [0,1]

    :param max_value: The maximal possible value
    :param min_value: The minimal possible value
    :param value: The value which should be normalized
    :return: The normalized value in [0,1]
    """
    return float((value - min_value)) / (max_value - min_value)


def get_nearby_grid(grid, grid_x, grid_y, offset_x, offset_y):
    """
    Returns the nearby grid array around grid_x, grid_y with offset offset_x, offset_y.

    :param grid: The whole grid array
    :param grid_x: The x coordinate
    :param grid_y: The y coordinate
    :param offset_x: The offset on the x axis
    :param offset_y: The offset on the y axis
    :return: A cutout of the grid array at grid_x, grid_y with the given offset
    """
    min_x, min_y = max(grid_x - offset_x, 0), max(grid_y - offset_y, 0)
    max_x, max_y = min(grid_x + offset_x + 1, grid.shape[0]), min(grid_y + offset_y + 1, grid.shape[1])

    return grid[min_x:max_x, min_y:max_y]


def get_merged_threshold_intervals(sorted_intervals: list[list[float]], threshold) -> list[list[float]]:
    """
    Merges intervals, which have a distance, which is smaller than the given threshold.

    :param sorted_intervals: The sorted non-overlapping intervals
    :param threshold: The threshold
    :return: The merged intervals sorted by their start value
    """
    stack = []
    if len(sorted_intervals) > 0:
        stack.append(sorted_intervals[0])

        for interval in sorted_intervals[1:]:
            if interval[0] - stack[-1][1] <= threshold:
                stack[-1][1] = interval[1]
            else:
                stack.append(interval)

    return stack


def get_merged_overlapping_intervals(intervals: list[list[float]]) -> list[list[float]]:
    """
    Merges overlapping intervals.

    :param intervals: The overlapping intervals as an array
    :return: The merged intervals sorted by their start value
    """
    stack = []
    if len(intervals) > 0:
        # Merge Overlapping Intervals

        # Sort based on start of intervals
        intervals.sort(key=lambda x: x[0])

        stack.append(intervals[0])

        for interval in intervals[1:]:
            # Check for overlapping interval on top of stack
            if stack[-1][0] <= interval[0] <= stack[-1][1]:
                stack[-1][1] = max(stack[-1][1], interval[1])
            else:
                stack.append(interval)

    return stack


def get_meters_per_second(kilometers_per_hour: float) -> float:
    """
    Turns km/h into m/s.

    :param kilometers_per_hour: The km/h value
    :return: Value in m/s
    """

    return float((kilometers_per_hour * 1000)) / 60.0 / 60.0


def get_kilometers_per_hour(meters_per_second: float) -> float:
    """
    Turns m/s into km/h.

    :param meters_per_second: The m/s value
    :return: Value in km/h
    """

    return float((meters_per_second / 1000)) * 60.0 * 60.0


def get_normalized_vector(x):
    """
    Normalizes x.

    :param x: The vector
    :return: The normalized vector
    """
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    else:
        return np.array([0, 0])


def get_vector(x):
    """
    Converts x to a numpy array vector.

    :param x: A vector-like object. Can be a list, tuple or a Point
    :return: x as a numpy array
    """
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, tuple):
        return np.array([x[0], x[1]])
    elif isinstance(x, shapely.geometry.Point):
        return np.array([x.x, x.y])
    else:
        raise ValueError("Unknown Type for vector conversion!")


def get_grid_index_from_pos(grid, x, y):
    """
    Returns the grid object index in the grid array based on a position measured in meters.

    :param grid: The grid array
    :param x: The x coordinate in meters
    :param y: The y coordinate in meters
    :return: The grid object index
    """
    index_x = int(x / ROAD_METER_COUNT)
    index_y = int(y / ROAD_METER_COUNT)

    index_x = max(0, min(index_x, grid.shape[0] - 1))
    index_y = max(0, min(index_y, grid.shape[1] - 1))

    return index_x, index_y


def get_tile_from_pos(grid, x, y):
    """
    Returns the tile index in the tile grid array based on a position measured in meters.

    :param grid: The tile grid array
    :param x: The x coordinate in meters
    :param y: The y coordinate in meters
    :return: The tile at the position
    """

    index_x, index_y = get_tile_index_from_pos(grid, x, y)
    return grid[index_x, index_y]


def get_tile_index_from_pos(grid, x, y):
    """
    Returns the tile index in the tile grid array based on a position measured in meters.

    :param grid: The tile grid array
    :param x: The x coordinate in meters
    :param y: The y coordinate in meters
    :return: The tile index
    """

    index_x = int(x / TILE_SIZE_METERS)
    index_y = int(y / TILE_SIZE_METERS)

    index_x = max(0, min(index_x, grid.shape[0] - 1))
    index_y = max(0, min(index_y, grid.shape[1] - 1))

    return index_x, index_y


class Turn:
    """
    Can be used to define the next action of a vehicle. It can either turn right, go straight or turn left.
    """
    MIN_VALUE = -1
    MAX_VALUE = 1

    RIGHT = 1
    STRAIGHT = 0
    LEFT = -1

    @staticmethod
    def get_name(turn):
        if turn == Turn.STRAIGHT:
            return "straight"
        elif turn == Turn.RIGHT:
            return "right"
        elif turn == Turn.LEFT:
            return "left"


class AgentTypes:
    """
    This class holds the different agent types.
    """

    DQN = 0
    ACCELERATE = 1
    IDM = 2
    TTC = 3
    Simulation = 4
    TTC_CREEP = 5

    def get_name(self, agent_type):
        for variable in vars(AgentTypes):
            if getattr(variable) == agent_type:
                return variable

        return "This agent type does not exist!"


class AgentDirections:
    """
    This class holds methods and values for relative directions. This can be used to specify if another car is to the
    right or left of an agent.
    """

    MIN_VALUE = 0
    MAX_VALUE = 3

    SAME = 0
    LEFT = 1
    OPPOSITE = 2
    RIGHT = 3

    @staticmethod
    def get_car_direction(car_direction, agent_direction):
        """
        This returns the relative direction of another car to the agent.

        :param car_direction: The direction of the other car
        :param agent_direction: The direction of the agent
        :return: The relative direction of the other car to the agent
        """
        if agent_direction == Directions.DOWN:
            if car_direction == Directions.LEFT:
                return AgentDirections.LEFT
            elif car_direction == Directions.DOWN:
                return AgentDirections.SAME
            elif car_direction == Directions.RIGHT:
                return AgentDirections.RIGHT
            elif car_direction == Directions.UP:
                return AgentDirections.OPPOSITE

        elif agent_direction == Directions.RIGHT:
            if car_direction == Directions.LEFT:
                return AgentDirections.OPPOSITE
            elif car_direction == Directions.DOWN:
                return AgentDirections.LEFT
            elif car_direction == Directions.RIGHT:
                return AgentDirections.SAME
            elif car_direction == Directions.UP:
                return AgentDirections.RIGHT

        elif agent_direction == Directions.UP:
            if car_direction == Directions.LEFT:
                return AgentDirections.RIGHT
            elif car_direction == Directions.DOWN:
                return AgentDirections.OPPOSITE
            elif car_direction == Directions.RIGHT:
                return AgentDirections.LEFT
            elif car_direction == Directions.UP:
                return AgentDirections.SAME

        elif agent_direction == Directions.LEFT:
            if car_direction == Directions.LEFT:
                return AgentDirections.SAME
            elif car_direction == Directions.DOWN:
                return AgentDirections.RIGHT
            elif car_direction == Directions.RIGHT:
                return AgentDirections.OPPOSITE
            elif car_direction == Directions.UP:
                return AgentDirections.LEFT


class Directions:
    """
    Holds methods and constants for directions.
    """

    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    @staticmethod
    def get_name(direction):
        """
        Returns the name of a direction.

        :param direction: The direction
        :return: The name of the direction
        """
        if direction == Directions.UP:
            return "up"
        elif direction == Directions.LEFT:
            return "left"
        elif direction == Directions.RIGHT:
            return "right"
        elif direction == Directions.DOWN:
            return "down"

    @staticmethod
    def get_opposite(direction):
        """
        Returns the opposite direction.

        :param direction: The direction
        :return: The opposite direction
        """
        return (2 + direction) % 4

    @staticmethod
    def get_opposite_offset(offset):
        """
        Returns the opposite offset.

        :param offset: The offset
        :return: The opposite offset
        """
        return offset[0] * -1, offset[1] * -1

    @staticmethod
    def get_grid_offset(direction):
        """
        Returns the grid offset based on a Direction.

        :param direction: The direction
        :return: The grid offset
        """
        if direction == Directions.UP:
            return 0, -1
        elif direction == Directions.LEFT:
            return -1, 0
        elif direction == Directions.DOWN:
            return 0, 1
        elif direction == Directions.RIGHT:
            return 1, 0

    @staticmethod
    def get_direction_from_offset(offset):
        """
        Returns the direction based on a grid offset.

        :param offset: The grid offset
        :return: The direction
        """
        if offset == (0, -1):
            return Directions.UP
        elif offset == (-1, 0):
            return Directions.LEFT
        elif offset == (0, 1):
            return Directions.DOWN
        elif offset == (1, 0):
            return Directions.RIGHT

    @staticmethod
    def get_direction_from_coords(start, end):
        """
        This method calculates the entry or exit direction from two points. End is always the middle point.
        Start is always the entry/exit point.

        :param start: The start point
        :param end: The end point
        :return: Entry or exit direction
        """
        x_start, y_start = start
        x_end, y_end = end

        x_dir = x_end - x_start
        y_dir = y_end - y_start

        if x_dir == 0:
            if y_dir > 0:
                return Directions.UP
            elif y_dir < 0:
                return Directions.DOWN
            else:
                raise ValueError("y_dir can not be 0 as well!")
        elif y_dir == 0:
            if x_dir > 0:
                return Directions.LEFT
            elif x_dir < 0:
                return Directions.RIGHT
            else:
                raise ValueError("x_dir can not be 0 as well!")
        else:
            raise ValueError("This only works if one of the coordinates stays the same!")

    @staticmethod
    def rotate90(direction, times):
        """
        Rotates a given direction 90 degrees times the given parameter.

        :param direction: The direction which should be rotated
        :param times: The count of 90 degree rotations
        :return: The rotated direction
        """
        return (-1 * times + direction) % 4


class LaneSides:
    FORWARD = 1
    BACKWARD = -1

    @staticmethod
    def inversed(side):
        """
        Returns the inversed lane side.

        :param side: The lane side which should be inversed
        :return: The inversed lane side
        """
        return side * -1


def get_compass_direction(p1: tuple[float, float], p2: tuple[float, float]) -> int:
    """
    Returns the compass direction from two points. Calculation is based on pygames' coordinate system.

    :param p1: The first point
    :param p2: The second point
    :return: The compass direction from the two points
    """

    angle = angle_between(p1, p2)

    if angle >= 315 or angle < 45:
        return Directions.LEFT
    elif 225 <= angle < 315:
        return Directions.UP
    elif 45 <= angle < 135:
        return Directions.DOWN
    else:
        return Directions.RIGHT


def angle_between(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Returns the angle between two points. Calculation is based on pygames' coordinate system.

    :param p1: The first point
    :param p2: The second point
    :return: The angle between the two points
    """

    if isinstance(p1, Point):
        p1 = p1.x, p1.y

    if isinstance(p2, Point):
        p2 = p2.x, p2.y

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(-dy, dx)
    rads %= 2 * math.pi
    degs = math.degrees(rads)
    return degs


def cut_line(line: LineString, distance: float) -> list[LineString, LineString]:
    """
    Cuts a line in two at a distance from its starting point.

    :param line: The LineString, which should be cut
    :param distance: The distance from the start of the given LineString
    :return:
    """
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [line, line]
    coords = list(line.coords)
    for i, point in enumerate(coords):
        point_distance = line.project(Point(point))
        if point_distance == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if point_distance > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


if __name__ == '__main__':
    pass

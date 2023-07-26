import numpy as np
import shapely.affinity
from shapely.geometry import LineString, box
from shapely import affinity

from .utils import *


class Tile():
    """
    The class for tiles, which roads are made of. A road is made of 6x6 tiles.
    """

    def __init__(self, id, text, config={}):
        # Set default configuration
        self.set_default_config()

        # Update configuration
        self.id = id
        self.text = text
        for attr, val in config.items():
            setattr(self, attr, val)

    def set_default_config(self):
        self.text = "Generic Tile"
        self.id = -1
        self.is_wall = False
        self.is_obstacle = False
        self.is_road = False

    def __str__(self):
        return self.text

    def __eq__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:  # it is better to use isinstance, but the relative importing breaks this technique
            return other.id == self.id
        return False


class ATiles:
    """
    A class, which holds all abstract tiles.
    """

    INNER_BORDER = Tile(-1, "inner_border")
    BORDER = Tile(0, "border")
    NOTHING = Tile(1, "nothing")
    ROAD_F = Tile(2, "road_f", config={'is_road': True})
    ROAD_B = Tile(3, "road_b", config={'is_road': True})
    WALL_F = Tile(4, "wall_f", config={'is_wall': True})
    WALL_B = Tile(5, "wall_b", config={'is_wall': True})
    OBSTACLE = Tile(6, "obstacle", config={'is_obstacle': True})


def inverse_tile(tile):
    """
    Returns the inverse tile of a given tile.
    :param tile: The tile that should be inversed
    :return: The inversed variant of that tile
    """
    if tile == ATiles.ROAD_F:
        return ATiles.ROAD_B
    elif tile == ATiles.ROAD_B:
        return ATiles.ROAD_F
    elif tile == ATiles.WALL_F:
        return ATiles.WALL_B
    elif tile == ATiles.WALL_B:
        return ATiles.WALL_F
    else:
        return tile


def get_oriented_line(line_string: LineString, rot90: int, flip_lr: bool, flip_ud: bool) -> LineString:
    """
    Orients a LineString using the given parameters.

    :param line_string: The line string, which should be oriented
    :param rot90: The amount of times the linestring gets rotated by 90 degrees
    :param flip_lr: Flip Left and Right?
    :param flip_ud: Flip Up and Down?
    :return: The new oriented LineString
    """

    coords = line_string.coords
    if flip_lr:
        coords = list(reversed(coords))
    if flip_ud:
        coords = list(reversed(coords))

        # The geometric object representing the lane
    line = LineString(coords)

    if rot90 > 0:
        line = affinity.rotate(line, 90 * rot90, origin=(6, 6))
    if flip_lr:
        line = affinity.scale(line, xfact=-1, yfact=1, origin=(6, 6))
    if flip_ud:
        line = affinity.scale(line, xfact=1, yfact=-1, origin=(6, 6))

    return line


def get_global_geometry(local_geometry, grid_x, grid_y):
    """
    Transforms a geometry in local space to global space based on a grid position.

    :param local_geometry: The local geometry
    :param grid_x: The grid x coord, where the geometry starts
    :param grid_y: he grid y coord, where the geometry starts
    :return: The geometry in global space
    """

    return shapely.affinity.translate(local_geometry, grid_x * ROAD_TILE_COUNT * TILE_SIZE_METERS,
                                      grid_y * ROAD_TILE_COUNT * TILE_SIZE_METERS)


class Lane():
    """
    The class which specifies lanes on roads.
    """

    def __init__(self, coords: list[tuple[float, float]], lane_side: int, config={}):
        """
        :param coords: The coordinates for the geometric line of the lane
        :param lane_side: The side of the lane
        :param config: Optional Configuration for the lane
        """

        # The car queue contains cars, which are currently on this lane
        self.car_queue = []

        # The upcoming car queue contains cars, which will arrive at this lane soon
        self.upcoming_car_queue = []

        # Set default configuration
        self.grid_x = 0
        self.grid_y = 0
        self.global_line = None

        # Flipped left to right
        self.flip_lr = False

        # Flipped up to down
        self.flip_ud = False

        # Times road is rotated by 90 degrees
        self.rot90 = 0

        # Inverse forward and backward lanes
        self.inversed = False

        # Either Forward or Backward lane
        self.lane_side = lane_side

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        if self.inversed:
            self.lane_side = LaneSides.inversed(self.lane_side)

        self.local_line = get_oriented_line(LineString(coords), rot90=self.rot90, flip_lr=self.flip_lr,
                                            flip_ud=self.flip_ud)
        new_coords = self.local_line.coords

        # The entry direction is the grid offset direction, that can be used to find the road before this road
        # The exit direction is the grid offset direction, that can be used to find the road after this road
        self.entry_direction = Directions.get_direction_from_coords(new_coords[0], new_coords[1])
        self.exit_direction = Directions.get_direction_from_coords(new_coords[-1], new_coords[-2])
        self.entry_offset = Directions.get_grid_offset(self.entry_direction)
        self.exit_offset = Directions.get_grid_offset(self.exit_direction)

        # The turn direction of the lane
        self.turn = self.get_turn(self.entry_direction, self.exit_direction)

        self.local_line = LineString(
            self.get_curve_points(self.local_line.coords[0], self.local_line.coords[2], self.local_line.coords[1]))

    @staticmethod
    def get_turn(entry: int, exit: int) -> int:
        """
        Returns the Turn (Left, Right, Straight) from an entry and exit direction.

        :param entry: The entry direction
        :param exit: The exit direction
        :return: The corresponding Turn
        """

        exit_o = Directions.get_opposite(exit)
        turn = None
        if entry == exit_o:
            turn = Turn.STRAIGHT
        else:
            if entry == Directions.DOWN:
                if exit == Directions.RIGHT:
                    turn = Turn.RIGHT
                elif exit == Directions.LEFT:
                    turn = Turn.LEFT
            elif entry == Directions.LEFT:
                if exit == Directions.DOWN:
                    turn = Turn.RIGHT
                if exit == Directions.UP:
                    turn = Turn.LEFT
            elif entry == Directions.UP:
                if exit == Directions.LEFT:
                    turn = Turn.RIGHT
                elif exit == Directions.RIGHT:
                    turn = Turn.LEFT
            elif entry == Directions.RIGHT:
                if exit == Directions.UP:
                    turn = Turn.RIGHT
                elif exit == Directions.DOWN:
                    turn = Turn.LEFT

        if turn is None:
            raise ValueError("Wrong value for turn direction!")

        return turn

    def register_car(self, car, upcoming=False):
        """
        Registers a vehicle on this lane.

        :param car: The vehicle, which should be registered
        :param upcoming: Specifies if the car is on this lane or if it will soon be on the lane
        """
        if upcoming:
            self.upcoming_car_queue.append(car)
        else:
            self.car_queue.append(car)

    def unregister_car(self, car, upcoming=False):
        """
        Unregisters a vehicle on this lane.

        :param car: The vehicle, which should be registered
        :param upcoming: Specifies if the car should be unregistered from the current car queue or the upcoming car queue
        """
        if upcoming:
            self.upcoming_car_queue.remove(car)
        else:
            self.car_queue.remove(car)

    def get_first_car(self):
        """
        Returns the first car at the beginning of this lane. This is the car, which joined this lane recently.

        :return: The first car
        """
        if len(self.car_queue) > 0:
            return self.car_queue[len(self.car_queue) - 1]
        else:
            return None

    def get_last_car(self):
        """
        Returns the last car at the end of this lane. This is the car, which will leave this lane first.

        :return: The last car
        """
        if len(self.car_queue) > 0:
            return self.car_queue[0]
        else:
            return None

    def get_road(self, grid: np.ndarray) -> "Road":
        """
        Returns the road for this lane.

        :param grid: The road grid
        :return: The road tile
        """
        next_road = grid[self.grid_x, self.grid_y]
        return next_road

    def get_previous_road(self, grid: np.ndarray) -> "Road":
        """
        Returns the previous road for this lane.

        :param grid: The road grid
        :return: The previous road tile
        """
        dx, dy = self.entry_offset
        previous_road = grid[self.grid_x + dx, self.grid_y + dy]
        return previous_road

    def get_next_road(self, grid: np.ndarray):
        """
        Returns the next road for this lane.

        :param grid: The road grid
        :return: The next road tile
        """
        dx, dy = self.exit_offset
        next_road = grid[self.grid_x + dx, self.grid_y + dy]
        return next_road

    def get_next_lanes(self, grid: np.ndarray):
        """
        Returns the next possible lanes for this lane.

        :param grid: The road grid
        :return: The next lanes
        """
        entry = Directions.get_opposite(self.exit_direction)
        return self.get_next_road(grid).lanes[entry]

    def get_previous_lanes(self, grid: np.ndarray):
        """
        Returns the previous possible lanes for this lane.

        :param grid: The road grid
        :return: The previous lanes
        """
        exit = Directions.get_opposite(self.entry_direction)
        return self.get_previous_road(grid).lanes_exit[exit]

    def initialize_map_coords(self, grid_x, grid_y):
        """
        Sets the grid coordinates for this lane.

        :param grid_x: The x coordinate on the grid
        :param grid_y: The y coordinate on the grid
        """
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.global_line = get_global_geometry(self.local_line, self.grid_x, self.grid_y)

    def clean_up(self):
        """
        Deletes unused objects.
        """
        del self.car_queue
        del self.upcoming_car_queue

    @staticmethod
    def get_curve_points(start, end, control, resolution=20):
        """
        Uses Bezier curves to get a curve path.

        :param start: Start Coordinates
        :param end: End Coordinates
        :param control: The Control Coordinate
        :param resolution: The resolution
        :return: The points on the curve path
        """
        # If curve is a straight line
        if (start[0] - end[0]) * (start[1] - end[1]) == 0:
            return [start, control, end]

        # If not return a curve
        path = []

        for i in range(resolution + 1):
            t = i / resolution
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
            path.append((x, y))

        return path


class GridObject(object):
    """
    The superclass for all grid objects including roads. A grid object is made out of tiles, which specify an occupancy grid.
    """

    def __init__(self, config={}):
        """
        :param config: The optional configuration
        """
        # Set default configuration
        self.flip_lr = False
        self.flip_ud = False
        self.rot90 = 0
        self.inversed = False
        self.name = "generic nothing"
        self.symbol = "g "
        self.tiles = [[ATiles.NOTHING for i in range(ROAD_TILE_COUNT)],
                      [ATiles.NOTHING for i in range(ROAD_TILE_COUNT)],
                      [ATiles.NOTHING for i in range(ROAD_TILE_COUNT)],
                      [ATiles.NOTHING for i in range(ROAD_TILE_COUNT)],
                      [ATiles.NOTHING for i in range(ROAD_TILE_COUNT)],
                      [ATiles.NOTHING for i in range(ROAD_TILE_COUNT)]]
        self.grid_x = 0
        self.grid_y = 0

        self.local_rect = box(0, 0, 12, 12)
        self.global_rect = None

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        if "fill" in config:
            fill_tile = config["fill"]
            self.tiles = [[fill_tile for i in range(ROAD_TILE_COUNT)],
                          [fill_tile for i in range(ROAD_TILE_COUNT)],
                          [fill_tile for i in range(ROAD_TILE_COUNT)],
                          [fill_tile for i in range(ROAD_TILE_COUNT)],
                          [fill_tile for i in range(ROAD_TILE_COUNT)],
                          [fill_tile for i in range(ROAD_TILE_COUNT)]]

        if isinstance(self.tiles, list):
            self.tiles = np.asarray(self.tiles, dtype=Tile)
            self.tiles = np.flipud(np.rot90(self.tiles, k=1))

        if self.rot90 > 0:
            self.tiles = np.rot90(self.tiles, k=self.rot90)
        if self.flip_lr:
            self.tiles = np.flipud(self.tiles)  # This is intentionally ud instead of lr
        if self.flip_ud:
            self.tiles = np.fliplr(self.tiles)  # This is intentionally lr instead of ud
        if self.inversed:
            self.tiles = np.array(
                [[inverse_tile(tile) for tile in x] for x in self.tiles])  # Switches tiles with their inverse tiles

    def clean_up(self):
        del self.tiles
        del self.global_rect
        del self.local_rect

    def is_road(self) -> bool:
        """
        :return: Is this a road?
        """
        return False

    def is_corner(self) -> bool:
        """
        :return: Is this a corner road?
        """
        return False

    def is_inter(self) -> bool:
        """
        :return: Is this an intersection?
        """
        return False

    def is_cross(self) -> bool:
        """
        :return: Is this an intersection with four entrances?
        """
        return False

    def is_obstacle(self) -> bool:
        """
        :return: Is this an obstacle?
        """
        return False

    def initialize_map_coords(self, grid_x, grid_y):
        """
        Sets the grid coordinates for this road.

        :param grid_x: The x coordinate on the grid
        :param grid_y: The y coordinate on the grid
        """
        self.grid_x = grid_x
        self.grid_y = grid_y

        self.global_rect = box(0 + grid_x * ROAD_METER_COUNT, 0 + grid_y * ROAD_METER_COUNT,
                               12 + grid_x * ROAD_METER_COUNT, 12 + grid_y * ROAD_METER_COUNT)


class Obstacle(GridObject):
    """
    An obstacle blocks the view of agents and is mainly placed at intersections.
    """

    def __init__(self, config={}):
        if "tiles" not in config:
            config["tiles"] = [[ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)],
                               [ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)],
                               [ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)],
                               [ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)],
                               [ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)],
                               [ATiles.OBSTACLE for i in range(ROAD_TILE_COUNT)]]
        super().__init__(config)

        self.min_x = 0
        self.min_y = 0
        self.max_x = 12
        self.max_y = 12
        self.local_obstacle_rect = box(self.min_x, self.min_y, self.max_x, self.max_y)
        self.global_obstacle_rect = None

    def is_obstacle(self):
        return True

    def initialize_map_coords(self, grid_x, grid_y):
        super().initialize_map_coords(grid_x, grid_y)
        road_size_meters = TILE_SIZE_METERS * ROAD_TILE_COUNT
        self.min_x += grid_x * road_size_meters
        self.min_y += grid_y * road_size_meters
        self.max_x += grid_x * road_size_meters
        self.max_y += grid_y * road_size_meters
        self.global_obstacle_rect = box(self.min_x, self.min_y, self.max_x, self.max_y)


class Road(GridObject):
    """
    The superclass for all roads. A road is made out of smaller Tiles, which specify an occupancy grid.
    """

    def __init__(self, config={}):
        self.right_of_way = []  # Right of way priority list
        self.lanes = {}
        self.lanes_exit = {}

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        if "tiles" not in config:
            config["tiles"] = [
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F]]
        super().__init__(config)

        self.lanes_list = []

        list_lanes = [Lane([(8, 12), (8, 6), (8, 0)], lane_side=LaneSides.FORWARD, config=config),
                      Lane([(4, 0), (4, 6), (4, 12)], lane_side=LaneSides.BACKWARD, config=config)]
        self.update_lanes(list_lanes)

        self.passing_cars = set()
        self.crossing_pedestrians = []

        self.lane_separator = get_oriented_line(LineString([(6, 0), (6, 12)]), flip_ud=self.flip_ud,
                                                flip_lr=self.flip_lr, rot90=self.rot90)
        lane_markings_list = [[(6, 1), (6, 3)], [(6, 5), (6, 7)], [(6, 9), (6, 11)]]
        self.lane_markings = self.get_oriented_lines(lane_markings_list)

    def clean_up(self):
        super().clean_up()
        del self.passing_cars
        del self.crossing_pedestrians
        del self.lane_markings
        del self.lane_separator
        del self.lanes_exit
        del self.lanes
        del self.intersecting_lanes
        for lane in self.lanes_list:
            lane.clean_up()
            del lane
        del self.lanes_list

    def is_lane_on_road(self, check_lane):
        """
        Checks if a given lane is on this road.

        :param check_lane: The lane to check for
        :return: Is this lane on this road?
        """
        return check_lane in self.lanes_list

    def is_agent_on_road(self):
        """
        :return: Is an agent on this road?
        """
        for lane in self.lanes_list:
            for car in lane.car_queue:
                if car.is_agent() or car.is_simulation_agent():
                    return True
        return False

    def is_agent_upcoming(self):
        """
        :return: Will an agent be soon on this road?
        """
        for lane in self.lanes_list:
            for car in lane.upcoming_car_queue:
                if car.is_agent():
                    return True
        return False

    def get_cars_on_road(self):
        """
        Returns all cars on this road.

        :return: A list of cars on this road
        """
        cars = []
        for lane in self.lanes_list:
            for car in lane.car_queue:
                cars.append(car)
        return cars

    def get_nearby_crossing_pedestrians(self, grid, offset=1):
        """
        Returns all nearby pedestrians, which are crossing the road.

        :param grid: The grid array
        :param offset: An offset measured in grid objects to get a nearby grid array
        :return: A list of nearby pedestrians, which are crossing the road
        """
        nearby_pedestrians = []

        offset_x = offset
        offset_y = offset

        nearby_roads = get_nearby_grid(grid, self.grid_x, self.grid_y, offset_x, offset_y)

        for index, nearby_road in np.ndenumerate(nearby_roads):
            if nearby_road.is_road():
                for pedestrian in nearby_road.crossing_pedestrians:
                    nearby_pedestrians.append(pedestrian)

        return nearby_pedestrians

    def get_nearby_cars(self, grid, offset=1):
        """
        Returns all nearby cars.

        :param grid: The grid array
        :param offset: An offset measured in grid objects to get a nearby grid array
        :return: A list of nearby cars
        """

        nearby_cars = []

        offset_x = offset
        offset_y = offset

        nearby_roads = get_nearby_grid(grid, self.grid_x, self.grid_y, offset_x, offset_y)

        for index, nearby_road in np.ndenumerate(nearby_roads):
            if nearby_road.is_road():
                for lane in nearby_road.lanes_list:
                    for car in lane.car_queue:
                        nearby_cars.append(car)

        return nearby_cars

    def initialize_map_coords(self, grid_x, grid_y):
        super().initialize_map_coords(grid_x, grid_y)

        for entries in self.lanes.values():
            for lane in entries:
                lane.initialize_map_coords(grid_x, grid_y)

    def get_oriented_lines(self, line_list):
        """
        Returns the oriented lines.

        :param line_list: The lines as a list
        :return: Oriented lines
        """
        lines = []
        for line in line_list:
            lines.append(get_oriented_line(LineString(line), flip_ud=self.flip_ud, flip_lr=self.flip_lr,
                                           rot90=self.rot90))
        return lines

    def register_crossing_pedestrian(self, pedestrian):
        """
        Register a crossing pedestrian.

        :param pedestrian: The pedestrian to register
        """
        self.crossing_pedestrians.append(pedestrian)

    def unregister_crossing_pedestrian(self, pedestrian):
        """
        Unregister a crossing pedestrian.

        :param pedestrian: The pedestrian to unregister
        """
        self.crossing_pedestrians.remove(pedestrian)

    def update_lanes(self, list_lanes):
        """
        Sorts the given lanes in dictionaries by specific attributes.

        :param list_lanes: The lanes in a list
        """
        # The attribute lanes contains all lanes sorted by their entry point. The four possible entry points
        # are the four different directions
        self.lanes = self.get_sorted_lanes(list_lanes)
        self.lanes_exit = self.get_sorted_lanes(list_lanes, exit=True)
        self.lanes_list = list_lanes

        self.intersecting_lanes = self.get_intersecting_lanes(list_lanes)

    @staticmethod
    def get_intersecting_lanes(list_lanes):
        """
        Sorts the list of lanes by intersecting lanes.

        :param list_lanes: The lanes, which should be sorted
        :return: The dictionary with the intersecting lanes
        """
        intersecting_lanes = {}
        for check_lane in list_lanes:
            intersecting_lanes[check_lane] = []
            for lane in list_lanes:
                if check_lane != lane:
                    if check_lane.local_line.intersects(lane.local_line):
                        intersecting_lanes[check_lane].append(lane)
        return intersecting_lanes

    def is_road(self):
        return True

    @staticmethod
    def get_sorted_lanes(list_lanes: list, exit=False) -> dict:
        """
        This returns the given lanes in a dictionary sorted by their entry point. This allows easy access to the next lane, when a car moves on to a new road.

        :param list_lanes: The lanes which should be sorted
        :return: The sorted lanes dictionary
        """
        lanes = {}
        for lane in list_lanes:
            if exit:
                direction = lane.exit_direction
            else:
                direction = lane.entry_direction
            if direction in lanes:
                lanes[direction].append(lane)
            else:
                lanes[direction] = [lane]
        return lanes


class Intersection(Road):
    """
    The superclass for intersections.
    """

    def __init__(self, config={}):
        if "tiles" not in config:
            config["tiles"] = [
                [ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F],
                [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F],
                [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F],
                [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B],
                [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B],
                [ATiles.WALL_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.WALL_B]]
        super().__init__(config)

        list_lanes = self.get_inter_lanes(config)

        self.update_lanes(list_lanes)

        self.local_inter_start_lines = self.get_oriented_lines(
            [[(1, 6), (1, 10)], [(6, 11), (10, 11)], [(11, 2), (11, 6)]])
        self.local_inter_end_lines = self.get_oriented_lines(
            [[(1, 2), (1, 6)], [(2, 11), (6, 11)], [(11, 6), (11, 10)]])
        self.lane_separator = None
        self.lane_markings = []

        self.cars_by_direction = {}

    def clean_up(self):
        super().clean_up()
        del self.cars_by_direction

    def initialize_map_coords(self, grid_x, grid_y):
        super().initialize_map_coords(grid_x, grid_y)

        self.global_inter_start_lines = []
        for line in self.local_inter_start_lines:
            self.global_inter_start_lines.append(get_global_geometry(line, grid_x, grid_y))

        self.global_inter_end_lines = []
        for line in self.local_inter_end_lines:
            self.global_inter_end_lines.append(get_global_geometry(line, grid_x, grid_y))

    def is_inter(self):
        return True

    def is_agent_waiting(self):
        """
        Checks if an agent is waiting at this intersection.

        :return: Is an agent waiting at this intersection?
        """
        for waiting_car in self.cars_by_direction.values():
            if waiting_car.is_agent() or waiting_car.is_simulation_agent():
                return True
        return False

    @staticmethod
    def get_inter_lanes(config):
        return [Lane([(8, 12), (8, 4), (0, 4)], lane_side=LaneSides.BACKWARD, config=config),
                Lane([(8, 12), (8, 8), (12, 8)], lane_side=LaneSides.BACKWARD, config=config),
                Lane([(0, 8), (4, 8), (4, 12)], lane_side=LaneSides.BACKWARD, config=config),
                Lane([(0, 8), (6, 8), (12, 8)], lane_side=LaneSides.BACKWARD, config=config),
                Lane([(12, 4), (6, 4), (0, 4)], lane_side=LaneSides.FORWARD, config=config),
                Lane([(12, 4), (4, 4), (4, 12)], lane_side=LaneSides.FORWARD, config=config)
                ]

    def calculate_recursive_waiting_cars(self, lane, entry_list, distance, grid, use_worst_case=True):
        """
        Recursively calculates waiting cars at the intersection and adds them to the given entry list.

        :param lane: The lane, where to start from
        :param entry_list: The entry list, which should be populated with waiting cars
        :param distance: The current distance
        :param grid: The grid array
        :param use_worst_case: If the worst case should be used to calculate the ttc
        """
        previous_lanes = lane.get_previous_lanes(grid)

        for previous_lane in previous_lanes:
            lane_length = previous_lane.local_line.length
            car = previous_lane.get_last_car()
            if car is not None:
                car_distance = max(distance + lane_length - car.x, 0)
                car_velocity = car.v

                if car_velocity > 0:
                    ttc = float(car_distance) / float(car_velocity)
                else:
                    if use_worst_case:
                        # What is the worst case ttc. The fastest ttc possible.
                        tuple = get_velocity_time_way_acceleration(s=car_distance, a=3, t=None, v_0=0)
                        ttc = max(tuple[0], tuple[1])
                    else:
                        ttc = 1000

                entry_list.append((car, distance, ttc))
            else:
                new_distance = distance + lane_length
                road = previous_lane.get_road(grid)
                if new_distance <= 36.0 and not road.is_inter():
                    self.calculate_recursive_waiting_cars(previous_lane, entry_list, new_distance, grid)

    def update_right_of_way(self, grid):
        """
        Determines, which car is allowed to pass the intersection.

        :param grid: The road array
        """

        # Berechne mit dem Weg-Zeit-Gesetz die Zeit, die ein Auto zum überqueren braucht
        # Annahme Kreuzung: 15m, Beschleuning 3 m/s^2, Anfangsgeschwindigkeit: 0 m/s
        # = 3,1623 s um Kreuzung zu überqueren
        # Alle Autos, die in weniger als 4 Sekunden an der Kreuzung eintreffen werden
        # als eintreffende Fahrzeuge gewertet

        self.cars_by_direction = {}

        for direction in self.right_of_way:  # First we need to get every waiting car from each direction of the intersection
            if direction in self.lanes:
                waiting_cars = []
                entry_lane = self.lanes[direction][0]

                self.calculate_recursive_waiting_cars(entry_lane, waiting_cars, 0, grid)

                # The cars which pass the thresholds to be considered a waiting car
                threshold_waiting_cars = []

                for car_entry in waiting_cars:
                    car, distance, ttc = car_entry
                    if car not in self.passing_cars:
                        # if car.is_agent():
                        #    print("TTC ",ttc, "Distance ",distance)
                        if ttc < 5.0 or distance < 6.0:
                            threshold_waiting_cars.append((car, distance, ttc))

                if len(threshold_waiting_cars):
                    self.cars_by_direction[direction] = min(threshold_waiting_cars, key=lambda t: t[2])[0]

        right_of_way_car = None  # This is the car, which currently has the right of way
        right_of_way_index = None  # This is the index of the direction of the car, which currently has the right of way
        for index, direction in enumerate(self.right_of_way):  # Check each direction in the order of right of way
            if direction in self.cars_by_direction:  # and not cars_by_direction[direction].is_agent():
                car = self.cars_by_direction[direction]

                if car.is_agent():
                    if not self.is_car_intersecting_with_waiting_car(car, self.cars_by_direction):
                        self.passing_cars.add(car)
                    else:
                        if right_of_way_car is not None:
                            if (
                                    car.route.next_inter_turn is not Turn.LEFT or right_of_way_car.route.next_inter_turn is Turn.LEFT) or (
                                    right_of_way_index == index - 1):
                                right_of_way_car = car
                                right_of_way_index = index
                        else:
                            right_of_way_car = car
                            right_of_way_index = index
                else:
                    # Car is not an agent. Here we also need to check if the car fits into the intersection
                    if not self.is_car_intersecting_with_waiting_car(car, self.cars_by_direction) \
                            and not self.is_car_intersecting_with_passing_car(car):
                        # This is the case, where the car does not interrupt waiting cars, nor passing cars and
                        # where the car also fits into the intersection
                        self.passing_cars.add(car)
                    else:
                        if right_of_way_car is not None:  # If no car has been found yet:

                            # If the car is not turning left, overwrite the current right of way car (Rechts Vor Links)
                            # Or if the right of way car is turning left, overwrite the right of way car (Vorrang)
                            # Or if the right of way car is to the left of the car, overwrite the right of way car (Rechts vor Links)
                            if (
                                    car.route.next_inter_turn is not Turn.LEFT or right_of_way_car.route.next_inter_turn is Turn.LEFT) or (
                                    right_of_way_index == index - 1):
                                right_of_way_car = car
                                right_of_way_index = index
                        else:
                            right_of_way_car = car
                            right_of_way_index = index
        if right_of_way_car is not None:
            if right_of_way_car.is_agent():
                self.passing_cars.add(right_of_way_car)
            else:
                if not self.is_car_intersecting_with_passing_car(right_of_way_car):
                    self.passing_cars.add(right_of_way_car)

    def get_car_with_priority(self, first_car, second_car, grid):
        """
        For two given cars this method checks which car has the priority.

        :param first_car: The first car
        :param second_car: The second car
        :param grid: The grid map
        :return: The car, which has the priority
        """

        car_list = [first_car, second_car]
        sorted_cars = []
        for car_index, car in enumerate(car_list):
            if car.current_lane.get_road(grid) is self:
                entry_lane = car.current_lane
            else:
                entry_lane = car.route.next_inter_lane
            if entry_lane not in self.lanes_list:
                return car_list[(car_index + 1) % len(car_list)]

            direction = entry_lane.entry_direction
            direction_index = self.right_of_way.index(direction)

            sorted_cars.append((direction_index, car, entry_lane))
        sorted_cars.sort(key=lambda t: t[0])

        right_of_way_car = None
        right_of_way_index = None
        right_of_way_turn = None
        for direction_index, car, entry_lane in sorted_cars:
            car_turn = entry_lane.turn
            if right_of_way_car is not None:
                if (car_turn is not Turn.LEFT or right_of_way_turn is Turn.LEFT) or (
                        right_of_way_index == direction_index - 1):
                    right_of_way_car = car
                    right_of_way_index = direction_index
                    right_of_way_turn = car_turn
            else:
                right_of_way_car = car
                right_of_way_index = direction_index
                right_of_way_turn = car_turn

        return right_of_way_car

    @staticmethod
    def can_fit(car):
        """
        Checks if a vehicle can fit, when driving trough the intersection.

        :param car: The car which should be checked
        :return: Can the car fit?
        """
        inter_distance = 0
        if car.inter_distance:
            inter_distance = car.inter_distance

        if car.leader is not None:
            if car.leader.free_road_length is not None:
                if car.leader_distance + car.leader.free_road_length < inter_distance + car.route.next_lane.local_line.length + car.length * 3:
                    return False
            else:
                return True
        return True

    def is_car_intersecting_with_waiting_car(self, car_entering, cars_by_direction):
        """
        Checks if the query car is intersecting with a lane of another waiting car.

        :param car_entering: The query car
        :param cars_by_direction: The cars sorted by entry direction
        :return: Is the car intersecting with another car?
        """
        entering_lane = car_entering.route.next_inter_lane

        for car in cars_by_direction.values():
            # Car is is driving towards the road
            if car is not car_entering:
                lane = car.route.next_inter_lane
                if entering_lane in self.intersecting_lanes.keys():
                    if lane in self.intersecting_lanes[entering_lane]:
                        return True
        return False

    def is_agent_violating_traffic_rules(self, agent, grid):
        if agent.current_lane.get_road(grid) is not self:
            return False

        if agent not in self.passing_cars:
            return True
        else:
            for car in self.passing_cars:
                if car != agent and car.current_lane.get_road(
                        grid) is not self:  # This car is now driving towards the intersection
                    other_entering_lane = car.route.next_inter_lane
                    if other_entering_lane in self.intersecting_lanes.keys():
                        if agent.current_lane in self.intersecting_lanes[other_entering_lane]:
                            return True

            return False

    def is_car_intersecting_with_passing_car(self, car_entering):
        """
        Checks if the query car is intersecting with a lane of a passing car on the intersection.

        :param car_entering: The query car
        :return: Is the car intersecting with a passing car?
        """
        entering_lane = car_entering.route.next_inter_lane

        for car in self.passing_cars:
            if car is not car_entering:
                lane = None
                if car.current_lane.grid_x == self.grid_x and car.current_lane.grid_y == self.grid_y:
                    # Car is currently on this road
                    lane = car.current_lane
                else:
                    # Car is is driving towards the road
                    lane = car.route.next_inter_lane

                if entering_lane in self.intersecting_lanes.keys():
                    if lane in self.intersecting_lanes[entering_lane]:
                        return True
        return False

    def car_passed(self, car):
        """
        If a car passed the intersection, remove it from the list.

        :param car: The car, which passed the intersection
        """
        if car in self.passing_cars:
            self.passing_cars.remove(car)


class Cross(Intersection):
    """
    The class for intersections with entrances on all four sides.
    """

    def __init__(self, config={}):
        config["tiles"] = [[ATiles.WALL_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.WALL_F],
                           [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F],
                           [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F],
                           [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B],
                           [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B],
                           [ATiles.WALL_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.WALL_B]]
        super().__init__(config)

        list_lanes = self.get_inter_lanes(config)
        additional_lanes = [Lane([(8, 12), (8, 6), (8, 0)], lane_side=LaneSides.BACKWARD, config=config),
                            Lane([(0, 8), (8, 8), (8, 0)], lane_side=LaneSides.BACKWARD, config=config),
                            Lane([(12, 4), (8, 4), (8, 0)], lane_side=LaneSides.BACKWARD, config=config),
                            Lane([(4, 0), (4, 4), (0, 4)], lane_side=LaneSides.FORWARD, config=config),
                            Lane([(4, 0), (4, 6), (4, 12)], lane_side=LaneSides.FORWARD, config=config),
                            Lane([(4, 0), (4, 8), (12, 8)], lane_side=LaneSides.FORWARD, config=config)
                            ]
        list_lanes.extend(additional_lanes)

        self.update_lanes(list_lanes)

        self.local_inter_start_lines = self.get_oriented_lines(
            [[(1, 6), (1, 10)], [(6, 11), (10, 11)], [(11, 2), (11, 6)], [(2, 1), (6, 1)]])
        self.local_inter_end_lines = self.get_oriented_lines(
            [[(1, 2), (1, 6)], [(2, 11), (6, 11)], [(11, 6), (11, 10)], [(6, 1), (10, 1)]])

        self.lane_separator = None

    def is_cross(self):
        return True


class Corner(Road):
    """
    The class for corner roads.
    """

    def __init__(self, config={}):
        config["tiles"] = [[ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F, ATiles.WALL_F],
                           [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                           [ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                           [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                           [ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F],
                           [ATiles.WALL_B, ATiles.ROAD_B, ATiles.ROAD_B, ATiles.ROAD_F, ATiles.ROAD_F, ATiles.WALL_F]]
        super().__init__(config)

        list_lanes = [Lane([(8, 12), (8, 4), (0, 4)], lane_side=LaneSides.FORWARD, config=config),
                      Lane([(0, 8), (4, 8), (4, 12)], lane_side=LaneSides.BACKWARD, config=config)]

        self.update_lanes(list_lanes)

        self.lane_separator = get_oriented_line(LineString([(0, 6), (6, 6), (6, 12)]), flip_ud=self.flip_ud,
                                                flip_lr=self.flip_lr, rot90=self.rot90)
        lane_markings_list = [[(1, 6), (3, 6)], [(5, 6), (6, 6), (6, 7)], [(6, 9), (6, 11)]]
        self.lane_markings = self.get_oriented_lines(lane_markings_list)

    def is_corner(self):
        return True


class ARoads:
    """
    A class, which holds all types of abstract grid objects.
    """

    OBSTACLE = Obstacle(config={"name": "obstacle", "symbol": "x "})
    NOTHING = GridObject(config={"name": "nothing", "symbol": ". ", "fill": ATiles.NOTHING})
    BORDER = GridObject(config={"name": "border", "symbol": ". ", "fill": ATiles.BORDER})
    INNER_BORDER = GridObject(config={"name": "inner_border", "symbol": ". ", "fill": ATiles.INNER_BORDER})
    BOTTOM_LEFT_UP = Corner(config={"name": "bottom_left_up", "symbol": "⌞↑", "rot90": 2, "inversed": True})
    BOTTOM_LEFT_DOWN = Corner(config={"name": "bottom_left_down", "symbol": "↓⌞", "rot90": 2, "inversed": False})
    BOTTOM_RIGHT_UP = Corner(config={"name": "bottom_right_up", "symbol": "⌟↑", "rot90": 1, "inversed": False})
    BOTTOM_RIGHT_DOWN = Corner(config={"name": "bottom_right_down", "symbol": "↓⌟", "rot90": 1, "inversed": True})
    TOP_LEFT_DOWN = Corner(config={"name": "top_left_down", "symbol": "↓⌜", "rot90": 3, "inversed": False})
    TOP_LEFT_UP = Corner(config={"name": "top_left_up", "symbol": "⌜↑", "rot90": 3, "inversed": True})
    TOP_RIGHT_DOWN = Corner(config={"name": "top_right_down", "symbol": "↓⌝", "rot90": 0, "inversed": True})
    TOP_RIGHT_UP = Corner(config={"name": "top_right_up", "symbol": "⌝↑", "rot90": 0, "inversed": False})
    UP = Road(config={"name": "up", "symbol": "↑ ", "rot90": 0, "inversed": False})
    LEFT = Road(config={"name": "left", "symbol": "← ", "rot90": 3, "inversed": False})
    DOWN = Road(config={"name": "down", "symbol": "↓ ", "rot90": 2, "inversed": False})
    RIGHT = Road(config={"name": "right", "symbol": "→ ", "rot90": 1, "inversed": False})

    INTER_UP = Intersection(config={"name": "inter_up", "symbol": "╦ ", "rot90": 0, "inversed": False,
                                    "right_of_way": [Directions.LEFT, Directions.DOWN, Directions.RIGHT]})
    INTER_UP_F = Intersection(config={"name": "inter_up_f", "symbol": "╦f", "rot90": 0, "inversed": True,
                                      "right_of_way": [Directions.LEFT, Directions.DOWN, Directions.RIGHT]})
    CROSS_UP_F = Cross(config={"name": "cross_up_f", "symbol": "✙f", "rot90": 0, "inversed": True,
                               "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP, Directions.LEFT]})

    INTER_LEFT = Intersection(config={"name": "inter_left", "symbol": "╠ ", "rot90": 3, "inversed": False,
                                      "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP]})
    INTER_LEFT_F = Intersection(config={"name": "inter_left_f", "symbol": "╠f", "rot90": 3, "inversed": True,
                                        "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP]})
    CROSS_LEFT_F = Cross(config={"name": "cross_left_f", "symbol": "✙f", "rot90": 3, "inversed": True,
                                 "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP, Directions.LEFT]})

    INTER_DOWN = Intersection(config={"name": "inter_down", "symbol": "╩ ", "rot90": 2, "inversed": False,
                                      "right_of_way": [Directions.RIGHT, Directions.UP, Directions.LEFT]})
    INTER_DOWN_F = Intersection(config={"name": "inter_down_f", "symbol": "╩f", "rot90": 2, "inversed": True,
                                        "right_of_way": [Directions.RIGHT, Directions.UP, Directions.LEFT]})
    CROSS_DOWN_F = Cross(config={"name": "cross_down_f", "symbol": "✙f", "rot90": 2, "inversed": True,
                                 "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP, Directions.LEFT]})

    INTER_RIGHT = Intersection(config={"name": "inter_right", "symbol": "╣ ", "rot90": 1, "inversed": False,
                                       "right_of_way": [Directions.UP, Directions.LEFT, Directions.DOWN]})
    INTER_RIGHT_F = Intersection(config={"name": "inter_right_f", "symbol": "╣f", "rot90": 1, "inversed": True,
                                         "right_of_way": [Directions.UP, Directions.LEFT, Directions.DOWN]})
    CROSS_RIGHT_F = Cross(config={"name": "cross_right_f", "symbol": "✙f", "rot90": 1, "inversed": True,
                                  "right_of_way": [Directions.DOWN, Directions.RIGHT, Directions.UP, Directions.LEFT]})


if __name__ == '__main__':
    road = Road(config={"flip_lr": True})
    symbols = np.array([[y.id for y in x] for x in road.tiles])
    print(np.flipud(np.rot90(symbols, k=1)))

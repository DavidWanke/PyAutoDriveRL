from .utils import *
from shapely.geometry import box
import numpy as np
from src.env.model.simulation.config import Config

CONFIG = Config()
rng = np.random.default_rng()


def set_seed(seed):
    global rng
    rng = np.random.default_rng(seed)


def set_config(config):
    global CONFIG
    CONFIG = config


class Pedestrian():
    """
    The superclass for Pedestrians. Pedestrians walk at the sides of roads and can eventually cross roads.
    """

    class States:
        WALK = 0
        CROSS = 1

    def __init__(self, start_road, grid):
        self.v = get_meters_per_second(kilometers_per_hour=(6.0 * self.get_rand_factor()))

        self.size = 1
        self.grid = grid

        self.lookahead = 50

        self.current_road = start_road
        self.grid_pos = np.array([start_road.grid_x, start_road.grid_y])

        self.direction_offset = (0, 0)

        self.cross_collision_area = None
        self.cross_line = None
        self.route_line = None
        self.route_look_back = 3

        self.state = self.States.WALK

        self.cross_timer = rng.integers(5, 10, endpoint=True)

        self.last_tile = None

        self.pos = self.choose_position_on_road(start_road)
        self.rect = box(self.pos[0] - self.size / 2, self.pos[1] - self.size / 2, self.pos[0] + self.size / 2,
                        self.pos[1] + self.size / 2)

    def get_rand_factor(self, mean=1, deviation=0.1, lower_cut_off=None, upper_cut_off=None):
        """
        Returns a random factor using a normal distribution.

        :param mean: The mean of the normal distribution
        :param deviation: The deviation of the normal distribution
        :param lower_cut_off: The minimal allowed value
        :param upper_cut_off: The maximal allowed value
        :return: A random factor
        """

        factor = rng.normal(mean, deviation)
        factor = np.clip(factor, 0.5, 1.5)

        if lower_cut_off:
            factor = max(lower_cut_off, factor)
        if upper_cut_off:
            factor = min(upper_cut_off, factor)

        return factor

    def clean_up(self):
        """
        Deletes unused objects.
        """
        del self.current_road
        del self.grid
        del self.last_tile

    def choose_position_on_road(self, road):
        """
        Returns a spawn position for a pedestrian given a road.

        :param road: The road
        :return: The position on the road
        """
        for index, tile in np.ndenumerate(road.tiles):
            if tile.is_wall:
                pos = np.array([road.grid_x, road.grid_y]) * ROAD_METER_COUNT + np.asarray(index) * TILE_SIZE_METERS

                pos += TILE_SIZE_METERS / 2
                return pos
        raise ValueError("No spawn position for pedestrian found!")

    def check_walk_state(self, grid_tiles):
        """
        Update for the WALK state. Here the pedestrian checks, if it should cross the road.

        :param grid_tiles: The grid tile array
        """
        if self.current_road.is_road() and not self.current_road.is_inter() and not self.current_road.is_corner():
            if self.cross_timer <= 0:
                ttc_thresh = CONFIG.pedestrian_ttc_factor * (12.0 / self.v)

                if self.get_approximate_ttc() > ttc_thresh:
                    self.direction_offset = self.get_initial_cross_direction(grid_tiles)

                    nearby_cars = self.current_road.get_nearby_cars(self.grid)
                    area_distance = 6.0
                    self.cross_line = LineString([tuple(self.pos),
                                                  tuple(self.pos + np.asarray(self.direction_offset) * (
                                                          ROAD_METER_COUNT - 2))])
                    self.cross_collision_area = self.cross_line.buffer(area_distance, cap_style=2)
                    should_cross = True
                    for car in nearby_cars:
                        car_rect, car_angle = car.get_info()
                        if self.cross_collision_area.intersects(car_rect):
                            should_cross = False

                    if should_cross:
                        self.state = self.States.CROSS
                        self.cross_timer = rng.integers(20, 40, endpoint=True)
                        self.current_road.register_crossing_pedestrian(self)
                    else:
                        self.cross_timer = rng.integers(1, 3, endpoint=True)
                        self.cross_collision_area = None
                else:
                    self.cross_timer = rng.integers(1, 3, endpoint=True)

    def switch_states(self, grid_tiles, current_tile):
        """
        Here the pedestrian checks, if it should switch states.

        :param grid_tiles: The grid tile array
        :param current_tile: The current tile, that the Pedestrian is standing on
        """
        # Switch between States
        if self.state is self.States.WALK:
            self.check_walk_state(grid_tiles)
        if self.state is self.States.CROSS:
            if self.last_tile is not None and not self.last_tile.is_wall and current_tile.is_wall:
                self.state = self.States.WALK
                self.cross_collision_area = None
                self.current_road.unregister_crossing_pedestrian(self)

    def update(self, grid_tiles):
        """
        This should be called at every step in the simulation and will update the state of the pedestrian.

        :param grid_tiles: The grid tile array
        """
        check_pos = self.pos - np.asarray(self.direction_offset) * TILE_SIZE_METERS / 2
        current_tile = get_tile_from_pos(grid_tiles, check_pos[0], check_pos[1])
        self.cross_timer -= CONFIG.step_length

        self.switch_states(grid_tiles, current_tile)

        # Do actions according to current state
        if self.state is self.States.WALK:
            self.direction_offset = self.get_sidewalk_direction(grid_tiles)

        if self.direction_offset is not None:
            self.pos = self.pos + np.asarray(self.direction_offset) * self.v * CONFIG.step_length
            self.rect = box(self.pos[0] - self.size / 2, self.pos[1] - self.size / 2, self.pos[0] + self.size / 2,
                            self.pos[1] + self.size / 2)

            look_back = self.route_look_back
            look_ahead = 8
            self.route_line = LineString([self.pos - np.asarray(self.direction_offset) * look_back,
                                          self.pos + np.asarray(self.direction_offset) * look_ahead])

        self.grid_pos = (self.pos / ROAD_METER_COUNT).astype(int)
        self.current_road = self.grid[self.grid_pos[0], self.grid_pos[1]]
        self.last_tile = current_tile

    def get_approximate_ttc(self):
        """
        This method will calculate the TTC with cars on the road. This is done under the assumption, that the pedestrian would cross the road in this exact moment.

        :return: The smallest TTC with other cars
        """
        ttc_list = []

        if len(self.current_road.lanes_list) != 2:
            raise ValueError("There must be two lanes on this straight road!")

        lane1 = self.current_road.lanes_list[0]
        lane2 = self.current_road.lanes_list[1]

        # First check the current two lanes
        for lane in [lane1, lane2]:
            lane_pos = lane.global_line.project(Point(self.pos))
            for car in lane.car_queue:
                if car.x <= lane_pos + car.length + 1:
                    if car.v > 0:
                        ttc = float(lane_pos - car.x) / float(car.v)
                        return ttc
                    else:
                        return 1000
            # Then recursively check previous lanes
            self.calculate_recursive_ttc(lane, ttc_list, lane_pos)

        if len(ttc_list) > 0:
            return min(ttc_list)
        else:
            return 1000

    def calculate_recursive_ttc(self, lane, ttc_list, distance):
        """
        This method recursively calculates the smallest TTC given a lane and a distance.

        :param lane: The lane, where the calculation starts
        :param ttc_list: A list of already calculated TTC values
        :param distance: The starting distance
        :return: The smallest TTC
        """
        previous_lanes = lane.get_previous_lanes(self.grid)

        for previous_lane in previous_lanes:
            lane_length = previous_lane.local_line.length
            car = previous_lane.get_last_car()
            if car is not None:
                car_distance = distance + lane_length - car.x
                car_velocity = car.v

                if car_velocity > 0:
                    ttc_list.append(float(car_distance) / float(car_velocity))
                else:
                    ttc_list.append(1000)
            else:
                new_distance = distance + lane_length
                if new_distance < self.lookahead:
                    self.calculate_recursive_ttc(previous_lane, ttc_list, new_distance)

    def get_initial_cross_direction(self, grid_tiles):
        """
        Returns a random direction, where the pedestrian can cross the road.

        :param grid_tiles: The grid tile array
        :return: A random cross direction
        """
        check_pos = self.pos
        tile_index = get_tile_index_from_pos(grid_tiles, check_pos[0], check_pos[1])

        possible_directions = []

        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for offset in offsets:
            x, y = tile_index
            x += offset[0]
            y += offset[1]
            if 0 <= x < grid_tiles.shape[0] and 0 <= y < grid_tiles.shape[1]:
                if grid_tiles[x, y].is_road:
                    possible_directions.append(offset)

        if len(possible_directions) > 0:
            return tuple(rng.choice(possible_directions))
        else:
            return None

    def get_sidewalk_direction(self, grid_tiles):
        """
        Returns a random walking direction for the sidewalk.

        :param grid_tiles: The grid tile array
        :return: A random walking direction
        """
        check_pos = self.pos - np.asarray(self.direction_offset) * TILE_SIZE_METERS / 2
        tile_index = get_tile_index_from_pos(grid_tiles, check_pos[0], check_pos[1])

        possible_directions = []

        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for offset in offsets:
            x, y = tile_index
            x += offset[0]
            y += offset[1]
            if 0 <= x < grid_tiles.shape[0] and 0 <= y < grid_tiles.shape[1]:
                if grid_tiles[x, y].is_wall:
                    possible_directions.append(offset)

        if self.direction_offset in possible_directions:
            return self.direction_offset
        else:
            current_opposite = Directions.get_opposite_offset(self.direction_offset)
            if len(possible_directions) > 1:
                if current_opposite in possible_directions:
                    possible_directions.remove(current_opposite)
            return tuple(rng.choice(possible_directions))

import gym
import numpy as np
import shapely.affinity
from shapely.geometry import LineString, box
from .utils import *
from src.env.model.simulation.config import Config

CONFIG = Config()


def set_config(config):
    global CONFIG
    CONFIG = config


class StateRepresentation:
    """
    This is an interface, which can be used for different state representations for the agent.
    """

    def __init__(self, agent):
        self.observation_space = None
        self.agent = agent

    def get_observation(self):
        """
        Gets the current observation.

        :return: The observation as an array
        """
        raise NotImplementedError("This method is not implemented and needs to be overriden by a StateRepresentation!")

    def clean_up(self):
        """
        Cleans up unused objects.
        """
        del self.agent
        del self.observation_space


class Patch:
    """
    Defines a patch for the invariant environment representation for the agent.
    """

    def __init__(self, polygon, width, length, t_max, s_start_ego, coming_from_ego):
        """
        :param polygon: The geometric representation of the patch
        :param width: The width of the patch
        :param length: The length of the patch
        :param t_max: Maximum arrival time in seconds used for normalization
        :param s_start_ego: The distance from the front of the agent to this patch
        :param coming_from_ego: The direction where the agent is coming from when driving towards this patch
        """
        self.tto_other_next = 0
        self.tto_ego = 0
        self.ttv_ego = 0
        self.intersection = None
        self.priority = None
        self.car_data = {Directions.DOWN: [], Directions.RIGHT: [], Directions.LEFT: [], Directions.UP: []}
        self.t_max = t_max

        self.tto_other = 0
        self.ttv_other = 0
        self.turn_other = Turn.STRAIGHT
        self.s_start_other = 0
        self.coming_from_other = 0

        self.relative_direction_other = AgentDirections.SAME

        self.coming_from_ego = coming_from_ego
        self.s_start_ego = s_start_ego

        self.width = width
        self.length = length

        self.polygon = polygon  # The geometric representation of the patch as a polygon

    def check_intersecting_route(self, car_route, car_velocity, car_length, route_width=0.1, look_back=0.0,
                                 turn=Turn.STRAIGHT, car_object=None, grid=None) -> bool:
        """
        Checks if the given car route intersects this patch and calculates the entries of the patch.

        :param grid: The grid array
        :param car_object: The car object
        :param turn: The turn that the car currently takes
        :param car_route: The route line, which the car is following
        :param car_velocity: The velocity of the car
        :param car_length: The length of the car
        :param look_back: This is optional and determines how many meters behind the object the route line starts. This means that s_start needs to be offseted
        :return: A Boolean specifying, if the given car route is intersecting with this patch
        """
        intersect_route = car_route.buffer(route_width / 2, cap_style=2)

        if intersect_route.intersects(self.polygon):
            # The other vehicle crosses the route of the agent
            intersecting_geometry = intersect_route.intersection(self.polygon)

            first_intersecting_geometry = None
            first_intersection = None
            if isinstance(intersecting_geometry, shapely.geometry.base.BaseMultipartGeometry):
                for geom in intersecting_geometry.geoms:
                    if not geom.is_empty:
                        first_intersecting_geometry = geom
                        break
            else:
                first_intersecting_geometry = intersecting_geometry

            if first_intersecting_geometry is not None:
                if first_intersecting_geometry.is_empty:
                    return False
                if not first_intersecting_geometry.is_valid:
                    return False
            else:
                return False

            if isinstance(first_intersecting_geometry, shapely.geometry.LineString):
                first_intersection = Point(first_intersecting_geometry.coords[0])
            elif isinstance(first_intersecting_geometry, shapely.geometry.Polygon):
                first_intersection = Point(first_intersecting_geometry.exterior.coords[0])

            distance_to_intersection = car_route.project(first_intersection)

            coming_from = None
            if car_object is not None:
                if grid is None:
                    raise ValueError("The grid needs to be included in this case!")
                patch_coords = self.polygon.exterior.coords[0]
                road = grid[get_grid_index_from_pos(grid, patch_coords[0], patch_coords[1])]
                road_lanes = road.lanes_list

                for lane in road_lanes:
                    if lane in car_object.route.route_lanes:
                        coming_from = lane.entry_direction
                        break

            if coming_from is None:
                route_point1 = car_route.interpolate(0.95 * distance_to_intersection)
                route_point2 = car_route.interpolate(0.95 * distance_to_intersection + 0.1)

                coming_from = get_compass_direction((route_point1.x, route_point1.y), (route_point2.x, route_point2.y))

            s_start = distance_to_intersection - car_length / 2
            s_end = s_start + car_length + 2

            if look_back > 0:
                s_start -= look_back

            s_start = max(0, s_start)
            s_end = max(0, s_end)

            self.add_other(s_start, s_end, car_velocity, coming_from, turn, car_object)

            return True
        return False

    def add_ego(self, agent_length, agent_v):
        """
        Updates the tto and ttv values of the ego vehicle.

        :param agent_v: The velocity of the agent
        :param agent_length: The length of the agent
        """
        self.tto_ego, self.ttv_ego = self.get_tt_ego(agent_length, agent_v)

    def get_tt_ego(self, agent_length, agent_v):
        """
        Gets the tto and ttv values of the ego vehicle.

        :param agent_v: The velocity of the agent
        :param agent_length: The length of the agent
        :return: A tuple containing tto_ego and ttv_ego
        """
        s_start = max(0, self.s_start_ego)
        s_end = s_start + agent_length + 2

        if agent_v > 0:
            tto_ego = float(s_start) / float(agent_v)
            ttv_ego = float(s_end) / float(agent_v)
        else:
            if s_start == 0:
                tto_ego = 0.0
            else:
                tto_ego = self.t_max
            ttv_ego = self.t_max

        return tto_ego, ttv_ego

    def add_other(self, s_start, s_end, car_velocity, coming_from, turn, car_object):
        """
        Adds a car to the dictionary of incoming cars of the patch.

        :param s_start: The distance from the front of the car to this patch
        :param s_end: The distance from the back of the car to this patch
        :param car_velocity: The velocity of the car
        :param coming_from: The direction, that the car is coming from, when driving towards this patch
        :param turn: The turn, that the car will take when driving towards this patch
        :param car_object: The car object itself
        """
        self.car_data[coming_from].append((s_start, s_end, car_velocity, coming_from, turn, car_object))

    def get_normalized(self, value):
        """
        Returns the normalized tt value between [0,1]. Normalization uses the t_max value.

        Booleans are handled differently than tt values. If the value is None, it will be returned as 1.0.
        If it is False, it will be 0.5 and if it is True, it will be 0.0.

        :param value: The value which should be normalized. This can be an Integer or a Boolean
        :return: The normalized value in [0,1]
        """
        if value is None:
            return 1.0

        if isinstance(value, bool):
            if value is True:
                return 0.0
            else:
                return 0.5
        else:
            return float(min(value, self.t_max)) / float(self.t_max)

    def update_values(self, grid, agent):
        """
        Calculates and updates patch entries using the incoming car data.

        :param grid: The grid array
        :param agent: The agent car object
        """

        car_data = self.car_data.copy()

        other_cars = []
        other_next_cars = []
        for direction in car_data.keys():
            data = car_data[direction].copy()

            if len(data) > 0:
                # Tuple looks like (s_start, s_end, car_velocity, coming_from, turn)
                # We only want to consider the car from one specific direction, which has the shortest way to the patch
                # -> The first car, which will drive over it

                other_car = min(data, key=lambda t: t[0])
                other_cars.append(other_car)

                data.remove(other_car)
                other_next_cars += data

        tt_intervals = []
        for cars in [other_cars, other_next_cars]:
            tt_part_intervals = []
            for car in cars:
                s_start, s_end, car_velocity, coming_from, turn, car_object = car

                if car_velocity > 0:
                    tto = float(s_start) / car_velocity
                    ttv = float(s_end) / car_velocity
                else:
                    if s_start < 3.0:  # The case, where the car stands on the patch
                        tto = 0.0
                    else:
                        tto = self.t_max
                    ttv = self.t_max

                tt_part_intervals.append([tto, ttv, s_start, coming_from, turn, car_object])

            merged_intervals = get_merged_overlapping_intervals(tt_part_intervals)

            merged_intervals = get_merged_threshold_intervals(merged_intervals, threshold=2)

            tt_intervals.append(merged_intervals)

        tt_other_intervals = tt_intervals[0]
        tt_other_next_intervals = tt_intervals[1]

        if len(tt_other_intervals) > 0:
            tto, ttv, s_start, coming_from, turn, car_object = tt_other_intervals[0]

            self.tto_other = tto
            self.ttv_other = ttv
            self.s_start_other = s_start
            self.coming_from_other = coming_from
            self.turn_other = turn

            if car_object is None:
                self.priority = True
            else:

                patch_coords = self.polygon.exterior.coords[0]
                road = grid[get_grid_index_from_pos(grid, patch_coords[0], patch_coords[1])]

                if road.is_inter():
                    if car_object in road.passing_cars or car_object.current_lane.get_previous_road(grid) is road or (car_object.current_lane.get_road(grid) is road and not car_object.is_agent()):
                        self.priority = True
                    else:
                        priority_car = road.get_car_with_priority(agent, car_object, grid)
                        if priority_car is car_object and not (agent in road.passing_cars or agent.current_lane.get_road(grid) is road):
                            self.priority = True
                        else:
                            self.priority = None
                else:
                    self.priority = True

            self.relative_direction_other = AgentDirections.get_car_direction(self.coming_from_other,
                                                                              self.coming_from_ego)
        else:
            self.tto_other = self.t_max
            self.ttv_other = self.t_max
            self.s_start_other = self.t_max

        if len(tt_other_next_intervals) > 0:
            self.tto_other_next = tt_other_next_intervals[0][0]
        else:
            self.tto_other_next = self.t_max


class IER(StateRepresentation):
    def __init__(self, agent, include_intersection):
        """
        :param agent: The agent car object
        :param include_intersection: A boolean, which specifies, if the intersection should be included in the state
        """
        super().__init__(agent)
        # Patch maximum time
        self.patch_t_max = 10.0

        self.include_intersection = include_intersection

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(CONFIG.observation_space_length,), seed=None)
        self.ier_patches = self.get_empty_patches()

    def clean_up(self):
        super().clean_up()
        del self.ier_patches

    def get_observation(self):
        state = np.zeros(self.observation_space.shape)

        if CONFIG.state_include_vehicle_info:
            state[0] = get_normalized(self.agent.route.next_inter_turn, Turn.MIN_VALUE, Turn.MAX_VALUE)
            index = 1
        else:
            index = 0
        for index1, patch in enumerate(self.ier_patches):
            values = [patch.tto_other, patch.ttv_other, patch.tto_other_next, patch.tto_ego]
            values.append(patch.priority)
            if self.include_intersection:
                values.append(patch.intersection)
            for index2, value in enumerate(values):
                state[index] = patch.get_normalized(value)
                index += 1

            if CONFIG.state_include_vehicle_info:
                distance = get_normalized(np.clip(patch.s_start_other, 0, 100), min_value=0, max_value=100)
                turn = get_normalized(patch.turn_other, min_value=Turn.MIN_VALUE, max_value=Turn.MAX_VALUE)
                direction = get_normalized(patch.relative_direction_other, min_value=AgentDirections.MIN_VALUE,
                                           max_value=AgentDirections.MAX_VALUE)

                for value in [distance, turn, direction]:
                    state[index] = value
                    index += 1

        assert index == len(state)

        return state

    def update(self):
        """
        Calculates the current state and updates the patches.
        """
        patches = self.get_empty_patches()
        agent = self.agent
        grid = self.agent.grid

        last_start_line = None
        last_end_line = None
        for patch in patches:

            # First we check for intersections
            pos_x, pos_y = patch.polygon.exterior.coords[0]
            grid_x, grid_y = get_grid_index_from_pos(agent.grid, pos_x, pos_y)
            check_roads = get_nearby_grid(agent.grid, grid_x, grid_y, 1, 1)
            for index, road in np.ndenumerate(check_roads):
                if road.is_inter():
                    for start_line in road.global_inter_start_lines:
                        if start_line is not last_start_line and start_line.intersects(patch.polygon):
                            patch.intersection = True
                            last_start_line = start_line
                    for end_line in road.global_inter_end_lines:
                        if end_line is not last_end_line and end_line.intersects(patch.polygon):
                            patch.intersection = False
                            last_end_line = end_line

            # Calculate ego TTO and TTV
            patch.add_ego(agent.length, agent.v)

        # Check for crossing pedestrians
        for route_lane in agent.route.route_lanes:
            road = route_lane.get_road(agent.grid)
            for pedestrian in road.crossing_pedestrians:
                if agent.is_pedestrian_visible(pedestrian):
                    for patch in patches:
                        intersected = patch.check_intersecting_route(pedestrian.route_line, pedestrian.v, pedestrian.size,
                                                                     pedestrian.size, look_back=pedestrian.route_look_back, grid=grid)
                        if intersected:
                            break

        # First gather all cars, which might cross the route of the agent soon
        checking_cars = []
        for route_lane in agent.route.route_lanes:
            checking_lanes = []
            checking_lanes.extend(route_lane.get_road(agent.grid).intersecting_lanes[route_lane])
            checking_lanes.append(route_lane)
            for intersecting_lane in checking_lanes:
                checking_cars.extend(intersecting_lane.car_queue)
                checking_cars.extend(intersecting_lane.upcoming_car_queue)

        # Check for every car if it crosses a specific patch of the agent
        checking_cars = list(set(checking_cars))
        for car in checking_cars:
            if car is not agent and agent.is_car_visible(car):
                additional_look_back = 1.5
                look_back = car.length / 2 + additional_look_back
                car_route = car.route.get_route_line(look_back)
                for patch in patches:
                    intersected = patch.check_intersecting_route(car_route, car.v, car.length, CAR_MAX_WIDTH + 0.25, look_back=additional_look_back,
                                                                 turn=car.route.next_inter_turn, car_object=car, grid=grid)
                    if intersected:
                        break

        # Now check for hidden lanes and calculate TTO and TTV values
        if agent.hidden_polygons is not None:
            for index, grid_object in np.ndenumerate(agent.visible_grid):
                if grid_object.is_road():
                    if grid_object.global_rect.overlaps(agent.hidden_polygons):
                        for entries in grid_object.lanes.values():
                            for hidden_lane in entries:
                                if hidden_lane.global_line.intersects(agent.hidden_polygons.boundary):
                                    self.update_ier_state_for_hidden_lane(hidden_lane, patches, grid)

        for patch in patches:
            patch.update_values(agent.grid, agent)

        self.ier_patches = patches

    def update_ier_state_for_hidden_lane(self, hidden_lane, patches, grid):
        """
        Updates the ier state for one given hidden lane. A hidden lane is a lane, which intersects with the edge of the shadows of the visibility polygon.

        :param grid: The grid array
        :param hidden_lane: The hidden lane
        :param patches: The ier patches
        """

        agent = self.agent

        # Hidden lane is a lane, which intersects at least with one point on the edge of
        # the shadows

        # Get the intersecting points with the edge of the shadows
        intersecting_points_geo = hidden_lane.global_line.intersection(
            agent.hidden_polygons.boundary)
        intersecting_points_list = []

        if isinstance(intersecting_points_geo, shapely.geometry.base.BaseMultipartGeometry):
            for point in intersecting_points_geo.geoms:
                # self.points.append((point.x, point.y))
                intersecting_points_list.append(point)
        else:
            # self.points.append((intersecting_points_geo.x, intersecting_points_geo.y))
            intersecting_points_list.append(intersecting_points_geo)

        # Check for each point found
        for point in intersecting_points_list:
            # This is the list of lanes on the imaginary route
            seen_lanes = [hidden_lane]

            # This is the list of the next possible lanes
            next_lanes = hidden_lane.get_next_lanes(agent.grid)

            # These are the two lines, which will be created, when cutting the hidden lane
            # at the point, which intersects with the shadow
            cut_hidden_lane = cut_line(hidden_lane.global_line,
                                       distance=hidden_lane.global_line.project(point))

            # The first line is the part of the hidden lane, which should be considered.
            # Normally, this is the part, which does not lie in the shadows.
            # In some cases, it is the part, which lies in the shadows, since the car
            # is driving towards the shadow.
            first_line = None
            index = 0
            for part in cut_hidden_lane:
                if index == 0:
                    if not agent.hidden_polygons.contains(Point(part.coords[0])):
                        # This line does not lay in the shadow

                        if agent.route.is_lane_on_route(hidden_lane):
                            first_line = cut_hidden_lane[1]
                            # Since the part is on the route line, the car
                            # is driving towards the shadow and the line, which
                            # lies in the shadow should be considered
                        else:
                            first_line = cut_hidden_lane[index]
                            # The line which does not lie in the shadow should be considered
                        break
                elif index == 1:
                    if not agent.hidden_polygons.contains(Point(part.coords[-1])):
                        # This line does not lay in the shadow

                        if agent.route.is_lane_on_route(hidden_lane):
                            first_line = cut_hidden_lane[0]
                            # Since the part is on the route line, the car
                            # is driving towards the shadow and the line, which
                            # lies in the shadow should be considered
                        else:
                            first_line = cut_hidden_lane[index]
                            # The line which does not lie in the shadow should be considered
                        break

                index += 1

            distance = 0
            route_coords = []

            if first_line is not None:
                route_coords.extend(first_line.coords)
                distance = first_line.length

            while len(next_lanes) == 1 and distance < 6:
                distance += next_lanes[0].global_line.length
                seen_lanes.append(next_lanes[0])
                next_lanes = next_lanes[0].get_next_lanes(agent.grid)

            for seen_lane in seen_lanes[1:]:
                route_coords.extend(seen_lane.global_line.coords)

            intersected = False
            for next_lane in next_lanes:
                # Normally there should be only one next lane. Though on intersections there are multiple possible
                # next lanes. So, we check all possibilities on the intersection.
                if not intersected:
                    coords = route_coords.copy()
                    coords.extend(next_lane.global_line.coords)

                    route = LineString(coords)
                    if route.intersects(agent.route_line):
                        # self.lines.append(route)
                        for patch in patches:
                            if not intersected:
                                intersected = patch.check_intersecting_route(route, get_meters_per_second(50.0),
                                                                             car_length=12.0,
                                                                             route_width=CAR_MAX_WIDTH + 0.25, grid=grid)

    def get_empty_patches(self):
        """
        Returns the empty lookahead patches for the invariant environment representation of the agent.

        :return: A list with all the empty lookahead patches
        """
        agent = self.agent

        patches = []
        route_line = agent.route_line
        step_size = 1  # 1 meter between patches
        lookahead = agent.route.lookahead

        old_pos = Point(route_line.coords[0])
        old_pos_list = []

        distance = 0

        check_condition = True
        while check_condition:
            patch_width = agent.width + 1

            pos = route_line.interpolate(distance)
            rect = box(pos.x - step_size / 2, pos.y - patch_width / 2, pos.x + step_size / 2, pos.y + patch_width / 2)

            angle = agent.get_angle(old_pos, pos)

            if distance > 6:
                coming_from = get_compass_direction(old_pos_list[-5], old_pos_list[-1])
            else:
                coming_from = get_compass_direction(old_pos, pos)

            polygon = shapely.affinity.rotate(rect, -angle, origin=rect.centroid)

            if distance > 1:
                patches.append(Patch(polygon, width=patch_width, length=step_size, t_max=self.patch_t_max,
                                     s_start_ego=distance - agent.look_back - agent.length / 2 - 1,
                                     coming_from_ego=coming_from))

            old_pos_list.append(old_pos)
            old_pos = pos

            distance += 1

            check_condition = distance - agent.look_back - agent.length / 2 - 1 < lookahead

        return patches

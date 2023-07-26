import math
import weakref
import numpy as np
import shapely.affinity
from shapely.geometry import LineString, Polygon, box, Point
from shapely.validation import explain_validity
from .road import Lane
import shapely.ops
from shapely.errors import TopologicalError
from src.env.model.simulation.config import Config
from src.env.model.simulation.state_representation import IER
from .utils import *

CONFIG = Config()
RNG_VEHICLE = np.random.default_rng()


def set_seed(seed):
    global RNG_VEHICLE
    RNG_VEHICLE = np.random.default_rng(seed)


def set_config(config):
    global CONFIG
    CONFIG = config


class Vehicle(object):
    """
    The abstract superclass for all vehicles.
    """

    def __init__(self, start_lane, grid, config={}):
        """
        :param start_lane: The lane, which the vehicle spawns on
        :param grid: The array, which contains the road grid
        :param config: Optional configuration for the vehicle
        """
        self.current_lane = start_lane
        self.current_lane.register_car(self)

        self.x = 0.0  # The 1D position on the current lane
        self.v = 0.0  # The 1D velocity of the vehicle
        self.a = 0.0  # The 1D acceleration of the vehicle
        self.length = 4
        self.width = 2
        self.grid = grid  # The road grid
        self.route = Route(self, self.grid, CONFIG.default_route_lookahead)  # The pre-calculated route of the vehicle

        self.stop = False

        self.old_angle = 0
        self.angle = 0  # The heading angle

        self.old_pos = Point(0, 0)
        self.pos = Point(0, 0)  # The 2D position in the world environment
        self.direction_vector = np.array([0, 0])

        # The last polygon, which was determined for this vehicle.
        # This was not necessarily determined in the last step.
        self.old_rect = box(0, 0, self.length, self.width)

        # The route line, which is used for geometric analysis
        self.route_line = LineString([(0, 0), (1, 0)])

        # Leader Info
        self.leader = None
        self.leader_distance = None
        self.inter_distance = None
        self.pedestrian_distance = None

        # The wait time at intersections
        self.wait_time = 0

        self.current_info = None

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def is_agent(self) -> bool:
        """
        Checks if this vehicle is an agent.

        :return: Boolean, which specifies if this vehicle is an agent
        """
        return False

    def is_simulation_agent(self) -> bool:
        """
        Checks if this vehicle is a simulation agent.

        :return: Boolean, which specifies if this vehicle is a simulation agent
        """
        return False

    def clean_up(self):
        """
        Cleans up the object. Deletes unused objects.
        """
        self.route.clean_up()
        del self.route
        del self.grid
        del self.route_line
        del self.leader

    def is_almost_colliding(self) -> (bool, bool):
        """
        Checks if this vehicle is almost colliding with another vehicle.

        :return: Boolean, which specifies if this vehicle is almost colliding
        """
        grid = self.grid
        road = self.current_lane.get_road(grid)

        collision_pedestrian = False
        collision_car = False

        if self.leader and self.leader_distance < 1.0:
            collision_car = True

        rect, angle = self.get_info()

        for pedestrian in road.crossing_pedestrians:
            if pedestrian.rect.distance(rect) < 1.0:
                collision_pedestrian = True
                break

        if road.is_inter() and not collision_car:
            nearby_cars = road.get_nearby_cars(grid)
            for car in nearby_cars:
                if car is not self and car is not self.leader \
                        and car.current_lane in road.intersecting_lanes[self.current_lane]:
                    car_road = car.current_lane.get_road(grid)

                    if car_road.is_inter():
                        car_rect, car_angle = car.get_info()

                        if rect.distance(car_rect) < 1.0:
                            collision_car = True
                            break

        return collision_pedestrian, collision_car

    def is_colliding(self):
        """
        Checks if this vehicle is colliding with another vehicle.

        :return: Boolean, which specifies if this vehicle is colliding
        """
        grid = self.grid
        road = self.current_lane.get_road(grid)
        rect, angle = self.get_info()

        collision_pedestrian = False
        collision_car = False

        for pedestrian in road.crossing_pedestrians:
            if pedestrian.rect.intersects(rect):
                collision_pedestrian = True
                break

        nearby_cars = road.get_nearby_cars(grid)

        for car in nearby_cars:
            if car is not self:
                car_rect, car_angle = car.get_info()

                if rect.intersects(car_rect):
                    collision_car = True
                    break

        return collision_pedestrian, collision_car

    def update_position(self):
        """
        Updates the 1D position of the vehicle for one time step.
        """

        # Delta time. This is the time that passes between steps
        dt = CONFIG.step_length

        # Calculates the position and velocity in the current step
        if self.v + self.a * dt < 0:  # The velocity is not allowed to be negative
            self.x -= 1 / 2 * self.v * self.v / self.a  # Adjust the position
            self.v = 0
        else:  # Standard case
            self.x += self.v * dt + 1 / 2 * self.a * math.pow(dt, 2)
            self.v += self.a * dt

        # Get the first vehicle in front of the vehicle (the "leader")
        self.leader, self.leader_distance = self.route.get_leader_info_on_route()
        if self.leader:
            self.leader_distance = self.leader_distance - (self.length / 2) - (self.leader.length / 2)

        # Get the distance to the next intersection
        self.inter_distance = self.route.get_inter_distance_on_route()

        self.pedestrian_distance = self.route.get_pedestrian_distance_on_route()

        # Get the free road length (the smallest distance to an upcoming object)
        self.free_road_length = self.get_free_road_length(self.leader_distance, self.inter_distance,
                                                          self.pedestrian_distance)

    def update_current_lane(self):
        """
        Checks if the lane needs to be changed and updates the current lane.
        """

        # If a vehicles position on a lane is greater than the lane's length,
        # it needs to be moved to the next lane
        if self.x >= self.current_lane.global_line.length:
            self.x -= self.current_lane.global_line.length
            self.current_lane = self.route.move_to_next_lane()

    def update(self):
        """
        Updates the vehicle for one step.
        """
        pass

    def get_angle(self, old_pos, new_pos) -> float:
        """
        Returns heading angle using two consecutive positions.

        :param old_pos: The old 2D position
        :param new_pos: The new 2D position
        :return: The heading angle
        """

        new_angle = angle_between((old_pos.x, old_pos.y), (new_pos.x, new_pos.y))
        new_angle -= 180

        diff_angle = self.old_angle - new_angle
        if abs(diff_angle) > 1:
            return new_angle

        return new_angle

    def get_info(self) -> tuple[Polygon, float]:
        """
        Returns some info on the vehicle, which should not be calculated for each vehicle in every step,
        since it is computationally expensive.

        :return: Tuple - 0: The geometric representation as a polygon; 1: The heading angle of the vehicle
        """
        if self.current_info is None:

            rect = self.old_rect

            diff = np.array([self.pos.x - rect.centroid.x, self.pos.y - rect.centroid.y])

            rect = shapely.affinity.translate(rect, diff[0], diff[1])

            # Calculates the heading angle using the old position and the new position
            if self.v > 0.2:
                angle = self.get_angle(self.old_pos, self.pos)
            else:
                angle = self.get_angle(self.pos, self.current_lane.global_line.interpolate(self.x + 0.01, False))

            diff_angle = self.old_angle - angle

            if abs(diff_angle) > 1:
                rect = shapely.affinity.rotate(rect, angle=diff_angle, use_radians=False)
                self.old_angle = angle

            self.old_rect = rect

            self.current_info = rect, self.angle

            return rect, self.angle
        else:
            return self.current_info

    def update_2d_position(self):
        """
        Updates the 2D position of the vehicle for one time step.
        """

        # Calculates new 2D position of the vehicle
        self.old_pos = self.pos
        self.pos = self.current_lane.global_line.interpolate(self.x, False)

        difference = get_vector(self.pos) - get_vector(self.old_pos)
        # print((difference), np.linalg.norm(difference))
        self.direction_vector = get_normalized_vector(difference)

    @staticmethod
    def get_free_road_length(leader_distance: float, inter_distance: float, pedestrian_distance: float) -> float:
        """
        Returns the free road length.

        :param pedestrian_distance: The distance to the next pedestrian
        :param leader_distance: The distance to the leader
        :param inter_distance: The distance to the next intersection
        :return: The smallest distance to the next object
        """
        free_road_length = None
        values = [leader_distance, inter_distance, pedestrian_distance]
        for value in values:
            if free_road_length is None:
                free_road_length = value
            else:
                if value and value < free_road_length:
                    free_road_length = value

        return free_road_length


class Agent(Vehicle):
    """
    This is the superclass for all agents.
    """

    def __init__(self, start_lane, grid, config={}):
        """
        :param start_lane: The lane, which the agent spawns on
        :param grid: The array, which contains the road grid
        :param config: Optional configuration for the agent
        """
        super().__init__(start_lane, grid, config)

        self.route = Route(self, self.grid, CONFIG.agent_route_lookahead)

        self.view_distance = 100

        self.visibility_ray_pairs = []
        self.hidden_polygons = None
        self.hidden_polygons_list = []
        self.visible_grid = self.get_visible_grid(grid)

        self.coming_from = Directions.UP

        self.agent_waiting_time = 0.0

        # Temp for drawing functions
        self.points = []
        self.lines = []
        self.polygons = []

        self.shield_interrupted = None

        self.look_back = self.length / 2 + 3

        self.action_acceleration = 0.0

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

        self.state_representation = IER(agent=self, include_intersection=CONFIG.state_include_intersection)

    def is_agent(self):
        return True

    def clean_up(self):
        super().clean_up()

        self.state_representation.clean_up()
        del self.state_representation
        del self.visible_grid
        del self.hidden_polygons
        del self.hidden_polygons_list
        del self.visibility_ray_pairs
        del self.points
        del self.lines
        del self.polygons

    def set_action(self, action):
        """
        Sets the action of the agent.

        :param action: The action id representing an acceleration value
        """
        if action == 0:
            self.action_acceleration = 3.0
        elif action == 1:
            self.action_acceleration = 1.5
        elif action == 2:
            self.action_acceleration = 0.0
        elif action == 3:
            self.action_acceleration = -1.5
        elif action == 4:
            self.action_acceleration = -3.0
        elif action == 5:
            self.action_acceleration = -7.0

        self.set_shield_action()

    def set_shield_action(self):
        """
        Overwrites the current action if the shield would need to interrupt.
        """
        self.shield_interrupted = None
        shield_action = None

        if CONFIG.shield_include_layer_7:
            shield_action = self.get_shield_action(max_decceleration=-7.0)

        if CONFIG.shield_include_layer_3:
            if shield_action is None:
                shield_action = self.get_shield_action(max_decceleration=-3.0)

        if shield_action and shield_action < self.action_acceleration:
            self.action_acceleration = shield_action
            self.shield_interrupted = shield_action

    def get_shield_action(self, max_decceleration, just_priority=True):
        """
        Calculates and returns the shield deceleration using a maximum deceleration.

        :param max_decceleration: The maximal possible deceleration
        :param just_priority: If this is True,
        then the agent will always stop before a patch which has priority. Otherwise, the agent will only stop if the
        time intervals of upcoming cars intersect with the agent's time interval on the patch.
        :return: The shield deceleration
        """

        if self.v == 0.0 and self.action_acceleration <= 0.0:
            return None

        patches = self.state_representation.ier_patches

        thresh = CONFIG.shield_time_thresh
        d_breaking = (-(math.pow(self.v, 2))) / (2 * max_decceleration)
        d_thresh = 2.5 + d_breaking * CONFIG.shield_distance_percent

        new_velocity = self.v + self.action_acceleration * CONFIG.step_length
        if new_velocity < 0:
            new_velocity = 0

        for patch in patches[4:]:
            if (patch.priority is True or patch.tto_other < 0.1) or not CONFIG.shield_use_priority:
                if patch.tto_other < patch.t_max or (just_priority and patch.priority):
                    tto_ego, ttv_ego = patch.get_tt_ego(self.length, new_velocity)
                    if (patch.tto_other - thresh) <= tto_ego <= (patch.ttv_other + thresh) or \
                            (patch.tto_other - thresh) <= ttv_ego <= (patch.ttv_other + thresh) \
                            or patch.tto_other < 0.1 or (just_priority and patch.priority):
                        d_intersection = patch.s_start_ego
                        if 0 <= d_intersection < d_breaking + d_thresh:
                            return max_decceleration

        return None

    def is_violating_traffic_rules(self, grid):
        """
        Checks if the agent is violating traffic rules.

        :param grid: The grid array
        :return: Is the agent violating traffic rules?
        """
        current_road = self.current_lane.get_road(grid)
        if current_road.is_inter():
            for patch in self.state_representation.ier_patches[4:]:
                patch_coords = patch.polygon.exterior.coords[0]
                road = grid[get_grid_index_from_pos(grid, patch_coords[0], patch_coords[1])]

                if road is current_road and patch.priority and patch.tto_other < patch.t_max:
                    return True

        return False

    def get_patch_free_road_length(self):
        """
        Returns the free road length based on patches.

        :return: Free road length
        """
        for patch in self.state_representation.ier_patches[4:]:
            if (patch.priority or patch.intersection) and patch.s_start_ego >= 0:
                return patch.s_start_ego

        return None

    def update(self):
        """
        Updates the agent. This should be called in every simulation step.
        """

        self.current_info = None

        self.points = []
        self.lines = []
        self.polygons = []

        self.update_current_lane()

        self.update_position()

        self.route_line = self.route.get_route_line(look_back=self.look_back)

        self.a = self.action_acceleration

        if get_kilometers_per_hour(self.v) < CONFIG.agent_waiting_v_tresh:
            self.agent_waiting_time += CONFIG.step_length
        else:
            self.agent_waiting_time = 0

        self.update_2d_position()

        self.update_visibility()

        if self.old_pos != self.pos:
            self.coming_from = get_compass_direction(self.old_pos, self.pos)

        self.state_representation.update()

    """
    ###################################################################################################################
    
    Visibility
    
    ###################################################################################################################
    """

    def is_car_visible(self, car: Vehicle) -> bool:
        """
        Checks if the given vehicle can be seen by the agent.

        :param car: The car which should be checked
        :return: Boolean specifying, if the given car can be seen by the agent
        """
        rect, angle = car.get_info()

        if rect is not None and self.hidden_polygons is not None:
            if rect.covered_by(self.hidden_polygons):
                # self.polygons.append(rect)
                return False
        return True

    def is_pedestrian_visible(self, pedestrian):
        rect = pedestrian.rect

        if rect.covered_by(self.hidden_polygons):
            # self.polygons.append(rect)
            return False
        return True

    def update_visibility(self):
        """
        Calculates the visible area from the perspective of the agent.
        """

        grid_x, grid_y = self.current_lane.grid_x, self.current_lane.grid_y
        self.visible_grid = self.get_visible_grid(self.grid)

        min_object = self.visible_grid[0, 0]
        max_object = self.visible_grid[self.visible_grid.shape[0] - 1, self.visible_grid.shape[1] - 1]

        min_x, min_y = min_object.grid_x, min_object.grid_y
        max_x, max_y = max_object.grid_x + 1, max_object.grid_y + 1

        visibility_box = box(min_x * ROAD_METER_COUNT, min_y * ROAD_METER_COUNT, max_x * ROAD_METER_COUNT,
                             max_y * ROAD_METER_COUNT)
        visible_polygon = Polygon(visibility_box)
        self.visibility_ray_pairs = []
        self.hidden_polygons_list = []

        union_hidden_polygons = []

        top_right = 0
        bottom_right = 1
        bottom_left = 2
        top_left = 3

        agent_rect, agent_angle = self.get_info()

        agent_front_left = get_vector(agent_rect.exterior.coords[2])
        agent_front_right = get_vector(agent_rect.exterior.coords[3])

        agent_front_middle = agent_front_left + (agent_front_right - agent_front_left) / 2

        # self.points.append((front_middle[0], front_middle[1]))

        for index, grid_object in np.ndenumerate(self.visible_grid):
            if grid_object.is_obstacle():
                corner1 = None
                corner2 = None

                # 3 xxxx 0    # Indexes of corners of default box polygons
                #   xxxx
                #   xxxx
                # 2 xxxx 1

                if grid_x < grid_object.grid_x:
                    if grid_y < grid_object.grid_y:
                        corner1 = top_right
                        corner2 = bottom_left
                    elif grid_y == grid_object.grid_y:
                        corner1 = top_left
                        corner2 = bottom_left
                    elif grid_y > grid_object.grid_y:
                        corner1 = top_left
                        corner2 = bottom_right
                elif grid_x == grid_object.grid_x:
                    if grid_y < grid_object.grid_y:
                        corner1 = top_left
                        corner2 = top_right
                    elif grid_y == grid_object.grid_y:
                        raise ValueError("An Agent can not be on the same grid position as an obstacle!")
                    elif grid_y > grid_object.grid_y:
                        corner1 = bottom_left
                        corner2 = bottom_right
                elif grid_x > grid_object.grid_x:
                    if grid_y < grid_object.grid_y:
                        corner1 = top_left
                        corner2 = bottom_right
                    elif grid_y == grid_object.grid_y:
                        corner1 = top_right
                        corner2 = bottom_right
                    elif grid_y > grid_object.grid_y:
                        corner1 = top_right
                        corner2 = bottom_left

                obstacle_rect = grid_object.global_obstacle_rect.exterior.coords

                vec_corner1 = get_vector(obstacle_rect[corner1])
                vec_corner2 = get_vector(obstacle_rect[corner2])

                vector1 = vec_corner1 - agent_front_middle
                vector2 = vec_corner2 - agent_front_middle

                vector1 = vector1 / np.linalg.norm(vector1)
                vector2 = vector2 / np.linalg.norm(vector2)

                line1 = LineString(
                    [self.pos, obstacle_rect[corner1], agent_front_middle + ((self.view_distance * 1.5) * vector1)])
                line2 = LineString(
                    [self.pos, obstacle_rect[corner2], agent_front_middle + ((self.view_distance * 1.5) * vector2)])

                hidden_polygon = Polygon(
                    (line1.coords[1], line2.coords[1], line2.coords[2], line1.coords[2], line1.coords[1]))

                if hidden_polygon.is_valid:
                    union_hidden_polygons.append(hidden_polygon)
                    union_hidden_polygons.append(grid_object.global_obstacle_rect)
                    self.hidden_polygons_list.append((grid_object, hidden_polygon))

                self.visibility_ray_pairs.append((line1, line2))

        try:
            hidden_polygon = shapely.ops.unary_union(union_hidden_polygons)
            visibility_box = visibility_box.difference(hidden_polygon)
            visible_polygon = visible_polygon.difference(visibility_box)
        except TopologicalError as e:
            print("Topological Error occured:" + str(e))
            visibility_box = box(min_x * ROAD_METER_COUNT, min_y * ROAD_METER_COUNT, max_x * ROAD_METER_COUNT,
                                 max_y * ROAD_METER_COUNT)
            visible_polygon = Polygon(visibility_box)
            visible_polygon = visible_polygon.difference(visibility_box)

        self.hidden_polygons = visible_polygon

    def get_visible_grid(self, grid: np.ndarray) -> np.ndarray:
        grid_x, grid_y = self.current_lane.grid_x, self.current_lane.grid_y
        offset_x, offset_y = int(self.view_distance / 12), int(self.view_distance / 12)

        return get_nearby_grid(grid, grid_x, grid_y, offset_x, offset_y)


class AgentAccelerate(Agent):
    """
    A very simple agent, which accelerates until it reaches 50km/h.
    """

    def set_action(self, action):
        if get_kilometers_per_hour(self.v) < 50.0:
            self.action_acceleration = 3.0
        else:
            self.action_acceleration = 0.0

        self.set_shield_action()


class AgentTTC(Agent):
    """
    A rule based agent, which uses TTC and the IDM.
    """

    MOVE_INTER = 4

    def __init__(self, start_lane: Lane, grid: np.ndarray, config={}):
        super().__init__(start_lane, grid, config)

        # IDM Default Parameters
        self.s_0 = 2  # Jam distance (minimum desired distance to next vehicle)
        self.v_0 = get_meters_per_second(kilometers_per_hour=50.0)  # Maximum desired velocity
        self.delta = 4  # Acceleration exponent
        self.T = 1.5  # Safe time headway (reaction time)
        self.a_0 = 3.0  # Maximum acceleration
        self.b_0 = 3.0  # Desired deccelaration
        self.ab_sqrt = (2 * math.sqrt(self.a_0 * self.b_0))

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def is_agent_intersecting_with_car(self, other_car):
        """
        Checks if this agent will intersect with the route of another given car.

        :param other_car: The other car
        :return: Will the agent intersect with the route of the car?
        """
        other_car_lane_next = other_car.route.next_inter_lane
        other_car_lane_current = other_car.current_lane
        agent_lane_next = self.route.next_inter_lane
        agent_lane_current = self.current_lane

        next_inter = None
        other_inter_lane = None
        agent_inter_lane = None
        for other_lane in [other_car_lane_next, other_car_lane_current]:
            if other_lane:
                other_road = other_lane.get_road(self.grid)
                for agent_lane in [agent_lane_next, agent_lane_current]:
                    if agent_lane:
                        agent_road = agent_lane.get_road(self.grid)
                        if other_road.is_inter() and agent_road.is_inter() and other_road == agent_road:
                            next_inter = agent_road
                            other_inter_lane = other_lane
                            agent_inter_lane = agent_lane

        if next_inter:
            if other_car is not self:
                if other_inter_lane in next_inter.intersecting_lanes:
                    if agent_inter_lane in next_inter.intersecting_lanes[other_inter_lane]:
                        return True

        return False

    def can_pass_inter(self, inter):
        """
        Checks if the agent can pass the given intersection. This check is based on a TTC threshold.

        :param inter: The intersection
        :return: Can the agent pass the intersection?
        """

        waiting_cars = []
        for direction in inter.right_of_way:
            if direction in inter.lanes and direction != self.route.next_inter_lane.entry_direction:
                entry_lane = inter.lanes[direction][0]

                inter.calculate_recursive_waiting_cars(entry_lane, waiting_cars, 0, self.grid, use_worst_case=False)

        for car in inter.get_cars_on_road():
            waiting_cars.append((car, 0, 0))

        only_intersecting_cars = []
        for car_tuple in waiting_cars:
            if self.is_agent_intersecting_with_car(car_tuple[0]) and self.is_car_visible(car=car_tuple[0]):
                only_intersecting_cars.append(car_tuple)

        ttc = 1000
        if len(only_intersecting_cars) > 0:
            car, distance, ttc = min(only_intersecting_cars, key=lambda t: t[2])

        if ttc < CONFIG.agent_ttc_thresh:
            return False
        else:
            return True

    def get_inter_distance(self, ignore_passing=False):
        """
        Calculates the distance to the next intersection.

        :return: The distance to the next intersection
        """

        distance = self.current_lane.local_line.length - self.x - (self.length)

        found_inter = False
        for lane in self.route.route_lanes[1:]:
            road = self.grid[lane.grid_x, lane.grid_y]
            if road.is_inter() and (ignore_passing or not self.can_pass_inter(road)):
                distance = distance
                found_inter = True
                break

            distance += lane.local_line.length

        if found_inter:
            return distance + AgentTTC.MOVE_INTER
        else:
            return None

    def set_action(self, action):
        self.action_acceleration = np.clip(self.get_acceleration(), -7.0, 3.0)

    def get_acceleration(self):
        """
        Calculates the IDM acceleration.

        :return: The acceleration determined by the intelligent driver model
        """

        # Calculates the IDM Acceleration
        interaction_term = 0
        inter_distance = self.get_inter_distance()

        free_road_length = self.get_free_road_length(self.leader_distance, inter_distance, self.pedestrian_distance)
        if free_road_length is not None:
            leader_velocity = 0
            if self.leader:
                leader_velocity = self.leader.v

            delta_v = self.v - leader_velocity
            s = free_road_length
            desired_gap = self.s_0 + self.T * self.v + ((self.v * delta_v) / self.ab_sqrt)
            interaction_term = math.pow((desired_gap / s), 2)

        # IDM Formula
        return self.a_0 * (1 - math.pow((self.v / self.v_0), self.delta) - interaction_term)


class AgentTTCWithCreep(AgentTTC):
    """
    This is a combination of the TTC Agent and a creeping behavior at intersections.
    This is helpful for occluded intersections.
    """

    def can_pass_inter(self, inter):
        can_pass = super().can_pass_inter(inter)

        inter_distance = self.get_inter_distance(ignore_passing=True)
        if inter_distance is None:
            inter_distance = 1000

        return can_pass and inter_distance - AgentTTC.MOVE_INTER < 1.0


class AgentIDM(Agent):
    """
    This is a new rule based agent, which combines the IDM with the invariant environment representation.
    """
    def __init__(self, start_lane: Lane, grid: np.ndarray, config={}):
        super().__init__(start_lane, grid, config)

        # IDM Default Parameters
        self.s_0 = 2  # Jam distance (minimum desired distance to next vehicle)
        self.v_0 = get_meters_per_second(kilometers_per_hour=50.0)  # Maximum desired velocity
        self.delta = 4  # Acceleration exponent
        self.T = 1.5  # Safe time headway (reaction time)
        self.a_0 = 3.0  # Maximum acceleration
        self.b_0 = 3.0  # Desired deccelaration
        self.ab_sqrt = (2 * math.sqrt(self.a_0 * self.b_0))

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def set_action(self, action):
        idm_a = self.get_acceleration()
        idm_a = np.clip(idm_a, -7.0, 3.0)
        self.action_acceleration = idm_a

        # Only use shield layer 7
        self.set_shield_action()

    def get_acceleration(self):
        """
        Calculates the IDM acceleration.

        :return: The acceleration determined by the intelligent driver model
        """

        # Calculates the IDM Acceleration

        leader_velocity = 0
        leader_distance = 1000
        if self.leader:
            leader_distance = self.leader_distance
            leader_velocity = self.leader.v

        minimum_distance = 0
        for patch in self.state_representation.ier_patches[4:]:
            if patch.priority:
                minimum_distance = patch.s_start_ego
                break

        free_road_length = min(leader_distance, minimum_distance)

        interaction_term = 0
        if 0 < free_road_length < 1000:
            delta_v = self.v - leader_velocity
            s = free_road_length
            desired_gap = self.s_0 + self.T * self.v + ((self.v * delta_v) / self.ab_sqrt)
            interaction_term = math.pow((desired_gap / s), 2)

        # IDM Formula
        return self.a_0 * (1 - math.pow((self.v / self.v_0), self.delta) - interaction_term)


class AgentSimulation(Agent):
    """
    This agent represents the simulated vehicles in the simulation.
    """

    def __init__(self, start_lane: Lane, grid: np.ndarray, config={}):
        super().__init__(start_lane, grid, config)

        # IDM Default Parameters
        self.s_0 = 2  # Jam distance (minimum desired distance to next vehicle)
        self.v_0 = get_meters_per_second(kilometers_per_hour=50.0)  # Maximum desired velocity
        self.delta = 4  # Acceleration exponent
        self.T = 1.5  # Safe time headway (reaction time)
        self.a_0 = 3.0  # Maximum acceleration
        self.b_0 = 3.0  # Desired deccelaration
        self.ab_sqrt = (2 * math.sqrt(self.a_0 * self.b_0))

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def is_agent(self):
        return False

    def is_simulation_agent(self):
        return True

    def get_acceleration(self):
        """
        Calculates the IDM acceleration.

        :return: The acceleration determined by the intelligent driver model
        """

        # Calculates the IDM Acceleration
        interaction_term = 0
        free_road_length = self.get_free_road_length(self.leader_distance, self.inter_distance,
                                                     self.pedestrian_distance)
        if free_road_length is not None:
            leader_velocity = 0
            if self.leader:
                leader_velocity = self.leader.v

            delta_v = self.v - leader_velocity
            s = free_road_length
            desired_gap = self.s_0 + self.T * self.v + ((self.v * delta_v) / self.ab_sqrt)
            interaction_term = math.pow((desired_gap / s), 2)

        # IDM Formula
        return self.a_0 * (1 - math.pow((self.v / self.v_0), self.delta) - interaction_term)

    def set_action(self, action):
        idm_a = self.get_acceleration()
        idm_a = np.clip(idm_a, -7.0, 3.0)
        self.action_acceleration = idm_a


class IDMVehicle(Vehicle):
    """
    This is a vehicle, which is controlled by the intelligent driver model.
    """

    def get_rand_factor(self, mean=1, deviation=0.1, lower_cut_off=None, upper_cut_off=None):
        """
        Returns a random factor using a normal distribution.

        :param mean: The mean of the normal distribution
        :param deviation: The deviation of the normal distribution
        :param lower_cut_off: The minimal allowed value
        :param upper_cut_off: The maximal allowed value
        :return: A random factor
        """

        factor = RNG_VEHICLE.normal(mean, deviation)
        factor = np.clip(factor, 0.5, 1.5)

        if lower_cut_off:
            factor = max(lower_cut_off, factor)
        if upper_cut_off:
            factor = min(upper_cut_off, factor)

        return factor

    def __init__(self, start_lane: Lane, grid: np.ndarray, config={}):
        """
        :param start_lane: The lane, which the vehicle spawns on
        :param grid: The array, which contains the road grid
        :param config: Optional configuration for the vehicle
        """
        super().__init__(start_lane, grid, config)

        # IDM Default Parameters
        self.s_0 = 2  # Jam distance (minimum desired distance to next vehicle)
        self.v_0 = get_meters_per_second(kilometers_per_hour=50.0)  # Maximum desired velocity
        self.delta = 4  # Acceleration exponent
        self.T = 1.5  # Safe time headway (reaction time)
        self.a_0 = 3.0  # Maximum acceleration
        self.b_0 = 3.0  # Desired deccelaration
        self.ab_sqrt = (2 * math.sqrt(self.a_0 * self.b_0))

        self.v_0 = np.clip(self.v_0 * self.get_rand_factor(), get_meters_per_second(40.0), get_meters_per_second(60.0))
        self.delta *= self.get_rand_factor()
        self.T *= self.get_rand_factor()
        # self.a_0 *= self.get_rand_factor()
        # self.b_0 *= self.get_rand_factor()

        self.length = min(4.0 * self.get_rand_factor(), CAR_MAX_LENGTH)
        self.width = min(1.8 * self.get_rand_factor(), CAR_MAX_WIDTH)
        self.old_rect = box(0, 0, self.length, self.width)

        self.max_wait_time = 3 * self.get_rand_factor()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)

    def update(self) -> None:
        """
        Updates the vehicle for one step.
        """
        self.current_info = None

        self.update_current_lane()

        self.update_position()

        self.a = self.get_acceleration()

        """
        The wait_time is used for vehicles, which are waiting at an intersection. If the vehicle waits
        longer than 30 seconds, it will calculate a new random route
        """
        if self.v < 0.1 and self.inter_distance and self.inter_distance < 6 and not self.route.got_inter_pass:
            self.wait_time += CONFIG.step_length
        else:
            self.wait_time = 0

        if self.wait_time > self.max_wait_time:
            next_inter = self.route.next_inter_lane.get_road(self.grid)
            if not next_inter.is_agent_on_road() and not next_inter.is_agent_waiting() and not self.route.got_inter_pass:
                self.route.reroute()  # Calculate new random route
            self.wait_time = 0

        self.update_2d_position()

    def get_acceleration(self):
        """
        Calculates the IDM acceleration.

        :return: The acceleration determined by the intelligent driver model
        """

        # Calculates the IDM Acceleration
        interaction_term = 0
        inter_distance = self.inter_distance

        if inter_distance is not None:
            inter_distance += 0

        free_road_length = self.get_free_road_length(self.leader_distance, inter_distance, self.pedestrian_distance)
        if self.free_road_length is not None:
            leader_velocity = 0
            if self.leader:
                leader_velocity = self.leader.v

            delta_v = self.v - leader_velocity
            s = free_road_length
            desired_gap = self.s_0 + self.T * self.v + ((self.v * delta_v) / self.ab_sqrt)
            interaction_term = math.pow((desired_gap / s), 2)

        # IDM Formula
        return self.a_0 * (1 - math.pow((self.v / self.v_0), self.delta) - interaction_term)


class Route(object):
    """
    Manages the predefined route of a vehicle. A route is made out of a sequence of consecutive lanes.
    """

    def __init__(self, car: Vehicle, grid: np.ndarray, lookahead: float):
        """
        :param car: The vehicle, which is driving along this route
        :param grid: The array, which contains the roads
        :param lookahead: The maximum length of the pre-calculated route
        """
        self.route_lanes = [car.current_lane]
        self.last_lane = car.current_lane.get_previous_lanes(grid)[0]
        self.lookahead = lookahead
        self.route_length = car.current_lane.local_line.length

        self.grid = grid
        self.car = car
        self.current_turn = None
        self.next_lane = None
        self.next_turn = None
        self.next_inter_turn = Turn.STRAIGHT
        self.next_inter_lane = None
        self.got_inter_pass = False

        self.reroute()  # Calculate random route

    def clean_up(self):
        """
        Cleans up the object.
        """
        del self.route_lanes
        del self.last_lane
        del self.current_turn
        del self.grid
        del self.car
        del self.next_lane
        del self.next_inter_lane

    def get_route_line(self, look_back=None) -> LineString:
        """
        Returns the complete route as a LineString for geometric analysis.

        :param look_back: The amount of meters that the route should be offset
        :return: Route as a LineString
        """
        self.car.update_current_lane()

        route_lines = []

        # Cutting the current lane in two halves
        # -----o----->      current_lane o: car
        # -----             cut_line[0]
        #      ------       cut_line[1]

        if look_back is None:
            look_back = self.car.length / 2

        look_back = min(self.last_lane.global_line.length, look_back)

        distance = -look_back - self.car.length / 2

        if self.car.x > look_back:
            cut_lines = cut_line(self.car.current_lane.global_line,
                                 self.car.x - look_back)  # Cuts the current lane in two halves
            route_lines.append(cut_lines[1])  # This is the second part of the line
            distance += cut_lines[1].length
        else:
            # We need to consider the last lane too

            remaining_length = abs(self.car.x - look_back)
            cut_distance = self.last_lane.global_line.length - remaining_length
            cut_lines = cut_line(self.last_lane.global_line,
                                 cut_distance)  # Cuts the last lane in two halves
            route_lines.append(cut_lines[1])  # This is the second part of the last line
            distance += cut_lines[1].length

            route_lines.append(self.car.current_lane.global_line)
            distance += self.car.current_lane.global_line.length

        for lane in self.route_lanes[1:]:
            grid_line = lane.global_line
            if distance + grid_line.length <= self.lookahead:
                route_lines.append(grid_line)
                distance += grid_line.length
            else:
                cut_lines = cut_line(lane.global_line, self.lookahead - distance)
                route_lines.append(cut_lines[0])  # This is the first part of the line
                distance += cut_lines[0].length
                break

        route_line_coords = []
        for line in route_lines:
            route_line_coords.extend(line.coords)

        return LineString(route_line_coords)

    def reroute(self):
        """
        Calculates a new random route with the lookahead length.
        """
        for lane in self.route_lanes[1:]:
            lane.unregister_car(self.car, upcoming=True)

        self.route_lanes = [self.car.current_lane]
        self.route_length = self.route_lanes[0].local_line.length

        self.add_random_upcoming_lanes()

        self.current_turn = self.route_lanes[0].turn
        self.next_lane = self.route_lanes[1]
        self.next_turn = self.next_lane.turn

        self.next_inter_turn = self.get_next_intersection_turn()
        self.next_inter_lane = self.get_next_intersection_lane()

    def move_to_next_lane(self) -> Lane:
        """
        Gets called when a vehicle moves from one lane to the new upcoming lane.

        :return: The new current lane of the vehicle
        """

        old_lane = self.route_lanes.pop(0)
        old_lane.unregister_car(self.car)
        self.last_lane = old_lane
        self.route_length -= old_lane.local_line.length

        old_road = old_lane.get_road(self.grid)
        if old_road.is_inter():
            old_road.car_passed(self.car)  # Tell the intersection, that the vehicle passed the intersection

        self.add_random_upcoming_lanes()
        if len(self.route_lanes) > 0:
            new_lane = self.route_lanes[0]
            new_lane.register_car(self.car)
            new_lane.unregister_car(self.car, upcoming=True)

            new_road = new_lane.get_road(self.grid)

            if new_road.is_inter():
                self.got_inter_pass = False

            self.current_turn = self.route_lanes[0].turn
            self.next_lane = self.route_lanes[1]
            self.next_turn = self.next_lane.turn
            self.next_inter_turn = self.get_next_intersection_turn()
            self.next_inter_lane = self.get_next_intersection_lane()
            return new_lane

    def get_next_intersection_turn(self):
        """
        Returns the turn direction at the next intersection.

        :return: The turn direction at the next intersection
        """
        turn = Turn.STRAIGHT
        for lane in self.route_lanes:
            if lane.get_road(self.grid).is_inter():
                turn = lane.turn
                break

        return turn

    def get_next_intersection_lane(self):
        """
        Returns the entry lane at the next intersection.

        :return: The entry lane at the next intersection
        """
        new_lane = None
        for lane in self.route_lanes[1:]:
            if lane.get_road(self.grid).is_inter():
                new_lane = lane
                break

        return new_lane

    def is_lane_on_route(self, check_lane):
        """
        Checks if the given lane is on the route lanes.

        :param check_lane: The lane to check for
        :return: Is this lane on the route lanes?
        """
        return check_lane in self.route_lanes

    def is_lane_on_route_roads(self, check_lane):
        """
        Checks if the given lane is on the route roads.

        :param check_lane: The lane to check for
        :return: Is this lane on the route roads?
        """
        for lane in self.route_lanes:
            road = lane.get_road(self.grid)
            if road.is_lane_on_road(check_lane):
                return True
        return False

    def add_random_upcoming_lanes(self):
        """
        Adds random consecutive lanes to the route until the lookahead length is reached.
        """

        while self.route_length < self.lookahead + 30:
            lane = RNG_VEHICLE.choice(self.route_lanes[len(self.route_lanes) - 1].get_next_lanes(self.grid))
            lane.register_car(self.car, upcoming=True)
            self.route_lanes.append(lane)
            self.route_length += lane.local_line.length

    def get_pedestrian_distance_on_route(self) -> float or None:
        """
        Calculates the minimum distance to a pedestrian on the route.

        :return: The minimum distance to a pedestrian on the route
        """

        distances = []

        route_line = self.get_route_line(look_back=0)
        for lane in self.route_lanes:
            for pedestrian in lane.get_road(self.grid).crossing_pedestrians:
                if route_line.intersects(pedestrian.route_line):
                    intersection = route_line.intersection(pedestrian.route_line)
                    # todo: fix geosproject error
                    if isinstance(intersection, Point):
                        distance = route_line.project(intersection)
                        distances.append(distance)

        default_pedestrian_size = 1

        if len(distances) > 0:
            return min(distances) - (self.car.length / 2) - (default_pedestrian_size / 2)
        else:
            return None

    def get_inter_distance_on_route(self, ignore_passing=False) -> float or None:
        """
        Calculates the distance to the next intersection.

        :return: The distance to the next intersection
        """

        distance = self.car.current_lane.local_line.length - self.car.x - (self.car.length)

        found_inter = False
        for lane in self.route_lanes[1:]:
            road = self.car.grid[lane.grid_x, lane.grid_y]
            if road.is_inter() and self.car in road.passing_cars and road.can_fit(self.car):
                self.got_inter_pass = True

            if road.is_inter() and (not self.got_inter_pass or ignore_passing):
                distance = distance
                found_inter = True
                break

            distance += lane.local_line.length

        if found_inter:
            return distance
        else:
            return None

    def get_leader_info_on_route(self) -> tuple[Vehicle, float] or None:
        """
        Calculates the leader and the distance to the leader. Might be None.

        :return: Tuple - 0: the leader vehicle; 1: the distance to the leader
        """

        leader = self.get_leader_on_current_lane()

        if leader == self.car:
            leader = None
            distance = self.car.current_lane.local_line.length - self.car.x
            for lane in self.route_lanes[1:]:
                road = lane.get_road(self.grid)
                if road.is_inter():
                    leader_pairs = []
                    entry_lanes = road.lanes[lane.entry_direction]
                    for entry_lane in entry_lanes:
                        if len(entry_lane.car_queue) > 0:
                            leader = entry_lane.car_queue[len(entry_lane.car_queue) - 1]
                            new_distance = distance + min(leader.x, entry_lane.local_line.length)
                            leader_pairs.append((leader, new_distance))
                    if len(leader_pairs) > 0:
                        return min(leader_pairs, key=lambda t: t[1])
                else:
                    if len(lane.car_queue) > 0:
                        leader = lane.car_queue[len(lane.car_queue) - 1]
                        distance += min(leader.x, lane.local_line.length)
                        return leader, distance

                distance += lane.local_line.length

            if leader is None:
                distance = None
        else:
            distance = leader.x - self.car.x

        return leader, distance

    def get_leader_on_current_lane(self) -> Vehicle or None:
        """
        Returns the leader on the current lane. Might be None.

        :return: The leader on the current lane
        """
        lane = self.car.current_lane

        last_car = self.car
        leader = None
        for car in lane.car_queue:
            if car == self.car:  # When the car equals the query car, then we know that the last looked at car is the car in front
                leader = last_car
                break
            last_car = car

        return leader

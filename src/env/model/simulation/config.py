from src.env.model.simulation.utils import *


class Config:
    def __init__(self, config={}):

        # World Generation Values

        # Defining the minimum and maximum size for random map generation
        self.min_road_count_x = 15
        self.max_road_count_x = 30
        self.min_road_count_y = 15
        self.max_road_count_y = 30

        # The world generation algorithm creates circles. Here you can adjust the distance between these circles.
        self.min_between_inner_circles = 3
        self.max_between_inner_circles = 8

        # To connect the circles, the world generation algorithm creates cross roads.
        # The distance between cross roads can be adjusted.
        self.min_between_cross_roads = 2
        self.max_between_cross_roads = 10

        # Obstacles will only be generated around intersections.
        # The probability, that an obstacle is generated directly next to an intersection can be adjusted here.
        # Range from 0 to 100
        self.min_obstacle_percent = 0
        self.max_obstacle_percent = 0

        # This can be used to load the minimal example map.
        self.use_minimal_map = False

        # Adjust the number of Pedestrians
        self.min_pedestrians = 30
        self.max_pedestrians = 120
        self.pedestrian_ttc_factor = 1.25

        # Adjust the number of IDM cars
        self.min_idm_car_count = 30
        self.max_idm_car_count = 120

        # The route lookahead determines how many meters ahead the route of an IDM vehicle will be calculated.
        # This has influence on a lot of TTC calculations. So you should be careful, with changing this value.
        self.default_route_lookahead = 50

        # Agent

        # This is the lookahead of the agent.
        self.agent_route_lookahead = 50
        self.agent_count = 1
        self.agent_type = AgentTypes.DQN
        self.agent_ttc_thresh = 6.0  # Added at a later date. Do NOT CHANGE this here to preserve compatibility with older models
        self.agent_waiting_time_tresh = 20
        self.agent_waiting_v_tresh = 2.5

        # Simulation Step Length
        self.fps = 4
        self.step_length = 1 / float(self.fps)
        self.max_seconds = 120

        self.state_include_intersection = True
        self.state_include_vehicle_info = False
        self.shield_include_layer_3 = True
        self.shield_include_layer_7 = True
        self.shield_use_priority = True
        self.shield_time_thresh = 1.0
        self.shield_distance_percent = 0.3

        self.simulator_version = SimVersions.VERSION_5

        # Rewards
        self.k_v_upper = 0.06
        self.k_v_lower = 0.03
        self.k_a = 0.01
        self.k_c = 3
        self.k_c_abs = 25
        self.k_intersection = 0.2
        self.k_shield = 0.1
        self.k_distance = 0.2
        self.k_rule_violation = 5

        for attr, val in config.items():
            setattr(self, attr, val)

        if self.use_minimal_map:
            self.min_idm_car_count = 1
            self.max_idm_car_count = 30
            self.min_pedestrians = 0
            self.max_pedestrians = 20
            self.min_obstacle_percent = 0
            self.max_obstacle_percent = 100

        self.observation_space_length = 0

        patch_count = (self.agent_route_lookahead + 6)
        if self.state_include_intersection:
            self.observation_space_length = patch_count * 5
        else:
            self.observation_space_length = patch_count * 4

        if self.state_include_vehicle_info:
            self.observation_space_length += patch_count * 3 + 1

        self.observation_space_length += patch_count

        if not (
                self.min_obstacle_percent <= self.max_obstacle_percent and 0 <= self.min_obstacle_percent <= 100 and 0 <= self.max_obstacle_percent <= 100):
            raise ValueError("Unallowed configuration!")

        if self.min_between_inner_circles < 3:
            raise ValueError("Unallowed configuration!")
        if self.min_between_cross_roads < 1:
            raise ValueError("Unallowed configuration!")

        if self.min_between_cross_roads > self.max_between_cross_roads:
            raise ValueError("Unallowed configuration!")

        if self.min_between_inner_circles > self.max_between_inner_circles:
            raise ValueError("Unallowed configuration!")

        self.set_fps(self.fps)

    def set_fps(self, fps):
        self.fps = fps
        self.step_length = 1 / float(self.fps)
        return self.step_length

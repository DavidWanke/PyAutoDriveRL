import copy
import gc
import sys
import gym
import numpy as np

from src.env.model.simulation.utils import *
from src.env.model.simulation.road import ATiles
import src.env.model.simulation.car
import src.env.model.simulation.pedestrian
import src.env.model.simulation.generation.world_generation
import src.env.model.simulation.generation.minimal_map
import src.env.model.simulation.generation.crossing_roads
import src.env.model.simulation.state_representation
from src.env.model.simulation.generation.world_generation import generate_random_roads
from src.env.model.simulation.generation.minimal_map import generate_minimal_map
from src.env.model.simulation.car import IDMVehicle, Agent, AgentAccelerate, AgentIDM, AgentTTC, AgentSimulation, \
    AgentTTCWithCreep
from src.env.model.simulation.pedestrian import Pedestrian
from src.env.model.simulation.config import Config

RNG = np.random.default_rng()


class EnvLog:
    """
    This is a log class, which will be used to log data.
    """

    def __init__(self):
        self.steps = 0

        self.collisions = 0
        self.collisions_pedestrian = 0
        self.collisions_car = 0

        self.car_max_velocity = {"env": 0, "agent": 0, "nearby": 0}
        self.car_velocity = {"env": 0, "agent": 0, "nearby": 0}
        self.car_acceleration = {"env": 0, "agent": 0, "nearby": 0}
        self.car_decceleration = {"env": 0, "agent": 0, "nearby": 0}
        self.car_travelled_distance = {"env": 0, "agent": 0, "nearby": 0}
        self.car_count = {"env": 0, "agent": 0, "nearby": 0}

        self.agent_actions = {-7: 0, -3: 0, -1.5: 0, 0: 0, 1.5: 0, 3: 0}

        self.nearby_crossing_pedestrians_count = 0

        self.score = 0
        self.score_velocity = 0
        self.score_acceleration = 0
        self.score_collision = 0
        self.score_shield = 0
        self.score_intersection = 0
        self.score_distance = 0
        self.score_rule_violation = 0

        self.agent_timeouts = 0
        self.episodes = 0

    def update_car_values(self, key, car_list, env):
        count = len(car_list)
        if count > 0:
            individual_factor = 1.0 / count
            self.car_count[key] += count
            for car in car_list:
                self.car_max_velocity[key] = max(self.car_max_velocity[key], car.v)
                self.car_velocity[key] += individual_factor * car.v
                if car.a > 0:
                    self.car_acceleration[key] += individual_factor * car.a
                elif car.a < 0:
                    self.car_decceleration[key] += individual_factor * car.a
                self.car_travelled_distance[key] += individual_factor * car.v * env.config.step_length

    def log_to_dict(self, to_dict):
        log_dict = vars(self)

        agent_dict = to_dict
        for key in log_dict.keys():
            if isinstance(log_dict[key], int) or isinstance(log_dict[key], float):
                agent_dict[key] = log_dict[key]
            elif isinstance(log_dict[key], dict):
                for key2 in log_dict[key]:
                    agent_dict[str(key) + "_" + str(key2)] = log_dict[key][key2]


class Simulation(gym.Env):
    """
    This is the main simulation environment of the simulator. Everything here is measured in meters instead of pixels.
    This class implements gym interface.
    """

    def __init__(self, env_config={}, rank=0):
        self.config = Config(env_config)

        # The rank is used for multiprocessing. Each environment has a different rank.
        # Using the rank as a seed, we generate a new seed for all RNGs.
        # This results in different seeds for all environments.
        self.rank = rank
        self.rng_seed = np.random.default_rng(rank)

        # The size of the road grid measured in road tiles
        self.road_count_x = 0
        self.road_count_y = 0

        # Tracking the time, which is left in the current episode
        self.remaining_seconds = self.config.max_seconds

        # Action and Observation Spaces
        self.action_space = gym.spaces.Discrete(6, seed=None)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.config.observation_space_length,),
                                                seed=None)

        src.env.model.simulation.car.set_config(self.config)
        src.env.model.simulation.pedestrian.set_config(self.config)
        src.env.model.simulation.state_representation.set_config(self.config)

        self.temp_log = EnvLog()
        self.episode_log = EnvLog()

        self.last_seed = None

        self.collisions = 0

        self.vehicle_list = []
        self.agent_list = []
        self.pedestrian_list = []
        self.inter_list = []
        self.straight_road_list = []

        # The grid array contains grid objects. Grid objects are mainly roads, but can also be obstacles.
        # A grid object is made of 6x6 tiles
        self.grid = None

        # This array contains all tiles directly
        self.grid_tiles = None

    def get_observation(self):
        """
        Returns the observation of the current state for the agent.
        When there are multiple agents, this will return a list of observations.

        :return: An array containing the observation or a list of arrays containing the observations for each agent
        """
        if len(self.agent_list) > 0:
            if len(self.agent_list) > 1:
                observations = []
                for agent in self.agent_list:
                    observations.append(agent.state_representation.get_observation())
                return observations
            else:
                agent = self.agent_list[0]
                return agent.state_representation.get_observation()
        else:
            return np.zeros(self.observation_space.shape)

    def seed(self, seed=None) -> int:
        """
        Sets the seed for the random number generators of the simulator. If seed is None, the
        random number generators will not be reset.

        :param seed: The seed to use for the random number generators
        :return: The seed, which was used
        """

        global RNG

        if seed is None:
            seed = self.rng_seed.integers(low=0, high=sys.maxsize)

        RNG = np.random.default_rng(seed)
        src.env.model.simulation.car.set_seed(seed)
        src.env.model.simulation.pedestrian.set_seed(seed)
        src.env.model.simulation.generation.crossing_roads.set_seed(seed)
        src.env.model.simulation.generation.world_generation.set_seed(seed)
        src.env.model.simulation.generation.minimal_map.set_seed(seed)

        return seed

    def clean_up(self):
        """
        Cleans memory up and deletes all unneeded objects in the simulator.
        """

        del self.straight_road_list
        del self.inter_list
        del self.grid_tiles
        for vehicle in self.vehicle_list:
            vehicle.clean_up()
            del vehicle
        for index, grid_object in np.ndenumerate(self.grid):
            if grid_object:
                grid_object.clean_up()
            del grid_object
        del self.grid
        del self.vehicle_list

        # Garbage collect
        gc.collect()

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the simulator to a new random start state.

        :param seed: If provided, the seed will be used to set the seed for all random number generators
        :param return_info: Unused here
        :param options: Unused here
        :return: Returns the first observation of the new state
        """
        # First delete old objects
        self.clean_up()

        last_seed = self.seed(seed)
        self.last_seed = last_seed
        print(f"Resetting Environment | Process with Rank {self.rank} | Seed {last_seed}")
        self.remaining_seconds = self.config.max_seconds

        self.setup_roads()

        self.vehicle_list = []
        self.agent_list = []
        self.pedestrian_list = []

        car_spawn_roads = self.straight_road_list.copy()

        idm_car_count = RNG.integers(self.config.min_idm_car_count, self.config.max_idm_car_count, endpoint=True)
        pedestrian_count = RNG.integers(self.config.min_pedestrians, self.config.max_pedestrians, endpoint=True)

        for i in range(0, idm_car_count):
            if len(car_spawn_roads) > 0:
                road = RNG.choice(car_spawn_roads)
                car_spawn_roads.remove(road)
                first_lane = list(road.lanes.values())[0][0]
                if i < self.config.agent_count:
                    if self.config.agent_type == AgentTypes.DQN:
                        agent = Agent(first_lane, self.grid)
                    elif self.config.agent_type == AgentTypes.ACCELERATE:
                        agent = AgentAccelerate(first_lane, self.grid)
                    elif self.config.agent_type == AgentTypes.IDM:
                        agent = AgentIDM(first_lane, self.grid)
                    elif self.config.agent_type == AgentTypes.TTC:
                        agent = AgentTTC(first_lane, self.grid)
                    elif self.config.agent_type == AgentTypes.Simulation:
                        agent = AgentSimulation(first_lane, self.grid)
                    elif self.config.agent_type == AgentTypes.TTC_CREEP:
                        agent = AgentTTCWithCreep(first_lane, self.grid)
                    else:
                        raise ValueError("This agent type is not supported!")
                    self.vehicle_list.append(agent)
                    self.agent_list.append(self.vehicle_list[i])
                else:
                    self.vehicle_list.append(IDMVehicle(first_lane, self.grid))
            else:
                break

        pedestrian_spawn_roads = self.straight_road_list.copy()
        for i in range(0, pedestrian_count):
            if len(pedestrian_spawn_roads) > 0:
                road = RNG.choice(pedestrian_spawn_roads)
                pedestrian_spawn_roads.remove(road)

                self.pedestrian_list.append(Pedestrian(road, self.grid))

        return self.get_observation()

    def step(self, actions=0):
        """
        When being called, the simulator simulates one step and executes the given actions for the agents.
        When there are multiple agents, this method will return a list of observations and a list of rewards.

        :param actions: This can be an int or a list of ints for multiple agents
        :return: A tuple containing the observation (0), the reward as an int (1), a bool, which specifies, if the episode ended (2), and an info dictionary (3)
        """
        max_d = 0
        max_a = 0
        for car in self.vehicle_list:
            if car.a < 0 and car.a < max_d:
                max_d = car.a
            elif car.a > 0 and car.a > max_a:
                max_a = car.a

        if isinstance(actions, np.ndarray):
            if actions.shape == ():
                if actions is not None:
                    self.agent_list[0].set_action(actions)
            else:
                for index, agent_action in enumerate(actions):
                    if agent_action is not None:
                        self.agent_list[index].set_action(agent_action)
        else:
            if isinstance(self.agent_list[0], Agent):
                if actions is not None:
                    self.agent_list[0].set_action(actions)

        for car in self.vehicle_list:
            car.update()
        for road in self.inter_list:
            road.update_right_of_way(self.grid)
        for pedestrian in self.pedestrian_list:
            pedestrian.update(self.grid_tiles)

        done = False
        info = {"time_out": False, "collision": False, "rank": self.rank, "agent_waiting_time_out": False,
                "last_seed": self.last_seed}

        k_v_upper = self.config.k_v_upper
        k_v_lower = self.config.k_v_lower
        k_a = self.config.k_a
        k_c = self.config.k_c
        k_c_abs = self.config.k_c_abs
        k_intersection = self.config.k_intersection
        k_shield = self.config.k_shield
        k_distance = self.config.k_distance

        v_upper = get_meters_per_second(53)

        rewards = []
        for index, agent in enumerate(self.agent_list):
            if agent.v > v_upper:
                r_velocity = - k_v_upper * abs(agent.v - v_upper)
            else:
                r_velocity = + k_v_lower * abs(agent.v)

            r_acceleration = -k_a * ((2 ** abs(agent.a)) - 1)

            current_road = agent.current_lane.get_road(self.grid)

            r_intersection = 0
            if current_road.is_inter():
                r_intersection = - k_intersection

            r_shield = 0
            r_distance = 0
            if agent.shield_interrupted:
                r_shield = k_shield * agent.shield_interrupted
            else:
                if agent.v > 0.0:
                    free_road = agent.get_patch_free_road_length()

                    if free_road is None:
                        free_road = 50
                    free_road = np.clip(free_road, 0, 50)
                    r_distance = k_distance * ((abs(50 - free_road)) / 50.0)

            r_collision = 0
            colliding = agent.is_colliding()
            almost_colliding = agent.is_almost_colliding()
            collision_happened = colliding[0] or colliding[1] or almost_colliding[0] or almost_colliding[1]
            if collision_happened:
                info["collision"] = True
                done = True
                self.collisions += 1
                r_collision = -k_c * abs(get_kilometers_per_hour(agent.v)) - k_c_abs

            reward = (r_velocity + r_acceleration + r_shield + r_intersection + r_distance) * (
                        4 * self.config.step_length)
            reward += r_collision

            rewards.append(reward)

            if agent.agent_waiting_time > self.config.agent_waiting_time_tresh:
                done = True
                info["agent_waiting_time_out"] = True

            if index == 0:  # Only do this once
                self.remaining_seconds -= self.config.step_length
                if self.remaining_seconds <= 0:
                    info["time_out"] = True
                    done = True

            for agent_log in [self.temp_log, self.episode_log]:
                log = agent_log
                if done:
                    log.episodes += 1

                if info["agent_waiting_time_out"]:
                    print("Agent waiting time out!")
                    log.agent_timeouts += 1

                if collision_happened:
                    log.collisions += 1

                    if colliding[0] or almost_colliding[0]:
                        log.collisions_pedestrian += 1
                    if colliding[1] or almost_colliding[1]:
                        log.collisions_car += 1

                log.steps += 1
                log.score += reward
                log.score_velocity += r_velocity
                log.score_acceleration += r_acceleration
                log.score_collision += r_collision
                log.score_shield += r_shield
                log.score_intersection += r_intersection
                log.score_distance += r_distance

                if agent.a in log.agent_actions.keys():
                    log.agent_actions[agent.a] += 1

                log.update_car_values(key="env", car_list=self.vehicle_list, env=self)
                log.update_car_values(key="agent", car_list=self.agent_list, env=self)

                nearby_cars = current_road.get_nearby_cars(grid=self.grid, offset=2)
                nearby_cars_excluding_agents = [car for car in nearby_cars if car not in self.agent_list]

                log.update_car_values(key="nearby", car_list=nearby_cars_excluding_agents, env=self)

                nearby_pedestrians = current_road.get_nearby_crossing_pedestrians(grid=self.grid, offset=2)
                log.nearby_crossing_pedestrians_count += len(nearby_pedestrians)

            if done:
                self.episode_log.log_to_dict(info)
                self.episode_log = EnvLog()

        observation = self.get_observation()

        if len(rewards) > 1:
            agent_reward = rewards
        else:
            agent_reward = rewards[0]

        return observation, agent_reward, done, info

    def setup_roads(self):
        """
        Generates a random road grid using the config parameters.
        """
        self.road_count_x = RNG.integers(self.config.min_road_count_x, self.config.max_road_count_x, endpoint=True)
        self.road_count_y = RNG.integers(self.config.min_road_count_y, self.config.max_road_count_y, endpoint=True)

        if self.config.use_minimal_map:
            self.grid, road_count_x, road_count_y = generate_minimal_map(self.config)
            self.road_count_x = road_count_x
            self.road_count_y = road_count_y
        else:
            self.grid = generate_random_roads(self.config, self.road_count_x, self.road_count_y)

        self.grid_tiles = np.full(
            (self.road_count_x * ROAD_TILE_COUNT, self.road_count_y * ROAD_TILE_COUNT),
            fill_value=ATiles.NOTHING)
        for index, road in np.ndenumerate(self.grid):
            x, y = index
            tiles = road.tiles
            for sub_index, tile in np.ndenumerate(tiles):
                sub_x, sub_y = sub_index
                self.grid_tiles[x * ROAD_TILE_COUNT + sub_x, y * ROAD_TILE_COUNT + sub_y] = tile

        self.inter_list = []
        self.straight_road_list = []
        for index, grid_object in np.ndenumerate(self.grid):
            x, y = index
            new_copy = copy.deepcopy(grid_object)
            new_copy.initialize_map_coords(x, y)
            if new_copy.is_road():

                if new_copy.is_inter():
                    self.inter_list.append(new_copy)
                else:
                    if not new_copy.is_corner():
                        self.straight_road_list.append(new_copy)
            self.grid[index] = new_copy

    def render(self, mode="human"):
        """
        This method renders the current simulation state. This is not the method, which is preferred for rendering.
        But it can be used to show a simple state of the environment and works with multiple environments.

        :param mode: Unused here
        :return: An array containing the rgb values of the rendered image
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')

        offset = 200
        agent_road = self.agent_list[0].current_lane.get_road(self.grid)
        nearby_cars = agent_road.get_nearby_cars(self.grid, offset=offset)
        nearby_pedestrians = agent_road.get_nearby_crossing_pedestrians(self.grid, offset=offset)
        nearby_grid = get_nearby_grid(self.grid, grid_x=agent_road.grid_x, grid_y=agent_road.grid_y, offset_x=offset,
                                      offset_y=offset)

        for index, tile in np.ndenumerate(nearby_grid):
            if tile.is_road():
                xs, ys = tile.global_rect.exterior.xy
                ax.fill(xs, ys, alpha=0.5, fc='gray', ec='none')

        for car in nearby_cars:
            rect, angle = car.get_info()
            xs, ys = rect.exterior.xy

            color = 'blue'
            if car.is_agent():
                color = 'red'
            ax.fill(xs, ys, alpha=0.5, fc=color, ec='none')

        for pedestrian in nearby_pedestrians:
            xs, ys = pedestrian.rect.exterior.xy
            ax.fill(xs, ys, alpha=0.5, fc='green', ec='none')

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.flipud(data)

        plt.close(fig)

        return data

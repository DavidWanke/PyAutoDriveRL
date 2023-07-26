from src.env.model.simulation.simulation import Simulation
from src.env.model.simulation.config import Config
from constants import *
from stable_baselines3 import PPO, DQN

import pickle
from pathlib import Path


def start(args):
    #optimize_shield(args)
    find_collision_seeds(args)


def find_collision_seeds(args):
    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.experiment_name))
    rl_model_path = os.path.join(experiment_path, 'model', 'model')

    env_dict = pickle.load(open(os.path.join(experiment_path, "env" + '.pkl'), 'rb'))

    seed_min = 1

    seed_max = 1000

    env = Simulation(env_dict)
    env.config.set_fps(4)

    rl_model = DQN.load(rl_model_path, env)

    collision_seeds = []

    for seed in range(seed_min, seed_max):
        print(seed)
        print(collision_seeds)
        obs = env.reset(seed)
        done = False
        info = None
        episode_length = 0
        while not done and episode_length < 121.0:
            action, state = rl_model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            episode_length += env.config.step_length

        if info["collision"]:
            collision_seeds.append(seed)
            print("Found collision seed! :" + str(seed))
    print(collision_seeds)

def load_save_variable(args, file_name, value):
    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.experiment_name))
    optimize_path = os.path.join(experiment_path, "optimize", str(args.optimize_name))

    variable = None
    try:
        variable = pickle.load(open(os.path.join(optimize_path, file_name + '.pkl'), 'rb'))
    except (OSError, IOError) as e:
        with open(os.path.join(optimize_path, file_name + '.pkl'), 'wb') as file:
            pickle.dump(value, file)
            variable = value

    return variable


def optimize_shield(args):
    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.experiment_name))
    optimize_path = os.path.join(experiment_path, "optimize", str(args.optimize_name))

    if not os.path.exists(experiment_path):
        raise ValueError(f"The experiment {args.experiment_name} does not exist!")

    if os.path.exists(optimize_path):
        print("Trying to continue the optimization!")
        # raise ValueError(f"The optimize name {args.optimize_name} already exists! You need to choose a different name.")
    else:
        if not os.path.exists(os.path.join(experiment_path, "optimize")):
            Path(os.path.join(experiment_path, "optimize")).mkdir()
        Path(optimize_path).mkdir()

    rl_model_path = os.path.join(experiment_path, 'model', 'model')

    shield_time_range = [0.5, 1, 1.5, 2, 2.5]
    shield_distance_range = [2, 3, 4, 5, 6, 8, 12]

    env_config = Config()
    env_dict = vars(env_config)

    number_of_episodes_per_pair = 50
    episode_count = 0
    results = {}

    env_dict = load_save_variable(args, "env", env_dict)
    shield_time_range = load_save_variable(args, "shield_time_range", shield_time_range)
    shield_distance_range = load_save_variable(args, "shield_distance_range", shield_distance_range)
    number_of_episodes_per_pair = load_save_variable(args, "number_of_episodes_per_pair", number_of_episodes_per_pair)
    results = load_save_variable(args, "results", results)

    episode_weight = 1.0 / number_of_episodes_per_pair

    with open(os.path.join(optimize_path, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    seed = 1

    total_episode_count = len(shield_distance_range) * len(shield_time_range) * number_of_episodes_per_pair

    print(total_episode_count)

    for time_tresh in shield_time_range:
        for distance_thresh in shield_distance_range:
            dict_key = (time_tresh, distance_thresh)
            if dict_key in results:
                print("---------")
                print(
                    f"Already did {time_tresh} and distance_tresh {distance_thresh}.")
                episode_count += number_of_episodes_per_pair
                continue

            print("---------")
            print(
                f"Starting with time_tresh {time_tresh} and distance_tresh {distance_thresh}. Done: {episode_count / float(total_episode_count)}")

            avg_episode_length = 0.0
            avg_episode_score = 0.0
            avg_episode_collisions = 0.0
            number_of_collisions = 0

            env_dict["shield_time_thresh"] = time_tresh
            env_dict["shield_distance_thresh"] = distance_thresh
            env = Simulation(env_dict)
            env.reset(seed)

            rl_model = DQN.load(rl_model_path, env)

            for episode in range(0, number_of_episodes_per_pair):
                episode_length = 0.0
                episode_collisions = 0

                obs = env.reset()
                done = False
                info = None
                episode_score = 0
                while not done:
                    action, state = rl_model.predict(obs, deterministic=True)

                    obs, reward, done, info = env.step(action)

                    episode_length += env.config.step_length

                    episode_score += reward

                    if info["collision"]:
                        episode_collisions += 1
                        number_of_collisions += 1

                episode_count += 1

                print(episode_count)

                avg_episode_length += episode_weight * episode_length
                avg_episode_score += episode_weight * episode_score
                avg_episode_collisions += episode_weight * episode_collisions

            results[dict_key] = {"avg_episode_length": avg_episode_length,
                                 "avg_episode_score": avg_episode_score,
                                 "avg_episode_collisions": avg_episode_collisions,
                                 "number_of_collisions": number_of_collisions}
            print(f"time: {time_tresh}, distance: {distance_thresh}: {str(results[dict_key])}")

            with open(os.path.join(optimize_path, 'results.pkl'), 'wb') as file:
                pickle.dump(results, file)

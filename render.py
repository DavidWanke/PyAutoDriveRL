from src.env.view.view import View
from src.env.controller.controller import Controller
from src.env.model.model import Model
from src.env.events.event_manager import *
from stable_baselines3 import PPO, DQN
from src.env.model.simulation.simulation import Simulation, SimVersions
from src.env.model.simulation.config import Config
from src.env.model.simulation.utils import *
from constants import *
from src.env.view.constants import *
from src.env.view.recording.pygame_recorder import ScreenRecorder
from pathlib import Path
import pandas
import pickle
import re


def render(args):
    print("Rendering")

    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.name))

    if not os.path.exists(experiment_path):
        raise ValueError(f"The experiment {args.name} does not exist!")

    event_manager = EventManager()
    model = Model(event_manager)

    env_config = pickle.load(open(os.path.join(experiment_path, "env" + '.pkl'), 'rb'))

    env_config = Config(env_config)

    arg_dict = vars(args)
    for arg in arg_dict.keys():
        if str(arg).startswith(ENV_PREFIX):
            if arg_dict[arg] is not None:
                env_config.__setattr__(str(arg).split(ENV_PREFIX)[1], arg_dict[arg])

    env_config.set_fps(env_config.fps)

    env_dict = vars(env_config)
    print("Environment configuration " + str(env_dict))
    print("--------------------------")

    env_config = vars(env_config)
    if args.overwrite_env:
        config = Config()
        fresh_config = vars(config)
        for key in fresh_config:
            env_config[key] = fresh_config[key]

    # env_config["agent_waiting_time_tresh"] = 10
    # env_config["agent_type"] = AgentTypes.TTC_CREEP
    # env_config["shield_include_layer_3"] = False
    # env_config["use_minimal_map"] = True
    # env_config["agent_count"] = 1
    # env_config["min_idm_car_count"] = 15
    # env_config["max_idm_car_count"] = 20
    # env_config["min_road_count_x"] = 10
    # env_config["max_road_count_x"] = 10
    # env_config["min_road_count_y"] = 10
    # env_config["max_road_count_y"] = 10
    # env_config["min_between_inner_circles"] = 3
    # env_config["max_between_inner_circles"] = 3
    # env_config["min_between_cross_roads"] = 2
    # env_config["max_between_cross_roads"] = 3
    # env_config["min_pedestrians"] = 0
    # env_config["max_pedestrians"] = 0
    # env_config["pedestrian_ttc_tresh"] = 10.0

    print(env_config)
    env = Simulation(env_config)

    # env.config.min_obstacle_percent = 100
    # env.config.min_idm_car_count = 200
    # env.config.max_idm_car_count = 200
    # env.config.min_road_count_x = 30
    # env.config.min_road_count_y = 30
    # env.config.agent_type = AgentTypes.TTC_CREEP

    model.simulation = env
    env.reset()

    controller = Controller(event_manager=event_manager, model=model)
    view = View(event_manager=event_manager, model=model)

    rl_model_path = os.path.join(experiment_path, 'model', 'model')

    train_args = pickle.load(open(os.path.join(experiment_path, "args" + '.pkl'), 'rb'))

    if hasattr(train_args, 'algorithm'):
        if train_args.algorithm == "PPO":
            rl_model = PPO.load(rl_model_path, env)
        elif train_args.algorithm == "DQN":
            rl_model = DQN.load(rl_model_path, env)
    else:
        rl_model = DQN.load(rl_model_path, env)

    model.initialize()
    new_tick = TickEvent()
    episode = 0

    if not S_VIDEO_USE_ENV_FPS:
        env.config.set_fps(RENDER_FPS)
        video_name = f"output_{RENDER_FPS}.avi"
        view.video_recorder = ScreenRecorder(VIDEO_WIDTH, VIDEO_HEIGHT, RENDER_FPS,
                                             video_name)
    while model.running:
        if DEBUG and D_FORCE_SEED:
            obs = env.reset(D_FORCE_SEED)
        else:
            obs = env.reset()
        model.event_manager.post(ResetEvent())
        done = False
        score = 0
        while not done and model.running:
            if not model.pause:
                action = None
                if not model.debug_control:
                    action, state = rl_model.predict(obs, deterministic=True)

                obs, reward, done, info = env.step(action)
                # print(f"Reward {str(reward)}, Velocity {str(get_kilometers_per_hour(env.vehicle_list[0].v))}")

                if isinstance(reward, list):
                    pass
                else:
                    score += reward

            model.event_manager.post(new_tick)

        print(f'Episode: {episode}, Score: {score}')
        episode += 1


def render_collisions(args):
    evaluation_path = os.path.join(EXPERIMENT_PATH, str(args.name))
    log_file_path = os.path.join(evaluation_path, "log", "evaluation.csv.monitor.csv")
    max_seed_list = 10

    if not os.path.exists(evaluation_path):
        raise ValueError(f"The experiment {args.name} does not exist!")

    if not os.path.exists(os.path.join(evaluation_path, "videos")):
        Path(os.path.join(evaluation_path, "videos")).mkdir()

    model_name = "model"
    check_path = re.search(r".+\\evaluation\\.+", args.name)
    if check_path:
        experiment = args.name.split("\\")[0]
        print(experiment)
        model_path = os.path.join(EXPERIMENT_PATH, experiment, 'model', model_name)
    else:
        raise ValueError("An evaluation path must be specified")

    event_manager = EventManager()
    model = Model(event_manager)

    env_config = pickle.load(open(os.path.join(evaluation_path, "env" + '.pkl'), 'rb'))
    env = Simulation(env_config)
    model.simulation = env
    env.reset()

    controller = Controller(event_manager=event_manager, model=model)
    view = View(event_manager=event_manager, model=model)

    # model_path = os.path.join(experiment_path, 'model', 'model')
    rl_model = DQN.load(model_path, env)

    df = pandas.read_csv(log_file_path, skiprows=[0])
    df = df.rename(columns={"r": "reward", "t": "time", "l": "length"})

    df_collisions = df.loc[df["collisions"] > 0]
    seed_list = list(df_collisions['last_seed'])

    fps_list = [env.config.fps]

    model.initialize()
    new_tick = TickEvent()
    for seed in seed_list[0:10]:
        for fps in fps_list:
            episode_length = 0
            env.config.set_fps(fps)
            video_name = f"seed_{seed}_fps_{fps}.avi"
            view.video_recorder = ScreenRecorder(VIDEO_WIDTH, VIDEO_HEIGHT, fps,
                                                 out_file=os.path.join(evaluation_path, "videos", video_name))
            obs = env.reset(seed)
            model.event_manager.post(ResetEvent())
            done = False
            while not done and model.running and episode_length < 121.0:
                if not model.pause:
                    action = None
                    if not model.debug_control:
                        action, state = rl_model.predict(obs, deterministic=True)

                    obs, reward, done, info = env.step(action)
                    episode_length += env.config.step_length

                model.event_manager.post(new_tick)

            view.video_recorder.end_recording()

from src.env.model.simulation.config import Config
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, EveryNTimesteps
from src.training.env import get_env
from src.training.callbacks import *
import datetime

from stable_baselines3.dqn.policies import DQNPolicy

from constants import *
import pickle
from pathlib import Path

def start(args):
    print("Training")

    # ------------------------------------------------------------------------
    #                      Checking Paths
    # ------------------------------------------------------------------------

    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.name))
    model_path = os.path.join(experiment_path, 'model')
    log_path = os.path.join(experiment_path, 'log')
    if os.path.exists(experiment_path):
        raise ValueError(f"The experiment name {args.name} already exists! You need to choose a different name.")
    else:
        Path(experiment_path).mkdir()

    if not os.path.exists(log_path):
        Path(log_path).mkdir()

    # ------------------------------------------------------------------------
    #                      Environment
    # ------------------------------------------------------------------------

    env_config = Config()

    arg_dict = vars(args)
    for arg in arg_dict.keys():
        if str(arg).startswith(ENV_PREFIX):
            if arg_dict[arg] is not None:
                env_config.__setattr__(str(arg).split(ENV_PREFIX)[1], arg_dict[arg])

    env_config.set_fps(env_config.fps)

    env_config = Config(vars(env_config))

    env_dict = vars(env_config)
    print("Environment configuration " + str(env_dict))
    print("--------------------------")

    if args.seed:
        seed = args.seed
    else:
        seed = int(datetime.datetime.utcnow().timestamp())

    env = get_env(config=env_dict, seed=seed, number=args.num_workers,
                  filename=os.path.join(log_path, 'training_env.csv'))
    eval_env = get_env(config=env_dict, seed=seed, number=1, filename=os.path.join(log_path, 'eval_env.csv'))

    seed_list = (env.seed(seed))
    print("Seeds" + str(seed_list))

    # ------------------------------------------------------------------------
    #                      Model
    # ------------------------------------------------------------------------

    model = None
    policy_args = None
    if args.algorithm == "PPO":
        net_arch = [dict(pi=[150, 150], vf=[150, 150])]
        policy_args = {'net_arch': net_arch}
        print(policy_args)
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs=policy_args)
    elif args.algorithm == "DQN":
        if env_config.state_include_vehicle_info:
            net_arch = [170, 170]
        else:
            net_arch = [112, 112]

        policy_args = {'net_arch': net_arch}
        print(policy_args)

        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path, buffer_size=500000,
                    policy_kwargs=policy_args, exploration_fraction=args.exploration_fraction)
    else:
        raise ValueError(f"The algorithm {args.algorithm} is not supported!)")

    # ------------------------------------------------------------------------
    #                      Callbacks
    # ------------------------------------------------------------------------

    steps_per_episode = env_dict["max_seconds"] / (1.0 / env_dict["fps"])

    print("Steps per episode:" + str(steps_per_episode))

    checkpoint_callback = CheckpointCallback(save_freq=12 * steps_per_episode, save_path=model_path,
                                             name_prefix='check_model',
                                             verbose=1)

    log_callback = LogCallback(eval_freq=1 * steps_per_episode)

    eval_callback = EvalCallback(eval_env, eval_freq=6 * steps_per_episode, verbose=1,
                                 best_model_save_path=model_path)
    pause_callback = EveryNTimesteps(n_steps=200, callback=CustomCallBack())

    # ------------------------------------------------------------------------
    #                      Logging
    # ------------------------------------------------------------------------

    with open(os.path.join(experiment_path, 'env.pkl'), 'wb') as file:
        pickle.dump(env_dict, file)

    with open(os.path.join(experiment_path, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    with open(os.path.join(experiment_path, 'seeds.pkl'), 'wb') as file:
        pickle.dump(seed_list, file)

    with open(os.path.join(experiment_path, 'policy_args.pkl'), 'wb') as file:
        pickle.dump(policy_args, file)

    # ------------------------------------------------------------------------
    #                      Learning
    # ------------------------------------------------------------------------

    model.learn(total_timesteps=args.total_steps,
                callback=[checkpoint_callback, pause_callback, log_callback], progress_bar=True)

    model.save(os.path.join(model_path, "model"))


class CustomCallBack(BaseCallback):
    def _on_step(self) -> bool:
        print("Steps " + str(self.model.num_timesteps))
        return True

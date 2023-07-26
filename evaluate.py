import re

from constants import *
from stable_baselines3 import PPO, DQN
from src.training.env import get_env
from src.env.model.simulation.config import Config
from src.env.model.simulation.utils import *

from stable_baselines3.common.evaluation import evaluate_policy

from types import SimpleNamespace

import pickle
from pathlib import Path

def start(args):
    evaluate(args)

def evaluate(args):
    for model_name in args.model_names:
        experiment_path = os.path.join(EXPERIMENT_PATH, str(args.experiment_name))

        model_name_str = model_name.replace("\\",".")

        evaluation_name = f"{args.evaluation_name}_m.{model_name_str}_e.{args.num_episodes}_fps.{args.env_fps}"
        evaluation_path = os.path.join(experiment_path, "evaluation", evaluation_name)

        if os.path.exists(evaluation_path):
            raise ValueError(f"The evaluation name {evaluation_name} already exists!")
        else:
            Path(evaluation_path).mkdir(parents=True)

        if not os.path.exists(experiment_path):
            raise ValueError(f"The experiment name {args.experiment_name} does not exist!")

        log_path = os.path.join(evaluation_path, 'log')

        check_path = re.search(r".+\\model\\.+", model_name)

        if check_path:
            rl_model_path = os.path.join(EXPERIMENT_PATH, model_name)
        else:
            rl_model_path = os.path.join(experiment_path, 'model', model_name)

        env_dict = pickle.load(open(os.path.join(experiment_path, "env" + '.pkl'), 'rb'))
        env_dict["fps"] = args.env_fps

        if args.agent == "DQN":
            env_dict["agent_type"] = AgentTypes.DQN
        elif args.agent == "ACCELERATE":
            env_dict["agent_type"] = AgentTypes.ACCELERATE
        elif args.agent == "IDM":
            env_dict["agent_type"] = AgentTypes.IDM
        elif args.agent == "TTC":
            env_dict["agent_type"] = AgentTypes.TTC
        elif args.agent == "Simulation":
            env_dict["agent_type"] = AgentTypes.Simulation
        elif args.agent == "TTC_CREEP":
            env_dict["agent_type"] = AgentTypes.TTC_CREEP
        else:
            raise ValueError("This agent type is not supported!")

        env_config = Config(env_dict)

        arg_dict = vars(args)
        for arg in arg_dict.keys():
            if str(arg).startswith(ENV_PREFIX):
                if arg_dict[arg] is not None:
                    env_config.__setattr__(str(arg).split(ENV_PREFIX)[1], arg_dict[arg])
        env_config.set_fps(env_config.fps)

        env_dict = vars(env_config)

        print(env_dict)

        with open(os.path.join(evaluation_path, 'env.pkl'), 'wb') as file:
            pickle.dump(env_dict, file)

        with open(os.path.join(evaluation_path, 'args.pkl'), 'wb') as file:
            pickle.dump(args, file)

        env = get_env(config=env_dict, seed=args.seed, number=args.num_workers,
                      filename=os.path.join(log_path, 'evaluation.csv'))

        print("Seed | "+str(args.seed))

        rl_model = DQN.load(rl_model_path, env)

        evaluate_policy(model=rl_model,env=env,n_eval_episodes=args.num_episodes,deterministic=True,render=args.render)



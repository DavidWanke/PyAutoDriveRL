from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from src.env.model.simulation.simulation import Simulation
from src.env.model.simulation.simulation import EnvLog

def create_env(config, rank, seed):
    def _init():
        env = Simulation(config, rank=rank)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def get_env(config, number, seed, filename):
    if number > 1:
        env = SubprocVecEnv([create_env(config, rank, seed) for rank in range(number)])
    else:
        env = DummyVecEnv([create_env(config, 0, seed)])

    log_keys = ["collision", "time_out", "agent_waiting_time_out", "rank", "last_seed"]

    env_log = EnvLog()
    test_log = {}
    env_log.log_to_dict(test_log)

    log_keys += list(test_log.keys())

    env = VecMonitor(env, filename=filename, info_keywords=tuple(log_keys))

    return env

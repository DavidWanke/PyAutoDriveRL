from stable_baselines3.common.callbacks import BaseCallback
from src.env.model.simulation.simulation import EnvLog
from src.env.model.simulation.utils import *
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import Figure


class LogCallback(BaseCallback):

    def __init__(self, eval_freq, verbose=0):
        self.eval_freq = eval_freq

        self.record_dict = {"collisions": 0, "collisions_car": 0, "collisions_pedestrian": 0, "steps": 0, "episodes": 0,
                            "agent_timeouts": 0}

        super(LogCallback, self).__init__(verbose)

    def record_action_bar_chart(self, agent_actions, step_factor):
        x = agent_actions.keys()

        y = []

        for key in x:
            y.append(agent_actions[key] / step_factor)

        plt.xlabel('Actions')
        plt.ylabel('Frequency')

        tick_labels = [str(value) for value in x]

        figure = plt.figure()
        sub_plot = figure.add_subplot()
        sub_plot.bar(x, y, align='center', tick_label=tick_labels)
        sub_plot.set(xlabel='Actions', ylabel='Frequency')
        self.logger.record("trajectory/agent_actions", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        plt.close()

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            logs = self.training_env.get_attr("temp_log")

            individual_factor = 1.0 / len(logs)
            accumulated_log_dict = vars(EnvLog())
            for log in logs:
                log_dict = vars(log)

                for key in log_dict.keys():
                    if isinstance(log_dict[key], int) or isinstance(log_dict[key], float):
                        if "collisions" in key or key in ["steps", "agent_timeouts", "episodes"]:
                            accumulated_log_dict[key] += log_dict[key]
                        else:
                            accumulated_log_dict[key] += individual_factor * log_dict[key]
                    elif isinstance(log_dict[key], dict):
                        for key2 in log_dict[key]:
                            if "max" in key:
                                accumulated_log_dict[key][key2] = max(accumulated_log_dict[key][key2],
                                                                      log_dict[key][key2])
                            else:
                                accumulated_log_dict[key][key2] += individual_factor * log_dict[key][key2]

            log_dict = accumulated_log_dict
            steps = log_dict["steps"]
            step_factor = float(steps) / len(logs)

            for key in log_dict:
                if key not in ["steps", "agent_timeouts", "episodes"] and "collisions" not in key:

                    if isinstance(log_dict[key], int) or isinstance(log_dict[key], float):
                        prefix = ""
                        if key.startswith("score"):
                            prefix += "score/"
                        self.logger.record(prefix + str(key), log_dict[key] / step_factor)
                    elif isinstance(log_dict[key], dict):
                        for key2 in log_dict[key]:
                            value = 0
                            if "max" in key:
                                value = log_dict[key][key2]
                            else:
                                value = log_dict[key][key2] / step_factor

                            prefix = ""
                            if key.startswith("car"):
                                prefix += "car/"

                                if "velocity" in key:
                                    value = get_kilometers_per_hour(value)

                            if key.startswith("agent"):
                                prefix += "agent/"

                            self.logger.record(prefix + str(key) + "/" + str(key2), value)
                else:
                    self.record_dict[key] += log_dict[key]

            for key in self.record_dict:
                prefix = ""
                if key.startswith("collisions"):
                    prefix += "collisions/"

                self.logger.record(prefix + str(key), self.record_dict[key])

            if self.n_calls % (self.eval_freq * 8) == 0:
                self.record_action_bar_chart(log_dict["agent_actions"], step_factor)

            self.logger.dump(self.num_timesteps)

            self.training_env.set_attr("temp_log", EnvLog())

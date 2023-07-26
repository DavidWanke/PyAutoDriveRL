import os
import time
from time import sleep

import numpy as np
import pandas
import plotly.express as px
import pandas as pd
import pickle

from src.plotting.barchart import plotly_bar_charts_3d
import plotly.graph_objects as go
from constants import *
import plotly.express as px
import hashlib

import uuid

import plotly.io as pio

pio.kaleido.scope.default_format = "svg"

from scipy import signal

from src.env.model.simulation.simulation import EnvLog

import re
import ast
from src.env.model.simulation.config import Config


def get_path_name_row(df, path):
    new_row = dict(df)
    new_row["path_name"] = path
    new_row = pd.DataFrame(new_row, index=[0])

    return new_row

LOOKUP_AXIS_TITLES = {
    "path_name": r"$\text{Agenten}$",
    "action": r"$\text{Aktion als Beschleunigung in } m/s^2$",
    "probability": r"$\text{Relative Häufigkeit}$",
    "car_velocity_agent": r"$⌀\text{ Geschwindigkeit in km/h}$",
    "car_acceleration_agent": r"$⌀\text{ Positive Beschleunigung in }m/s^2$",
    "collisions": r"$\text{Absolute Häufigkeit der Kollisionen}$",
    "collision_efficiency_agent": r"$\text{Kollisions-Quotient}$",
    "energy_efficiency_agent": r"$\text{Beschleunigungs-Quotient}$",
    "reward": r"$⌀\text{ Belohnung pro Episode}$",
}

LOOKUP_EXPERIMENT = {
    "FINAL_EXPERIMENT_1": r"$\text{RL-EXP1-Agent}$",
    "FINAL_EXPERIMENT_2": r"$\text{RL-EXP2-Agent}$",
    "FINAL_EXPERIMENT_MIN_MAP": r"$\text{RL-Agent}$",
    "FINAL_EXPERIMENT_1_NO_SHIELD": r"$\text{RL-NO-SHIELD-Agent}$",
"FINAL_EXPERIMENT_1\evaluation\RL_AGENT": r"\text{RL-Agent}",

    "FINAL_EXPERIMENT_2\evaluation\ACCELERATE_AGENT": r"\text{IER+-ACC-Agent}",
    "FINAL_EXPERIMENT_2\evaluation\IDM_AGENT": r"\text{IER+-IDM-Agent}",
    "FINAL_EXPERIMENT_2\evaluation\RL_AGENT_m.check_model_23777280_steps_e.1000_fps.24": r"\text{RL-Agent OLD}",
    "FINAL_EXPERIMENT_2\evaluation\RL_AGENT_m.check_model_23362560_steps_e.1000_fps.24": r"\text{RL-EXP2-Agent}",
    "FINAL_EXPERIMENT_1\evaluation\RL_AGENT_OBSTACLES_NEW": r"\text{RL-EXP1-Agent}",
    "FINAL_EXPERIMENT_2\evaluation\TTC_CREEP_AGENT_12_NEW": r"\text{TTC-CREEP-Agent}",
    "FINAL_EXPERIMENT_2\evaluation\TTC_AGENT_12_NEW": r"\text{TTC-Agent}",
    "FINAL_EXPERIMENT_2\evaluation\RL_AGENT_m.FINAL_EXPERIMENT_MIN_MAP.model.model_e.1000_fps.24": r"\text{RL-EXP3-Agent}",

    "FINAL_EXPERIMENT_1\evaluation\IDM_AGENT": r"\text{IER+-IDM-Agent}",
    "FINAL_EXPERIMENT_1\evaluation\ACCELERATE_AGENT": r"\text{IER+-ACC-Agent}",
    "FINAL_EXPERIMENT_1\evaluation\SIMULATION_AGENT_NEW": r"\text{SIM-Agent}",
    "FINAL_EXPERIMENT_1_NO_SHIELD\evaluation\RL_AGENT_NO_SHIELD_m.check_model_24883200_steps_e.1000_fps.24": r"$\text{RL-NO-SHIELD-Agent}$",

    "FINAL_EXPERIMENT_MIN_MAP\evaluation\ACCELERATE_AGENT": r"\text{IER+-ACC-Agent}",
    "FINAL_EXPERIMENT_MIN_MAP\evaluation\IDM_AGENT": r"\text{IER+-IDM-Agent}",
    "FINAL_EXPERIMENT_MIN_MAP\evaluation\RL_AGENT_m.model_e.1000_fps.24": r"\text{RL-Agent}",
    "FINAL_EXPERIMENT_MIN_MAP\evaluation\RL_AGENT_m.FINAL_EXPERIMENT_2.model.check_model_23362560_steps_e.1000_fps.24": r"\text{RL-EXP2-Agent}",
    "FINAL_EXPERIMENT_MIN_MAP\evaluation\TTC_CREEP_AGENT_12": r"\text{TTC-CREEP-Agent}",

    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_4_NEW": r"\text{4}s",
    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_6_NEW": r"\text{6}s",
    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_8_NEW": r"\text{8}s",
    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_10_NEW": r"\text{10}s",
    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_12_NEW": r"\text{TTC-Agent}",
    "FINAL_EXPERIMENT_1\evaluation\TTC_AGENT_14_NEW": r"\text{14}s",
    
    "FINAL_EXPERIMENT_1\evaluation\performance_check_m.check_model_24468480_steps_e.100": r"\text{24.468.480}",
    "FINAL_EXPERIMENT_1\evaluation\performance_check_m.check_model_24606720_steps_e.100": r"\text{24.606.720}",
    "FINAL_EXPERIMENT_1\evaluation\performance_check_m.check_model_24744960_steps_e.100": r"\text{24.744.960}",
    "FINAL_EXPERIMENT_1\evaluation\performance_check_m.check_model_24883200_steps_e.100": r"\text{24.883.200}",
    "FINAL_EXPERIMENT_1\evaluation\performance_check_m.model_e.100": r"\text{25.000.000}"
}

class PlotModes():
    SAVE = "SAVE"
    SHOW = "SHOW"

def plot_evaluation_results(args):
    all_mean_df = pd.DataFrame()
    all_sum_df = pd.DataFrame()

    image_path = os.path.join("saved_images", "graphs", "evaluation",
                              f"random_plot.pdf")
    fig = create_legend(args.paths)
    fig.write_image(image_path, width=500, height=50, scale=2)

    mode = args.mode

    time.sleep(5)

    base_image_path = os.path.join("saved_images", "graphs","evaluation")
    experiment_name = ".".join(args.paths)
    hex_dig = hashlib.sha1(experiment_name.encode("utf-8")).hexdigest()
    base_image_name = str(hex_dig)
    for index, path in enumerate(args.paths):
        evaluation_path = os.path.join(EXPERIMENT_PATH, path)
        if not os.path.exists(evaluation_path):
            raise ValueError(f"The path {evaluation_path} does not exist!")

        check_path = re.search(r".+\\evaluation\\.+", path)

        if not check_path:
            raise ValueError(f"The path {evaluation_path} is not an evaluation path!")

        log_file_path = os.path.join(evaluation_path, "log", "evaluation.csv.monitor.csv")
        env_dict = pickle.load(open(os.path.join(evaluation_path, "env" + '.pkl'), 'rb'))
        env_config = Config(env_dict)

        df = pandas.read_csv(log_file_path, skiprows=[0])
        df = df.rename(columns={"r": "reward", "t": "time", "l": "length"})

        df_collisions = df.loc[df["collisions"] > 0]
        print(f"Collisions seeds for {path}: {list(df_collisions['last_seed'])}")

        sum_df = df.sum(axis=0)
        mean_df = df.mean(axis=0)

        for key in mean_df.keys():
            if "velocity" in key:
                mean_df[key] = sum_df[key] / sum_df["steps"]
                mean_df[key] *= 3.6

            if "count" in key or "acceleration" in key or "decceleration" in key:
                mean_df[key] = sum_df[key] / sum_df["steps"]

            if "score" in key and key != "score":
                mean_df[key] = sum_df[key] / sum_df["steps"]
                mean_df[key] *= 4 * 120

        all_sum_df = pd.concat([all_sum_df, get_path_name_row(sum_df, path)])
        all_mean_df = pd.concat([all_mean_df, get_path_name_row(mean_df, path)])

        actions = [-7, -3, -1.5, 0, 1.5, 3]
        actions_dict = {"action": [], "probability": []}
        for action in actions:
            actions_dict["action"].append(str(action))
            actions_dict["probability"].append(sum_df["agent_actions_" + str(action)])

        actions_df = pd.DataFrame(actions_dict)
        actions_df["probability"] /= actions_df["probability"].sum(axis=0)

        fig = get_bar_figure(actions_df, "action", "probability", show_annotations=False, use_one_color=True,y_range=[0,1],start_color_index=index)
        path_str = path.replace("\\",".")
        image_path = os.path.join(base_image_path,
                                  f"{path_str}.action.pdf")
        if mode == PlotModes.SAVE:
            fig.write_image(image_path, width=600, height=338, scale=2)
        else:
            fig.show()

    all_mean_df["energy_efficiency_agent"] = all_mean_df["car_acceleration_agent"] / (
            all_mean_df["car_velocity_agent"])
    all_mean_df["collision_efficiency_agent"] = all_sum_df["collisions"] / (
            all_mean_df["car_velocity_agent"])

    mean_columns = ["collision_efficiency_agent", "energy_efficiency_agent", "car_velocity_agent",
                    "reward", "car_count_agent", "car_acceleration_agent", "car_decceleration_agent"]

    env_log = EnvLog()
    env_dict = vars(env_log)

    for key in env_dict.keys():
        if "score" in key:
            mean_columns.append(key)

    for column in mean_columns:
        fig = get_bar_figure(all_mean_df, x_column="path_name", y_column=column)
        image_path = os.path.join(base_image_path,
                                  f"{base_image_name}_col.{column}.pdf")
        if mode == PlotModes.SAVE:
            fig.write_image(image_path, width=600, height=338, scale=2)
        else:
            fig.show()

    sum_columns = ["collisions", "collisions_pedestrian"]
    for column in sum_columns:
        fig = get_bar_figure(all_sum_df, x_column="path_name", y_column=column)
        image_path = os.path.join(base_image_path,
                                  f"{base_image_name}_col.{column}.pdf")
        if mode == PlotModes.SAVE:
            fig.write_image(image_path, width=600, height=338, scale=2)
        else:
            fig.show()


def get_bar_figure(df, x_column, y_column, start_color_index=0, show_annotations=True, use_one_color=False, y_range=None):
    colors = px.colors.qualitative.Plotly.copy()[start_color_index:]
    if use_one_color:
        colors = colors[0]

    x_agents = []
    for evaluation in df[x_column]:
        if evaluation in LOOKUP_EXPERIMENT:
            x_agents.append(LOOKUP_EXPERIMENT[evaluation])
        else:
            x_agents.append(evaluation)

    # Balkendiagramm erstellen
    trace = go.Bar(
        x=x_agents,
        y=df[y_column],
        marker=dict(color=colors)
    )

    annotations = []
    if show_annotations:
        annotations = [
            go.layout.Annotation(
                x=x,
                y=y,
                text=f"${round(y, ndigits=3)}$",
                showarrow=False,
                font=dict(size=14),
                xanchor="center",
                yanchor="bottom"
            )
            for x, y in zip(x_agents, df[y_column])
        ]

    x_title = x_column
    y_title = y_column

    if x_column in LOOKUP_AXIS_TITLES:
        x_title = LOOKUP_AXIS_TITLES[x_column]
    if y_column in LOOKUP_AXIS_TITLES:
        y_title = LOOKUP_AXIS_TITLES[y_column]

    # Layout erstellen
    layout = go.Layout(
        xaxis=dict(title=x_title, tickprefix=r"$", ticksuffix=r"$",tickfont=dict(size=10)),
        yaxis=dict(title=y_title, tickprefix=r"$", ticksuffix=r"$"),
        margin=dict(t=0, r=0, l=0, b=0),
        annotations=annotations
    )

    fig = go.Figure(data=[trace], layout=layout)

    if y_range:
        fig.update_layout(
            yaxis=dict(range=y_range)
        )
    return fig


def plot_optimization_results(args):
    # Optimization Plotting

    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.name))
    optimize_path = os.path.join(experiment_path, "optimize")

    optimization_dirs = [name for name in os.listdir(optimize_path) if os.path.isdir(os.path.join(optimize_path, name))]

    for directory_name in optimization_dirs:
        directory_path = os.path.join(optimize_path, directory_name)
        results = pickle.load(open(os.path.join(directory_path, 'results.pkl'), 'rb'))
        distance_range = pickle.load(open(os.path.join(directory_path, 'shield_distance_range.pkl'), 'rb'))
        time_range = pickle.load(open(os.path.join(directory_path, 'shield_time_range.pkl'), 'rb'))

        number_of_collisions_df = []
        number_of_collisions = []

        properties = {"number_of_collisions": [], "avg_episode_score": [], "avg_episode_collisions": []}

        for distance in distance_range:
            for time in time_range:
                if (time, distance) in results:
                    entry = results[(time, distance)]

                    for key in properties.keys():
                        properties[key].append(entry[key])

                    number_of_collisions_df.append([time, distance, entry["number_of_collisions"]])

        df = pd.DataFrame(number_of_collisions_df, columns=['Time', 'Distance', 'Collisions'])

        print(df.head())

        for key in properties.keys():
            fig = plotly_bar_charts_3d(time_range, distance_range, properties[key], hover_info="z",
                                       title=str((args.name) + "/optimize/" + str(directory_name)),
                                       x_title="Time in s", y_title="Distance in m", z_title=key, color='x',
                                       step=1)
            fig.show()


LOOKUP_GRAPH = {
    "score/score": {
        "y": r"$⌀ \ r_{agent}\text{ pro Simulationsschritt}$"
    },
    "score/score_acceleration": {
        "y": r"$⌀ \ r_{acceleration}\text{ pro Simulationsschritt}$"
    },
    "score/score_collision": {
        "y": r"$⌀ \ r_{collision}\text{ pro Simulationsschritt}$"
    },
    "score/score_distance": {
        "y": r"$⌀ \ r_{distance}\text{ pro Simulationsschritt}$"
    },
    "score/score_intersection": {
        "y": r"$⌀ \ r_{intersection}\text{ pro Simulationsschritt}$"
    },
    "score/score_shield": {
        "y": r"$⌀ \ r_{shield}\text{ pro Simulationsschritt}$"
    },
    "score/score_velocity": {
        "y": r"$⌀ \ r_{velocity}\text{ pro Simulationsschritt}$"
    },
    "car/car_max_velocity/agent": {
        "y": r"$\text{Maximale Geschwindigkeit in km/h}$"
    },
    "car/car_velocity/agent": {
            "y": r"$⌀\text{ Geschwindigkeit in km/h}$"
        },
    "rollout/ep_len_mean": {
                "y": r"$⌀\text{ Episodenlänge in Simulationsschritten}$"
            },
    "rollout/ep_rew_mean": {
                "y": r"$⌀\text{ Belohnung pro Episode}$"
            },
"collisions/collisions": {
                "y": r"$\text{Absolute Häufigkeit der Kollisionen}$"
            }
}


def plot_multiple_experiments(args):
    print("Plot Experiment")
    graph_names = set()
    experiment_data = {}
    for path in args.paths:
        log_path = os.path.join(EXPERIMENT_PATH, path, "log")
        experiment_data[path] = convert_tb_data(log_path)
        for name in experiment_data[path]['name'].unique():
            graph_names.add(name)

    image_path = os.path.join("saved_images", "graphs","training",
                              f"random_plot.pdf")
    fig = create_legend(args.paths)
    fig.write_image(image_path, width=500, height=50, scale=2)

    image_experiment_name = ".".join(args.paths)
    for graph_name in graph_names:
        if graph_name in LOOKUP_GRAPH.keys():
            xs = []
            ys = []
            experiment_names = []
            for experiment in args.paths:
                df = experiment_data[experiment]
                graph_data = df.loc[df['name'] == graph_name]
                xs.append(graph_data["step"])
                ys.append(graph_data["value"])
                experiment_names.append(experiment)
            fig = plot_lines(xs, ys, experiment_names, graph_name, args.start_color_index)
            image_path = os.path.join("saved_images", "graphs", "training",
                                      f"{image_experiment_name}_{graph_name.replace('/', '.')}.pdf")
            fig.write_image(image_path,width=600, height=338, scale=2)


    image_path = os.path.join("saved_images", "graphs","training",
                              f"{image_experiment_name}_legend.pdf")
    fig = create_legend(args.paths, args.start_color_index)
    fig.write_image(image_path,width=500, height=50, scale=2)


def moving_std(x, window_size):
    """Berechnet die gleitende Standardabweichung."""
    if isinstance(x, pd.Series):
        x = x.values
    return np.array([np.std(x[max(0, i - window_size): i + 1]) for i in range(len(x))])


def moving_quantile(x, window_size, q_lower, q_upper):
    """Berechnet die gleitenden Quantile."""
    if isinstance(x, pd.Series):
        x = x.values
    lower = np.array([np.percentile(x[max(0, i - window_size): i + 1], q_lower) for i in range(len(x))])
    upper = np.array([np.percentile(x[max(0, i - window_size): i + 1], q_upper) for i in range(len(x))])
    return lower, upper


def get_experiment_colors():
    colors = px.colors.qualitative.Plotly.copy()
    rgb_colors = []
    for color in colors:
        h = color.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb_colors.append(rgb)

    return rgb_colors


def create_legend(experiment_names, start_color_index=0):
    colors = px.colors.qualitative.Plotly.copy()
    colors = colors[start_color_index:]
    # Erstellen Sie ein leeres Diagramm
    fig = go.Figure()

    # Fügen Sie für jedes Experiment einen unsichtbaren Scatter-Trace hinzu
    for index, experiment in enumerate(experiment_names):
        name = experiment
        if name in LOOKUP_EXPERIMENT:
            name = LOOKUP_EXPERIMENT[name]

        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=name,
            line=dict(color=colors[index]),
            showlegend=True
        ))

    # Legende formatieren und anzeigen
    fig.update_layout(
        showlegend=True,
        margin=dict(t=5,b=5),
        legend=dict(
            orientation='h',  # Legende horizontal ausrichten
            xanchor='center',  # Legende zentrieren
            yanchor='top',
            x=0.5,
            y=1.2,
        ),
        # Achsen unsichtbar machen
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # Plot-Hintergrund unsichtbar machen
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )

    return fig


def plot_lines(xs, ys, experiment_names, graph_name, start_color_index = 0):
    rgb_colors = get_experiment_colors()[start_color_index:]
    graph_lookup = LOOKUP_GRAPH[graph_name]
    if "rollout" in graph_name or "train" in graph_name:
        window_size = 512
    else:
        window_size = 64
    error_band_window_size = int(window_size / 2)
    q_lower, q_upper = 5, 95  # 5. und 95. Perzentil
    traces = []
    for index, experiment in enumerate(experiment_names):
        x = xs[index]
        y = ys[index]

        y_smooth = signal.savgol_filter(ys[index], window_size, 3)

        r, g, b = rgb_colors[index]
        transparent_color = f"rgba({r}, {g}, {b}, 0.2)"

        # Berechnung der gleitenden Standardabweichung
        y_std = moving_std(y, error_band_window_size)
        upper_bound = y_smooth + y_std
        lower_bound = y_smooth - y_std

        # Fehlerband, das die gleitende Standardabweichung darstellt
        trace_error_band_moving_std = go.Scatter(
            x=np.concatenate([x, x[::-1]]),  # x-Werte für beide Grenzen
            y=np.concatenate([upper_bound, lower_bound[::-1]]),  # y-Werte für beide Grenzen
            fill='toself',
            fillcolor=transparent_color,  # Anpassen der Farbe und Transparenz
            line=dict(color='rgba(255, 255, 255, 0)'),  # Linienfarbe auf transparent setzen
            showlegend=False,
            name=f"{experiment} (Fehlerband Standardabweichung)"
        )

        # Berechnung der gleitenden Quantile
        lower_bound, upper_bound = moving_quantile(y, error_band_window_size, q_lower, q_upper)

        # Fehlerband, das die gleitenden Quantile darstellt
        trace_error_band_moving_quantile = go.Scatter(
            x=np.concatenate([x, x[::-1]]),  # x-Werte für beide Grenzen
            y=np.concatenate([upper_bound, lower_bound[::-1]]),  # y-Werte für beide Grenzen
            fill='toself',
            fillcolor=transparent_color,  # Anpassen der Farbe und Transparenz
            line=dict(color='rgba(255, 255, 255, 0)'),  # Linienfarbe auf transparent setzen
            showlegend=False,
            name=f"{experiment} (Fehlerband Quantile)"
        )

        # Original Linie, weniger sichtbar
        trace_original = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{experiment} (Original)",
            line=dict(color=f"rgba({r}, {g}, {b}, 0.2)"),
            showlegend=False  # Optional: Legende für die ursprünglichen Linien ausblenden
        )

        experiment_name = experiment
        if experiment in LOOKUP_EXPERIMENT:
            experiment_name = LOOKUP_EXPERIMENT[experiment]

        # Geglättete Linie, im Vordergrund
        trace_smooth = go.Scatter(
            x=x,
            y=y_smooth,
            mode="lines",
            name=f"{experiment_name}",
            line=dict(color=f"rgba({r}, {g}, {b}, 1.0)")
        )
        traces += [trace_error_band_moving_std, trace_smooth]

    layout = go.Layout(
        xaxis=dict(title=r'$\text{Simulationsschritte}$', tickprefix=r"$", ticksuffix=r"$"),
        yaxis=dict(title=graph_lookup["y"], tickprefix=r"$", ticksuffix=r"$"),
        margin=dict(t=0,r=0,l=0,b=0),
        showlegend=False
    )

    # Erstellen der Figur und Plotten
    fig = go.Figure(data=traces, layout=layout)

    return fig


def plot_experiment_results(args):
    print("Plot Experiment")
    experiment_path = os.path.join(EXPERIMENT_PATH, str(args.name))

    # RL Training Plotting
    log_path = os.path.join(experiment_path, "log")

    all_data = convert_tb_data(log_path)

    graph_names = all_data['name'].unique()

    for graph_name in graph_names:
        data = all_data.loc[all_data['name'] == graph_name]

        fig = px.line(data, title=graph_name, x="step", y="value", labels=dict(index="Steps", value="Value"))
        fig.show()


def convert_tb_data(root_dir, sort_by=None) -> pd.DataFrame:
    # Source: https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)

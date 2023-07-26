import argparse
from constants import *

def optimize(args):
    import optimize
    optimize.start(args)

def plot(args):
    import plot

    if args.type == "evaluation":
        plot.plot_evaluation_results(args)
    elif args.type == "experiment":
        plot.plot_multiple_experiments(args)
        #plot.plot_optimization_results(args)
    else:
        print("You need to specify the plotting type!")
        return

def training(args):
    import train
    train.start(args)

def evaluating(args):
    import evaluate
    evaluate.start(args)

def rendering(args):
    import render
    if args.type == "default":
        render.render(args)
    elif args.type == "collisions":
        render.render_collisions(args)
    else:
        print("You need to specify the test type!")
        return

def test(args):
    import tests

    print()

    if args.type == "memory":
        tests.test_memory(args)
    elif args.type == "performance":
        tests.test_performance(args)
    else:
        print("You need to specify the test type!")
        return


if __name__ == "__main__":
    # This is the top level parser
    parser = argparse.ArgumentParser(prog="RL Self Driving",
                                     description="This is a Reinforcement Learning environment for "
                                                 "autonomous driving in urban scenarios.")

    subparsers = parser.add_subparsers(help="Subcommands")

    # parser for training
    parser_training = subparsers.add_parser('training', help="Training Help")
    parser_training.add_argument('name', help='The name of the experiment')
    parser_training.add_argument('--algorithm', choices=['PPO', 'DQN'])
    parser_training.add_argument('--num_workers', type=int)
    parser_training.add_argument('--total_steps', type=int)
    parser_training.add_argument('--exploration_fraction', type=float)
    parser_training.add_argument('--seed', type=int)
    parser_training.add_argument('--env_shield_include_layer_3', default=True, action=argparse.BooleanOptionalAction)
    parser_training.add_argument('--env_shield_include_layer_7', default=True, action=argparse.BooleanOptionalAction)
    parser_training.add_argument('--env_use_minimal_map', default=False, action=argparse.BooleanOptionalAction)
    parser_training.add_argument(f'--env_min_obstacle_percent', type=int)
    parser_training.add_argument(f'--env_max_obstacle_percent', type=int)
    parser_training.add_argument(f'--env_agent_waiting_time_tresh', type=float)
    parser_training.add_argument(f'--env_agent_waiting_v_tresh', type=float)
    parser_training.add_argument(f'--{ENV_PREFIX}fps', type=int)
    parser_training.add_argument(f'--{ENV_PREFIX}max_seconds', type=int)

    parser_training.set_defaults(func=training, total_steps=5000000, num_workers=1, algorithm='DQN', exploration_fraction=0.1)

    # ----------------------------------------------------------------------------------------------------------------- #
    #                                                   Rendering
    # ----------------------------------------------------------------------------------------------------------------- #

    # parser for rendering
    parser_rendering = subparsers.add_parser('rendering', help="rendering help")
    parser_rendering.set_defaults(func=rendering, type="none")

    render_subparsers = parser_rendering.add_subparsers()

    parser_rendering_default = render_subparsers.add_parser('default')
    parser_rendering_default.add_argument('name', help='The name of the experiment')
    parser_rendering_default.add_argument('--overwrite_env', action=argparse.BooleanOptionalAction)
    parser_rendering_default.add_argument(f'--env_min_obstacle_percent', type=int)
    parser_rendering_default.add_argument(f'--env_max_obstacle_percent', type=int)
    parser_rendering_default.set_defaults(type="default", overwrite_env=False)

    parser_rendering_default = render_subparsers.add_parser('collisions')
    parser_rendering_default.add_argument('name', help='The name of the evaluation experiment')
    parser_rendering_default.set_defaults(type="collisions")

    # ----------------------------------------------------------------------------------------------------------------- #
    #                                                   Testing
    # ----------------------------------------------------------------------------------------------------------------- #

    parser_test = subparsers.add_parser('test')
    parser_test.set_defaults(func=test, type="none")

    test_subparsers = parser_test.add_subparsers()

    parser_test_memory = test_subparsers.add_parser('memory')
    parser_test_memory.add_argument('--resets', type=int, help="The amount of times the environment should be reset")
    parser_test_memory.set_defaults(type="memory", resets=500)

    parser_test_performance = test_subparsers.add_parser('performance')
    parser_test_performance.add_argument('--episodes', type=int, help="The amount of times the environment should be reset")
    parser_test_performance.set_defaults(type="performance", episodes=10)

    # ----------------------------------------------------------------------------------------------------------------- #
    #                                                   Plotting
    # ----------------------------------------------------------------------------------------------------------------- #

    # parser for plotting
    parser_plotting = subparsers.add_parser('plotting', help="plotting help")
    parser_plotting.set_defaults(func=plot, type="none")

    plotting_subparsers = parser_plotting.add_subparsers()

    parser_plotting_experiment = plotting_subparsers.add_parser('experiment')
    parser_plotting_experiment.add_argument('--paths', required=True, type=lambda s: [str(item) for item in s.split(',')] )
    parser_plotting_experiment.add_argument('--start_color_index', type=int)
    parser_plotting_experiment.set_defaults(type="experiment",start_color_index=0)

    parser_plotting_evaluation = plotting_subparsers.add_parser('evaluation')
    parser_plotting_evaluation.add_argument('--paths', required=True, type=lambda s: [str(item) for item in s.split(',')] )
    parser_plotting_evaluation.add_argument('--mode', choices=['SHOW','SAVE'])
    parser_plotting_evaluation.set_defaults(type="evaluation",mode="SHOW")

    # ----------------------------------------------------------------------------------------------------------------- #
    #                                                   Optimize
    # ----------------------------------------------------------------------------------------------------------------- #

    # parser for optimizing
    parser_optimize = subparsers.add_parser('optimize', help="optimize help")
    parser_optimize.add_argument('experiment_name', help='The name of the experiment')
    parser_optimize.add_argument('optimize_name', help='The name of the optimization experiment')
    parser_optimize.set_defaults(func=optimize)

    # ----------------------------------------------------------------------------------------------------------------- #
    #                                                   Evaluate
    # ----------------------------------------------------------------------------------------------------------------- #

    # parser for evaluation
    parser_evaluation = subparsers.add_parser('evaluation', help="evaluation help")
    parser_evaluation.add_argument('experiment_name', help='The name of the experiment')
    parser_evaluation.add_argument('evaluation_name', help='The name of the evaluation')
    parser_evaluation.add_argument('--num_workers', type=int)
    parser_evaluation.add_argument('--num_episodes', type=int)
    parser_evaluation.add_argument('--model_names', required=True, type=lambda s: [str(item) for item in s.split(',')] )
    parser_evaluation.add_argument('--seed', type=int)
    parser_evaluation.add_argument('--agent', choices=['DQN', 'ACCELERATE', 'IDM', 'TTC', 'Simulation', 'TTC_CREEP'])
    parser_evaluation.add_argument('--env_shield_include_layer_3', default=True, action=argparse.BooleanOptionalAction)
    parser_evaluation.add_argument('--env_shield_include_layer_7', default=True, action=argparse.BooleanOptionalAction)
    parser_evaluation.add_argument(f'--env_min_obstacle_percent', type=int)
    parser_evaluation.add_argument(f'--env_max_obstacle_percent', type=int)
    parser_evaluation.add_argument(f'--env_agent_waiting_time_tresh', type=int)
    parser_evaluation.add_argument(f'--env_agent_waiting_v_tresh', type=float)
    parser_evaluation.add_argument(f'--env_agent_ttc_thresh', type=float)
    parser_evaluation.add_argument(f'--env_fps', type=int)
    parser_evaluation.add_argument('--render', action=argparse.BooleanOptionalAction)

    parser_evaluation.set_defaults(func=evaluating, num_episodes=10, num_workers=1, seed=42, env_fps=4, render=False, agent='DQN')

    args = parser.parse_args()
    args.func(args)

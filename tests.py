from src.env.model.simulation.simulation import Simulation
from src.env.model.simulation.config import Config
import time


def test_memory(args):
    import os, psutil
    env = Simulation()

    for i in range(0, args.resets):
        print("Reset: " + str(i))
        env.reset()
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        for j in range(0, 50):
            env.step(None)


def test_performance(args):
    env = Simulation()
    env.reset(1)

    # get the start time
    st = time.time()

    steps = 0
    for i in range(0, args.episodes):
        print("Episode: " + str(i))

        obs = env.reset()
        done = False
        while not done:
            action = None

            obs, reward, done, info = env.step(action)

            steps += 1
            print(steps)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print('Time per step:', elapsed_time / float(steps), 'seconds')

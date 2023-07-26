import sys

import numpy as np

import src.env.model.simulation.generation.crossing_roads
from src.env.model.simulation.road import ARoads
from src.env.model.simulation.generation.world_generation import generate_obstacles
from src.env.model.simulation.config import Config
from src.env.model.simulation.utils import SimVersions

np.set_printoptions(threshold=sys.maxsize)

CONFIG = Config()
RNG = np.random.default_rng()


def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)


def set_config(config):
    global CONFIG
    CONFIG = config

def show_array(array):
    print("------------------------------")
    numbers = np.flipud(np.rot90(array, k=1))
    symbols = np.array([[y.symbol for y in x] for x in numbers])
    print(symbols)


def generate_minimal_map(config):
    """
    Generates a random road net using the given config.

    :param config: The configuration for the generation
    :return: Array, which contains all grid objects
    """
    # Seed Info
    set_config(config)
    src.env.model.simulation.generation.crossing_roads.set_config(config)
    src.env.model.simulation.generation.world_generation.set_config(config)

    road_count_x = 11
    road_count_y = 9

    tile_count_x = road_count_x + 2
    tile_count_y = road_count_y + 2
    all_tiles = np.full((tile_count_x, tile_count_y), ARoads.BORDER, dtype=ARoads)

    all_tiles[2,2] = ARoads.TOP_LEFT_DOWN
    all_tiles[2,3] = ARoads.DOWN
    all_tiles[2, 4] = ARoads.DOWN

    all_tiles[2, 5] = ARoads.INTER_LEFT
    all_tiles[3, 5] = ARoads.LEFT
    all_tiles[4, 5] = ARoads.LEFT

    all_tiles[2, 6] = ARoads.DOWN
    all_tiles[2, 7] = ARoads.BOTTOM_LEFT_DOWN
    all_tiles[3, 7] = ARoads.TOP_RIGHT_DOWN
    all_tiles[3, 8] = ARoads.BOTTOM_LEFT_DOWN
    all_tiles[4, 8] = ARoads.RIGHT

    all_tiles[5, 8] = ARoads.INTER_DOWN
    all_tiles[5, 7] = ARoads.UP
    all_tiles[5, 6] = ARoads.UP

    all_tiles[6, 8] = ARoads.RIGHT
    all_tiles[7, 8] = ARoads.RIGHT
    all_tiles[8, 8] = ARoads.RIGHT
    all_tiles[9, 8] = ARoads.RIGHT
    all_tiles[10, 8] = ARoads.BOTTOM_RIGHT_UP
    all_tiles[10, 7] = ARoads.UP
    all_tiles[10, 6] = ARoads.UP

    all_tiles[10, 5] = ARoads.INTER_RIGHT
    all_tiles[9, 5] = ARoads.LEFT
    all_tiles[8, 5] = ARoads.LEFT
    all_tiles[7, 5] = ARoads.LEFT
    all_tiles[6, 5] = ARoads.LEFT

    all_tiles[10, 4] = ARoads.UP
    all_tiles[10, 3] = ARoads.TOP_RIGHT_UP
    all_tiles[9, 3] = ARoads.BOTTOM_LEFT_UP
    all_tiles[9, 2] = ARoads.TOP_RIGHT_UP
    all_tiles[8, 2] = ARoads.LEFT
    all_tiles[7, 2] = ARoads.LEFT
    all_tiles[6, 2] = ARoads.LEFT

    all_tiles[5, 2] = ARoads.INTER_UP
    all_tiles[5, 3] = ARoads.UP
    all_tiles[5, 4] = ARoads.UP

    all_tiles[4, 2] = ARoads.LEFT
    all_tiles[3, 2] = ARoads.LEFT

    all_tiles[5, 5] = ARoads.CROSS_UP_F

    for index, tile in np.ndenumerate(all_tiles):
        if not tile.is_road():
            all_tiles[index] = ARoads.NOTHING

    generate_obstacles(all_tiles)

    return all_tiles[1: tile_count_x - 1, 1: tile_count_y - 1], road_count_x, road_count_y


if __name__ == '__main__':
    all_tiles,_,_ = generate_minimal_map(None)
    show_array(all_tiles)

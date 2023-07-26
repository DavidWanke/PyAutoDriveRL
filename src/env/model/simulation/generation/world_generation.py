import sys

import numpy as np

import src.env.model.simulation.generation.crossing_roads
from .utils import Point
from src.env.model.simulation.road import ARoads
from src.env.model.simulation.generation.crossing_roads import generate_crossing_roads
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


def generate_random_roads(config, tile_count_x, tile_count_y):
    """
    Generates a random road net using the given config.

    :param tile_count_x:
    :param tile_count_y:
    :param config: The configuration for the generation
    :return: Array, which contains all grid objects
    """
    # Seed Info
    set_config(config)
    src.env.model.simulation.generation.crossing_roads.set_config(config)

    tile_count_x += 2
    tile_count_y += 2
    all_tiles = np.full((tile_count_x, tile_count_y), ARoads.BORDER, dtype=ARoads)

    top_left = Point(0, 0)
    bottom_right = Point(tile_count_x - 1, tile_count_y - 1)

    rectangle = (top_left.copy(), bottom_right.copy())

    all_tiles_pos = Point(0, 0)
    while bottom_right.x - top_left.x >= 5 and bottom_right.y - top_left.y >= 5:
        new_tiles, top_left, bottom_right = generate_random_circular_path((bottom_right.x - top_left.x) + 1,
                                                                          (bottom_right.y - top_left.y) + 1)

        all_tiles[all_tiles_pos.x:(all_tiles_pos.x + new_tiles.shape[0]),
        all_tiles_pos.y:(all_tiles_pos.y + new_tiles.shape[1])] = new_tiles

        next_rectangle = (Point(all_tiles_pos.x, all_tiles_pos.y),
                          Point(all_tiles_pos.x + new_tiles.shape[0] - 1, all_tiles_pos.y + new_tiles.shape[1] - 1))
        generate_crossing_roads(all_tiles, rectangle=rectangle, next_rectangle=next_rectangle)

        all_tiles_pos.x += top_left.x
        all_tiles_pos.y += top_left.y
        rectangle = next_rectangle

    # generate_crossing_roads(all_tiles, rectangles)

    for index, tile in np.ndenumerate(all_tiles):
        if not tile.is_road():
            all_tiles[index] = ARoads.NOTHING

    generate_obstacles(all_tiles)

    return all_tiles[1: tile_count_x - 1, 1: tile_count_y - 1]

def generate_obstacles(all_tiles):
    obstacle_percent = RNG.integers(CONFIG.min_obstacle_percent, CONFIG.max_obstacle_percent, endpoint=True)
    print("Obstacle Percent: " + str(obstacle_percent))
    for index, tile in np.ndenumerate(all_tiles):
        if tile.is_road() and tile.is_inter():
            generate_obstacles_around_inter(all_tiles, index, obstacle_percent)

def generate_obstacles_around_inter(all_tiles, index, obstacle_percent):
    generated_obstacles = []
    found_other_inter = False
    for offset_x in [-1, 0, 1]:
        for offset_y in [-1, 0, 1]:
            x = index[0] + offset_x
            y = index[1] + offset_y
            if 0 <= x < all_tiles.shape[0] and 0 <= y < all_tiles.shape[1]:
                tile = all_tiles[x, y]
                if tile == ARoads.NOTHING:

                    percent = RNG.integers(1, 100, endpoint=True)
                    if percent <= obstacle_percent:
                        generated_obstacles.append((x, y))
                elif tile.is_road() and tile.is_inter() and offset_x in [-1, 1] and offset_y in [-1, 1]:
                    # TODO: These obstacles cause deadlocks
                    found_other_inter = True
                    break

    if not found_other_inter:
        for x, y in generated_obstacles:
            all_tiles[x, y] = ARoads.OBSTACLE


def generate_random_circular_path(tile_count_x: int, tile_count_y: int) -> (np.ndarray, Point, Point):
    """
    Generates a random circular road path in the given array.
    Returns the road tiles and the top left and bottom right inner corners of the path.
    This can be used for generating an inner circular road path in the returned path.

    :param tile_count_x: The width of the array.
    :param tile_count_y: The height of the array.
    :return: tiles, top_left, bottom_right: The road tiles, which form a circular path and the inner corners of the path.
    """

    # Creates a border around the array. This prevents wrong index access.
    tiles = np.full((tile_count_x, tile_count_y), ARoads.BORDER, dtype=ARoads)
    tiles[1:tile_count_x - 1, 1:tile_count_y - 1] = ARoads.NOTHING

    if tile_count_x < 5 or tile_count_y < 5:
        raise ValueError("Tile count must be higher!")

    # The center of the array:
    center = Point(int(tile_count_x / 2), int(tile_count_y / 2))

    # Random space between inner circles
    space = RNG.integers(CONFIG.min_between_inner_circles, CONFIG.max_between_inner_circles, endpoint=True)

    # A box is created in the center of the array.
    # The path creation algorithm will have to go around that box, which will result in a circular path.

    if space >= center.x or space >= center.y:
        center_border_start = Point(center.x - int(center.x / 2), center.y - int(center.y / 2))
        center_border_end = Point(center.x + int(center.x / 2) - 1, center.y + int(center.y / 2) - 1)
    else:
        center_border_start = Point(space, space)
        center_border_end = Point(tile_count_x - space, tile_count_y - space)

    tiles[center_border_start.x:center_border_end.x + 1,
    center_border_start.y: center_border_end.y + 1] = ARoads.INNER_BORDER

    # Determines the start position of the path.
    start = Point(RNG.integers(1, center_border_start.x - 1, endpoint=True), center.y)

    # Visualization
    # 11111111111111111 # 1 - Border
    # 10000000000000001 # 2 - Center Box
    # 10000000000000001 # x - Possible Start Positions
    # 1xxx2222222220001 # Actual ids and size of center box may be different!
    # 10000000000000001 #
    # 10000000000000001 #
    # 11111111111111111 #

    # The current position in the array:
    pos = Point(start.x, start.y)

    # The last position in the array:
    last_pos = Point(pos.x, pos.y)

    # First road is facing downwards:
    tiles[pos.x, pos.y] = ARoads.DOWN

    # The current target direction. This is the direction, that the algorithm intends to go to.
    # Though the chosen direction can vary, since it is chosen randomly.
    # The chosen direction will never be the opposite of the target direction.
    targeted_direction = DOWN

    # The last direction that the algorithm took:
    last_direction = DOWN

    # When the path is connected, the algorithm should stop:
    connected = False

    # The path is almost connected, when the algorithm targets left and x is smaller than the start of the center box:
    almost_connected = False

    # This tracks the amount of times, that the algorithm should go in the target direction
    # without choosing random actions.
    forward_count = 1

    # This is the inner top left corner of the rectangle, which gets created by the path.
    top_left = Point(0, 0)

    # This is the inner bottom right corner of the rectangle, which gets created by the path.
    bottom_right = Point(tile_count_x - 1, tile_count_y - 1)

    while not connected:  # Do this until the path is connected!

        # The movement in the array of the chosen direction:
        dx = 0
        dy = 0
        if targeted_direction is LEFT and pos.x <= center_border_start.x:
            # This is the case where the path is almost connected:
            almost_connected = True
            if pos.x == start.x:  # Go down to the start position, if x is same as start x.
                dx = 0
                dy = 1
            else:  # Go to the left until x is the same as start x.
                dx = -1
                dy = 0
        else:  # The path is not almost connected. This means that random actions can be taken!

            if targeted_direction is DOWN and pos.y > center_border_end.y:
                # The current position is in the bottom left corner of the array.
                # Now the path should target the right direction.
                targeted_direction = RIGHT
                forward_count = 2
            elif targeted_direction is RIGHT and pos.x > center_border_end.x:
                # The current position is in the bottom right corner of the array.
                # Now the path should target the up direction.
                targeted_direction = UP
                forward_count = 2
            elif targeted_direction is UP and pos.y < center_border_start.y:
                # The current position is in the top right corner of the array.
                # Now the path should target the left direction.
                targeted_direction = LEFT
                forward_count = 2

            if forward_count > 0:  # The algorithm should go into the target direction without choosing random.
                dx = targeted_direction.dx
                dy = targeted_direction.dy
                forward_count -= 1
            else:  # The algorithm is allowed to choose a random action.
                dx, dy = targeted_direction.choose_random_action(tiles, pos.x, pos.y)
                forward_count = 2

        # Get the current direction from the movement in the array:
        current_direction = get_direction_from_movement(dx, dy)

        if not almost_connected:  # Only do this when the path is not almost connected:
            # In general this block updates the inner top left and inner bottom right corner of the path.
            # Also, a border behind the current position is created. This prevents the algorithm from going backwards.

            if targeted_direction is DOWN:
                if pos.x + dx > top_left.x:
                    top_left.x = pos.x + dx

                for index, element in enumerate(tiles[0:(center.x + 1), last_pos.y]):
                    if element == ARoads.NOTHING:
                        tiles[index, last_pos.y] = ARoads.INNER_BORDER
            elif targeted_direction is RIGHT:
                if pos.y + dy < bottom_right.y:
                    bottom_right.y = pos.y + dy

                for index, element in enumerate(tiles[last_pos.x, center.y:tile_count_y]):
                    if element == ARoads.NOTHING:
                        tiles[last_pos.x, center.y + index] = ARoads.INNER_BORDER
            elif targeted_direction is UP:
                if pos.x + dx < bottom_right.x:
                    bottom_right.x = pos.x + dx

                for index, element in enumerate(tiles[center.x:tile_count_x, last_pos.y]):
                    if element == ARoads.NOTHING:
                        tiles[center.x + index, last_pos.y] = ARoads.INNER_BORDER
            elif targeted_direction is LEFT:
                if pos.y + dy > top_left.y:
                    top_left.y = pos.y + dy

                for index, element in enumerate(tiles[last_pos.x, 0:(center.y + 1)]):
                    if element == ARoads.NOTHING:
                        tiles[last_pos.x, index] = ARoads.INNER_BORDER

        # Calculate the fitting road tile using the last and current direction:
        current_tile = get_tile_from_direction_change(last_direction, current_direction)

        # Place the road tile at the last position:
        tiles[last_pos.x, last_pos.y] = current_tile

        # Update the current position:
        pos.x += dx
        pos.y += dy
        last_direction = current_direction
        last_pos.x = pos.x
        last_pos.y = pos.y

        if pos.x == start.x and pos.y == center.y:  # This means that the path is connected!
            connected = True

            # Leave the while loop:
            break

    # Move the inner corners of the path:
    top_left.x += 1
    top_left.y += 1
    bottom_right.x -= 1
    bottom_right.y -= 1

    return tiles, top_left, bottom_right


def list_safe_remove(choice_list: list, value: object) -> list:
    if value in choice_list:
        choice_list.remove(value)
    return choice_list


class Direction:
    def __init__(self, dx, dy, text):
        self.text = text
        self.dx = dx
        self.dy = dy

    def choose_random_action(self, array, current_x, current_y):

        possible_x = [-1, 1]
        possible_y = [-1, 1]
        list_safe_remove(possible_x, self.dx * -1)
        list_safe_remove(possible_y, self.dy * -1)

        if array[current_x - 1, current_y] != ARoads.NOTHING:
            list_safe_remove(possible_x, -1)
        if array[current_x + 1, current_y] != ARoads.NOTHING:
            list_safe_remove(possible_x, 1)
        if array[current_x, current_y - 1] != ARoads.NOTHING:
            list_safe_remove(possible_y, -1)
        if array[current_x, current_y + 1] != ARoads.NOTHING:
            list_safe_remove(possible_y, 1)

        new_dx = 0
        new_dy = 0

        if len(possible_x) > 0 and len(possible_y) > 0:
            x_or_y = RNG.integers(0, 1, endpoint=True)
            if x_or_y == 0:
                new_dx = RNG.choice(possible_x)
            elif x_or_y == 1:
                new_dy = RNG.choice(possible_y)
        elif len(possible_x) > 0 and len(possible_y) == 0:
            new_dx = RNG.choice(possible_x)
        elif len(possible_x) == 0 and len(possible_y) > 0:
            new_dy = RNG.choice(possible_y)
        else:
            print("No action possible")

        return new_dx, new_dy

    def __str__(self):
        return self.text


UP = Direction(0, -1, "up")
DOWN = Direction(0, 1, "down")
RIGHT = Direction(1, 0, "right")
LEFT = Direction(-1, 0, "left")


def get_tile_from_direction_change(last_direction: Direction, new_direction: Direction):
    tile = None
    error = False
    if last_direction == UP:
        if new_direction == UP:
            tile = ARoads.UP
        elif new_direction == DOWN:
            error = True
        elif new_direction == LEFT:
            tile = ARoads.TOP_RIGHT_UP
        elif new_direction == RIGHT:
            tile = ARoads.TOP_LEFT_UP
        else:
            error = True
    elif last_direction == DOWN:
        if new_direction == UP:
            error = True
        elif new_direction == DOWN:
            tile = ARoads.DOWN
        elif new_direction == LEFT:
            tile = ARoads.BOTTOM_RIGHT_DOWN
        elif new_direction == RIGHT:
            tile = ARoads.BOTTOM_LEFT_DOWN
        else:
            error = True
    elif last_direction == LEFT:
        if new_direction == UP:
            tile = ARoads.BOTTOM_LEFT_UP
        elif new_direction == DOWN:
            tile = ARoads.TOP_LEFT_DOWN
        elif new_direction == LEFT:
            tile = ARoads.LEFT
        elif new_direction == RIGHT:
            error = True
        else:
            error = True
    elif last_direction == RIGHT:
        if new_direction == UP:
            tile = ARoads.BOTTOM_RIGHT_UP
        elif new_direction == DOWN:
            tile = ARoads.TOP_RIGHT_DOWN
        elif new_direction == LEFT:
            error = True
        elif new_direction == RIGHT:
            tile = ARoads.RIGHT
        else:
            error = True
    else:
        error = True

    if error:
        raise ValueError(
            f"Unallowed Direction change in world generation! Last Direction:{last_direction}, New Direction:{new_direction}")

    return tile


def get_direction_from_movement(dx: int, dy: int) -> Direction:
    """
    Returns the direction, which is implied by the movement of x and y.

    :param dx: The movement in x axis.
    :param dx: The movement in y axis.
    :return: direction: The direction implied by the movement.
    """

    if dx == 0 and dy == 1:
        current_direction = DOWN
    elif dx == 0 and dy == -1:
        current_direction = UP
    elif dx == 1 and dy == 0:
        current_direction = RIGHT
    elif dx == -1 and dy == 0:
        current_direction = LEFT
    else:
        raise ValueError("Unallowed Direction change in world generation!")

    return current_direction


if __name__ == '__main__':
    for i in range(10):
        tiles = generate_random_roads(10, 20)
        show_array(tiles)

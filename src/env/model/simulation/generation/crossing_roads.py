import numpy as np
from .utils import Point
from src.env.model.simulation.road import ARoads
from src.env.model.simulation.config import Config

CONFIG = Config()
RNG = np.random.default_rng()


def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)


def set_config(config):
    global CONFIG
    CONFIG = config


def generate_crossing_roads(all_tiles: np.ndarray, rectangle: tuple[Point, Point],
                            next_rectangle: tuple[Point, Point]) -> np.ndarray:
    """
    Generates crossing roads in the given array. This generation requires that the array already has circular paths
    and the associated bounding rectangles of the circular paths.

    :param all_tiles: The array with the circular paths of roads.
    :param rectangle: The bounding rectangle of the circular path, where the roads should start from.
    :param next_rectangle: The bounding rectangle of the next inner circular path, where the roads should connect to.
    :return: The array with crossing roads between the circular paths.
    """
    # Visualization
    # 00111111111111111 # 1 - Roads
    # 1110000x000000001 # 0 - Nothing
    # 10000011110000001 # x - Possible Crossing road.
    # 1xxxxx1001xxxxxx1 # Actual ids and size of center box may be different!
    # 10000011110000001 #
    # 1111000x000001111 #
    # 00011101111111000 #

    # Rectangles are specified by the top left and bottom right corners of the rectangles.
    top_left, bottom_right = rectangle
    next_top_left, next_bottom_right = next_rectangle

    # The center position of the inner circular path.
    next_center = Point(next_top_left.x + int((next_bottom_right.x - next_top_left.x) / 2),
                        next_top_left.y + int((next_bottom_right.y - next_top_left.y) / 2))

    # We need to generate the crossing roads on copies of the original array, since the already generated crossing roads
    # would otherwise mess up the generation of the other crossing roads.

    cross_dict = {}

    # The upper crossing roads:
    upper = __generate_upper_crossing_roads(np.copy(all_tiles), top_left, bottom_right, next_top_left,
                                            next_bottom_right, next_center, cross_dict)

    # The lower crossing roads:
    lower = __generate_lower_crossing_roads(np.copy(all_tiles), top_left, bottom_right, next_top_left,
                                            next_bottom_right, next_center, cross_dict)

    # The crossing roads at the left of the array:
    left = __generate_left_crossing_roads(np.copy(all_tiles), top_left, bottom_right, next_top_left, next_bottom_right,
                                          next_center, cross_dict)

    # The crossing roads at the right of the array:
    right = __generate_right_crossing_roads(np.copy(all_tiles), top_left, bottom_right, next_top_left,
                                            next_bottom_right, next_center, cross_dict)

    crossing_roads = [upper, lower, left, right]
    for index, value in np.ndenumerate(all_tiles):
        for roads in crossing_roads:
            # Now combine all generated crossing roads. Only if there is nothing currently in the array and
            # there is a road in the generated crossing roads array or there is a new intersection,
            # overwrite the value in the original array.
            if (not value.is_road() and roads[index].is_road()) or roads[index].is_inter():
                if not value.is_cross():
                    all_tiles[index] = roads[index]

    for index in cross_dict.keys():
        all_tiles[index] = cross_dict[index]

    return all_tiles


def is_inter_around_road(all_tiles, index, axis, cross_dict):
    indexes = []
    for offset in [1, -1]:
        for offset2 in [1, -1]:
            new_index = index[0] + offset, index[1] + offset2
            if 0 <= new_index[0] < all_tiles.shape[0] and 0 <= new_index[1] < all_tiles.shape[1]:
                indexes.append(new_index)
    in_range = False
    for index in indexes:
        if all_tiles[index].is_inter():
            return True
    return False


def __generate_right_crossing_roads(all_tiles: np.ndarray, top_left: Point, bottom_right: Point, next_top_left: Point,
                                    next_bottom_right: Point, next_center: Point, cross_dict) -> np.ndarray:
    '''
    Generates the crossing roads at the right of the array.

    :param all_tiles: The array with the circular paths.
    :param top_left: The top left of the outer circular path.
    :param bottom_right: The bottom right of the outer circular path.
    :param next_top_left: The top left of the inner circular path.
    :param next_bottom_right: The bottom right of the inner circular path.
    :param next_center: The center of the inner circular path.
    :return:
    '''

    # The position where the next crossing road would be generated.
    current_pos = Point(next_bottom_right.x, next_top_left.y + 1)

    # Do this while the position is within the inner bounding rectangle.
    while current_pos.y <= next_bottom_right.y:

        # The position where the crossing road should start.
        start_pos = None

        # The position where the crossing road should end.
        end_pos = None

        # Check if all seen roads between start and end are straight roads.
        # Do not generate crossing roads, where corners are.
        all_straight = True

        # Check if start and end were found.
        found_connections = 0

        # Last road that the algorithm looked at.
        last_tile = ARoads.NOTHING

        # Go from the center of the inner bounding rectangle to the end of the outer bounding rectangle.
        for index, value in enumerate(all_tiles[next_center.x: bottom_right.x + 1, current_pos.y]):

            # Only do this while start and end were not found:
            if found_connections < 2:
                if value.is_corner() or is_inter_around_road(all_tiles, (next_center.x + index, current_pos.y), 1,
                                                             cross_dict):  # If the road is a corner, there should not be a crossing road.
                    all_straight = False
                    break
                if (value == ARoads.UP and not last_tile.is_road()) or value == ARoads.INTER_LEFT_F:
                    # If there was nothing at the last road, then this is a connection point.

                    # The position of the connection point.
                    pos = Point(next_center.x + index, current_pos.y)

                    if found_connections == 0:  # If this is the first found connection, it is the start.
                        start_pos = pos
                    elif found_connections == 1:  # If this is the second found connection, it is the end.
                        end_pos = pos
                    found_connections += 1
                last_tile = value

        # Only generate a crossing road, if there is a start and an end and if all roads are straight (no corners).
        if all_straight and start_pos is not None and end_pos is not None:

            # Generate the road from start to end.
            for index, value in enumerate(all_tiles[start_pos.x: end_pos.x + 1, start_pos.y]):

                # The current position in the array:
                index_pos = start_pos.x + index

                tile = None
                if value == ARoads.UP:  # If there is already a straight road, generate an intersection:
                    if index_pos == end_pos.x:  # Depending on if this is the end generate the appropriate intersection:
                        tile = ARoads.INTER_RIGHT
                    else:
                        tile = ARoads.INTER_LEFT_F
                elif value == ARoads.INTER_LEFT_F:
                    tile = ARoads.CROSS_LEFT_F
                else:  # Else, generate a straight road:
                    tile = ARoads.LEFT
                all_tiles[index_pos, start_pos.y] = tile
                if tile.is_inter():
                    cross_dict[(index_pos, start_pos.y)] = tile
            current_pos.y += RNG.integers(CONFIG.min_between_cross_roads, CONFIG.max_between_cross_roads, endpoint=True)
        current_pos.y += 1

    return all_tiles


def show_array(array):
    print("------------------------------")
    numbers = np.flipud(np.rot90(array, k=1))
    symbols = np.array([[y.symbol for y in x] for x in numbers])
    print(symbols)


def __generate_left_crossing_roads(all_tiles: np.ndarray, top_left: Point, bottom_right: Point, next_top_left: Point,
                                   next_bottom_right: Point, next_center: Point, cross_dict) -> np.ndarray:
    '''
        Generates the crossing roads at the left of the array.

        :param all_tiles: The array with the circular paths.
        :param top_left: The top left of the outer circular path.
        :param bottom_right: The bottom right of the outer circular path.
        :param next_top_left: The top left of the inner circular path.
        :param next_bottom_right: The bottom right of the inner circular path.
        :param next_center: The center of the inner circular path.
        :return:
        '''

    # Works similar to __generate_right_crossing_roads. Only indices and Tiles are switched.
    # For more information check out: __generate_right_crossing_roads()

    current_pos = Point(top_left.x, next_top_left.y + 1)
    while current_pos.y <= next_bottom_right.y:
        start_pos = None
        end_pos = None
        all_straight = True
        found_connections = 0
        last_tile = ARoads.NOTHING
        for index, value in enumerate(all_tiles[top_left.x: next_center.x + 1, current_pos.y]):
            if found_connections < 2:
                if value.is_corner() or is_inter_around_road(all_tiles, (top_left.x + index, current_pos.y), 1,
                                                             cross_dict):
                    all_straight = False
                    break
                if (value == ARoads.DOWN and not last_tile.is_road()) or value == ARoads.INTER_RIGHT_F:
                    pos = Point(top_left.x + index, current_pos.y)
                    if found_connections == 0:
                        start_pos = pos
                    elif found_connections == 1:
                        end_pos = pos
                    found_connections += 1
                last_tile = value
        if all_straight and start_pos is not None and end_pos is not None:
            for index, value in enumerate(all_tiles[start_pos.x: end_pos.x + 1, start_pos.y]):
                index_pos = start_pos.x + index
                tile = None

                if value == ARoads.DOWN:
                    if index_pos == end_pos.x:
                        tile = ARoads.INTER_RIGHT_F
                    else:
                        tile = ARoads.INTER_LEFT
                elif value == ARoads.INTER_RIGHT_F:
                    tile = ARoads.CROSS_RIGHT_F
                else:
                    tile = ARoads.RIGHT
                all_tiles[index_pos, start_pos.y] = tile
                if tile.is_inter():
                    cross_dict[(index_pos, start_pos.y)] = tile
            current_pos.y += RNG.integers(CONFIG.min_between_cross_roads, CONFIG.max_between_cross_roads, endpoint=True)
        current_pos.y += 1

    return all_tiles


def __generate_lower_crossing_roads(all_tiles: np.ndarray, top_left: Point, bottom_right: Point, next_top_left: Point,
                                    next_bottom_right: Point, next_center: Point, cross_dict) -> np.ndarray:
    """
        Generates the crossing roads at the bottom of the array.

        :param all_tiles: The array with the circular paths.
        :param top_left: The top left of the outer circular path.
        :param bottom_right: The bottom right of the outer circular path.
        :param next_top_left: The top left of the inner circular path.
        :param next_bottom_right: The bottom right of the inner circular path.
        :param next_center: The center of the inner circular path.
        :return:
        """

    # Works similar to __generate_right_crossing_roads. Only indices and Tiles are switched.
    # For more information check out: __generate_right_crossing_roads()

    current_pos = Point(next_top_left.x, next_center.y)
    while current_pos.x <= next_bottom_right.x:
        start_pos = None
        end_pos = None
        all_straight = True
        found_connections = 0
        last_tile = ARoads.NOTHING
        for index, value in enumerate(all_tiles[current_pos.x, next_center.y: bottom_right.y + 1]):
            if found_connections < 2:
                if value.is_corner() or is_inter_around_road(all_tiles, (current_pos.x, next_center.y + index), 0,
                                                             cross_dict):
                    all_straight = False
                    break
                if (value == ARoads.RIGHT and not last_tile.is_road()) or value == ARoads.INTER_UP_F:
                    pos = Point(current_pos.x, next_center.y + index)
                    if found_connections == 0:
                        start_pos = pos
                    elif found_connections == 1:
                        end_pos = pos
                    found_connections += 1
                last_tile = value
        if all_straight and start_pos is not None and end_pos is not None:
            for index, value in enumerate(all_tiles[start_pos.x, start_pos.y: end_pos.y + 1]):
                index_pos = start_pos.y + index
                tile = None
                if value == ARoads.RIGHT:
                    if index_pos == end_pos.y:
                        tile = ARoads.INTER_DOWN
                    else:
                        tile = ARoads.INTER_UP_F
                elif value == ARoads.INTER_UP_F:
                    tile = ARoads.CROSS_UP_F
                else:
                    tile = ARoads.UP
                all_tiles[start_pos.x, index_pos] = tile
                if tile.is_inter():
                    cross_dict[(start_pos.x, index_pos)] = tile
            current_pos.x += RNG.integers(CONFIG.min_between_cross_roads, CONFIG.max_between_cross_roads, endpoint=True)
        current_pos.x += 1

    return all_tiles


def __generate_upper_crossing_roads(all_tiles: np.ndarray, top_left: Point, bottom_right: Point, next_top_left: Point,
                                    next_bottom_right: Point, next_center: Point, cross_dict) -> np.ndarray:
    """
        Generates the crossing roads at the top of the array.

        :param all_tiles: The array with the circular paths.
        :param top_left: The top left of the outer circular path.
        :param bottom_right: The bottom right of the outer circular path.
        :param next_top_left: The top left of the inner circular path.
        :param next_bottom_right: The bottom right of the inner circular path.
        :param next_center: The center of the inner circular path.
        :return:
        """

    # Works similar to __generate_right_crossing_roads. Only indices and Tiles are switched.
    # For more information check out: __generate_right_crossing_roads()

    current_pos = Point(next_top_left.x, top_left.y)
    while current_pos.x <= next_bottom_right.x:
        start_pos = None
        end_pos = None
        all_straight = True
        found_connections = 0
        last_tile = ARoads.NOTHING
        for index, value in enumerate(all_tiles[current_pos.x, top_left.y: next_center.y + 1]):
            if found_connections < 2:
                if value.is_corner() or is_inter_around_road(all_tiles, (current_pos.x, top_left.y + index), 0,
                                                             cross_dict):
                    all_straight = False
                    break
                if (value == ARoads.LEFT and not last_tile.is_road()) or value == ARoads.INTER_DOWN_F:
                    pos = Point(current_pos.x, top_left.y + index)
                    if found_connections == 0:
                        start_pos = pos
                    elif found_connections == 1:
                        end_pos = pos
                    found_connections += 1
                last_tile = value
        if all_straight and start_pos is not None and end_pos is not None:
            for index, value in enumerate(all_tiles[start_pos.x, start_pos.y: end_pos.y + 1]):
                index_pos = start_pos.y + index
                tile = None
                if value == ARoads.LEFT:
                    if index_pos == end_pos.y:
                        tile = ARoads.INTER_DOWN_F
                    else:
                        tile = ARoads.INTER_UP
                elif value == ARoads.INTER_DOWN_F:
                    tile = ARoads.CROSS_DOWN_F
                else:
                    tile = ARoads.DOWN
                all_tiles[start_pos.x, index_pos] = tile
                if tile.is_inter():
                    cross_dict[(start_pos.x, index_pos)] = tile
            current_pos.x += RNG.integers(CONFIG.min_between_cross_roads, CONFIG.max_between_cross_roads, endpoint=True)
        current_pos.x += 1

    return all_tiles

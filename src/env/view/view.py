import pygame
from pygame.locals import *
import pygame.freetype

import shapely.geometry.base
from shapely.geometry import Polygon

from .recording.pygame_recorder import ScreenRecorder
from ..model.simulation.utils import *
from ..model.simulation.road import *
from ..model.model import Model
from ..model.simulation.car import *
from ..model.simulation.pedestrian import *
from ..events.event_manager import *
from ..view.constants import *
import os
from colorsys import rgb_to_hls, hls_to_rgb
import colour


def adjust_color_lightness(color, factor):
    r, g, b = color
    h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


class DebugMenu:
    ENV = 0
    OBJECT = 1


class View(object):
    """
    Draws the state of the model on the screen.
    """

    def __init__(self, event_manager: EventManager, model: Model):
        """

        :param event_manager: The event manager that the view listens to.
        :param model: The model of the simulation.
        """

        self.event_manager = event_manager
        event_manager.register_listener(self)
        self.model = model
        self.sim = model.simulation
        self.is_initialized = False
        self.screen = None
        self.clock = None
        self.small_font = None
        self.font = None
        self.tracking_object = None
        self.tracking_patch_index = None
        self.debug_menu = DebugMenu.ENV

        self.view_port_rect = pygame.Rect(200, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        self.dummy_rect = pygame.Rect(500, 500, 40, 50)
        self.visible_bits = {}
        self.frame_counter = 0
        self.dt = 1.0 / 60
        self.scale_factors = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        self.render_toggled = True

        red = colour.Color("red")
        color_range = list(red.range_to(colour.Color("lime"), 50))
        self.color_gradient = []
        for color in color_range:
            self.color_gradient.append((color.rgb[0] * 255, color.rgb[1] * 255, color.rgb[2] * 255))

        self.zoom = 1.0
        self.t = 0.0

    def get_patch_color_from_value(self, normalized_value):
        color_index = int(len(self.color_gradient) * normalized_value)
        color_index = min(len(self.color_gradient) - 1, color_index)

        return self.color_gradient[color_index]

    def reset(self):
        self.initial_grid_surface = self.get_initial_map()
        self.scaled_grid_surfaces = self.get_scaled_grid_surfaces(self.initial_grid_surface)

        self.tracking_object = self.sim.agent_list[0]

        self.debug_grid_surface = pygame.Surface(self.initial_grid_surface.get_size())
        self.debug_grid_surface.set_alpha(128)
        self.debug_grid_surface.fill((255, 255, 255))
        self.debug_grid_surface.set_colorkey((255, 255, 255))

    def initialize(self):
        """
        Setup pygame and load graphical resources.
        """
        pygame.init()
        pygame.font.init()
        pygame.freetype.init()
        pygame.display.set_caption('RL Driving')
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), RESIZABLE)
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.SysFont('Comic Sans MS', 8)
        self.font = pygame.freetype.SysFont("Sans", 15)
        self.is_initialized = True

        self.view_port_surface = pygame.Surface(self.view_port_rect.size)

        self.view_port_visibility_surface = pygame.Surface(self.view_port_surface.get_size())
        self.view_port_visibility_surface.set_alpha(220)
        self.view_port_visibility_surface.fill((255, 255, 255))
        self.view_port_visibility_surface.set_colorkey((255, 255, 255))

        self.view_bottom_right_grid = self.get_road_index_from_viewport_pos(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        self.view_top_left_grid = self.get_road_index_from_viewport_pos(0, 0)

        self.video_recorder = None

        if S_RECORD_VIDEO:
            video_fps = RENDER_FPS
            if S_VIDEO_USE_ENV_FPS:
                video_fps = self.sim.config.fps

            self.video_recorder = ScreenRecorder(VIDEO_WIDTH, VIDEO_HEIGHT, video_fps)
            self.video_surface = pygame.Surface((VIDEO_WIDTH, VIDEO_HEIGHT))

        self.reset()

    def get_initial_map(self):
        map_surface = pygame.Surface(
            (self.sim.road_count_x * ROAD_PIXEL_COUNT, self.sim.road_count_y * ROAD_PIXEL_COUNT))
        for index, grid_object in np.ndenumerate(self.sim.grid):
            x, y = index
            map_surface.blit(self.get_grid_object_surface(grid_object),
                             (x * ROAD_PIXEL_COUNT, y * ROAD_PIXEL_COUNT))

        if DEBUG and D_SAVE_ENV_IMAGE:
            pygame.image.save(map_surface, self.get_unique_path("saved_images/maps/map.jpg"))
        return map_surface

    def get_scaled_grid_surfaces(self, initial_surface):
        scaled_maps = {}
        for factor in self.scale_factors:
            surface = pygame.transform.scale(initial_surface, (
                initial_surface.get_width() * factor, initial_surface.get_height() * factor))
            scaled_maps[factor] = surface
        return scaled_maps

    def get_closest_grid_surface(self, factor):
        array = np.asarray(self.scale_factors)
        index = np.abs((array - factor)).argmin()
        return self.scaled_grid_surfaces[array[index]]

    def notify(self, event: Event):
        """
        Receive events posted to the message queue.

        :param event: The new event.
        """
        if isinstance(event, InitializeEvent):
            self.initialize()
        elif isinstance(event, QuitEvent):
            self.is_initialized = False
            if self.video_recorder:
                self.video_recorder.end_recording()
            pygame.quit()
        elif isinstance(event, TickEvent):
            if not self.is_initialized:
                return
            current_state = self.model.state_machine.peek()
            if current_state == self.model.state_machine.IN_MENU:
                self.render_menu()
            elif current_state == self.model.state_machine.IN_SIMULATION:
                self.check_key_input()
                if self.render_toggled:
                    self.render_simulation()
        elif isinstance(event, KeyEvent):
            if event.char == 'r':
                self.render_toggled = not self.render_toggled
            elif event.char == '1':
                self.debug_menu = (self.debug_menu + 1) % 2
        elif isinstance(event, ResetEvent):
            self.reset()
        elif isinstance(event, MouseEvent):
            self.check_mouse_input(event.click_pos)

        self.dt = self.clock.tick(RENDER_FPS) / 1000
        # print(self.dt)

    def render_menu(self):
        pass

    def check_mouse_input(self, mouse_pos):
        if not DEBUG or not D_SELECTABLE_OBJECTS:
            self.tracking_patch_index = None
            self.tracking_object = None
            return

        mouse_pos = pygame.Vector2(mouse_pos)
        screen_size = pygame.Vector2(self.screen.get_size())

        normalized_mouse = (mouse_pos.elementwise() / screen_size)
        normalized_mouse.x *= SCREEN_WIDTH
        normalized_mouse.y *= SCREEN_HEIGHT

        normalized_mouse = self.scale_to_meters(normalized_mouse.copy(), zoomed=True)

        new_tracking_object = None

        for car in self.sim.vehicle_list:
            rect, angle = car.get_info()
            if self.is_mouse_on_rect(normalized_mouse, rect):
                new_tracking_object = car
                break

        for index, patch in enumerate(self.sim.agent_list[0].state_representation.ier_patches):
            if self.is_mouse_on_rect(normalized_mouse, patch.polygon):
                self.tracking_patch_index = index
                break

        for pedestrian in self.sim.pedestrian_list:
            if self.is_mouse_on_rect(normalized_mouse, pedestrian.rect):
                new_tracking_object = pedestrian
                break

        if new_tracking_object:
            self.debug_menu = DebugMenu.OBJECT

            self.tracking_object = new_tracking_object

    def get_tracking_info(self):
        obj = self.tracking_object
        info_text = ""
        if obj is None:
            info_text = "No object tracked. Click on an object to track it."
            return info_text

        attributes = []
        if isinstance(obj, Vehicle):
            attributes = ["x", "v", "a", "length", "width", "stop", "old_angle", "angle", "old_pos", "pos",
                          "direction_vector", "leader_distance", "inter_distance", "pedestrian_distance", "wait_time"]
            if isinstance(obj, Agent):
                info_text = f"Tracking Agent {obj.__class__}\n  \n"
                attributes += ["look_back", "action_acceleration", "view_distance", "coming_from", "agent_waiting_time"]
            elif isinstance(obj, IDMVehicle):
                info_text = "Tracking IDM Vehicle \n  \n"
                attributes += ["s_0", "v_0", "delta", "T", "a_0", "b_0", "max_wait_time"]

            info_text += f"current_turn: {obj.route.current_turn} \n"
            info_text += f"next_turn: {obj.route.next_turn} \n"
            info_text += f"next_inter_turn: {obj.route.next_inter_turn} \n"

        elif isinstance(obj, Pedestrian):
            info_text = "Tracking Pedestrian \n  \n"
            attributes = ["v", "size", "lookahead", "grid_pos", "direction_offset", "route_look_back", "state",
                          "cross_timer", "pos"]
        obj_dict = vars(obj)
        info_text += f"km/h: {round(get_kilometers_per_hour(obj_dict['v']), 3)} \n"

        for attribute in attributes:
            value = obj_dict[attribute]

            if isinstance(value, float):
                value = round(value, 3)
            elif isinstance(value, Point):
                value = "Point " + str((round(value.x, 3), round(value.y, 3)))

            info_text += f"{attribute}: {value} \n"

        if obj and self.tracking_patch_index:
            info_text += f"\n Tracking Patch {self.tracking_patch_index} \n \n"

            patch = self.sim.agent_list[0].state_representation.ier_patches[self.tracking_patch_index]
            patch_dict = vars(patch)
            for key in patch_dict:
                if key != "polygon":
                    info_text += f"{key}: {patch_dict[key]} \n"
        else:
            self.tracking_patch_index = None

        return info_text

    def is_mouse_on_rect(self, mouse_pos, rect):
        return rect.contains(Point(mouse_pos))

    def check_key_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]:
            self.view_port_rect.y += (12 / self.zoom) * 60 * self.dt
        if keys[pygame.K_UP]:
            self.view_port_rect.y -= (12 / self.zoom) * 60 * self.dt
        if keys[pygame.K_LEFT]:
            self.view_port_rect.x -= (12 / self.zoom) * 60 * self.dt
        if keys[pygame.K_RIGHT]:
            self.view_port_rect.x += (12 / self.zoom) * 60 * self.dt
        if keys[pygame.K_s]:
            if not self.model.debug_control:
                if self.zoom > 0.025:
                    self.zoom -= 0.01 * 60 * self.dt
                    self.zoom = max(self.zoom, 0.025)
            else:
                self.sim.agent_list[0].set_action(4)
        elif keys[pygame.K_w]:
            if not self.model.debug_control:
                if self.zoom < 10:
                    self.zoom += 0.01 * 60 * self.dt
            else:
                self.sim.agent_list[0].set_action(0)
        else:
            if self.model.debug_control:
                self.sim.agent_list[0].set_action(0)

        if self.model.pause:
            if keys[pygame.K_o]:
                self.model.step()

    def render_simulation(self):
        if S_CAMERA_FOLLOW:
            self.camera_follow(self.sim.vehicle_list[0])

        self.view_port_rect.size = pygame.Vector2(VIEWPORT_WIDTH, VIEWPORT_HEIGHT) / self.zoom

        self.view_bottom_right_grid = self.get_road_index_from_viewport_pos(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        self.view_top_left_grid = self.get_road_index_from_viewport_pos(0, 0)

        grid_surface = self.initial_grid_surface

        self.view_port_visibility_surface.fill((255, 255, 255))

        if self.zoom >= 0.25:
            temp_surface = pygame.Surface(self.view_port_rect.size)
            temp_surface.fill((0, 0, 0))
            temp_surface.blit(grid_surface, (0, 0), self.view_port_rect)
            if DEBUG and D_SHOW_VISIBILITY:
                temp_surface.blit(self.debug_grid_surface, (0, 0), self.view_port_rect)
                self.debug_grid_surface.fill((255, 255, 255))

            pygame.transform.scale(temp_surface, self.view_port_surface.get_size(), self.view_port_surface)
        else:
            self.view_port_surface.fill((0, 0, 0))
            temp_surface = pygame.transform.scale(self.get_closest_grid_surface(self.zoom), (
                grid_surface.get_width() * self.zoom, grid_surface.get_height() * self.zoom))

            self.view_port_surface.blit(temp_surface,
                                        (-self.view_port_rect.x * self.zoom, -self.view_port_rect.y * self.zoom))

        if DEBUG:
            for agent in self.sim.agent_list:
                self.draw_agent_info(agent)
                for point in agent.points:
                    self.draw_circle(point, Colors.RED, 5)
                for line in agent.lines:
                    self.draw_line(line, Colors.BLUE, width=4)

        if self.zoom >= 0.4:
            if not DEBUG or (DEBUG and D_SHOW_CARS):
                self.draw_cars()
            self.draw_pedestrians()

        if DEBUG:
            for agent in self.sim.agent_list:
                for polygon in agent.polygons:
                    self.draw_polygon(polygon, Colors.RED)

        if DEBUG and D_SHOW_VIEWPORT_SIZE:
            pygame.draw.rect(self.debug_grid_surface, Colors.RED, (
                self.view_bottom_right_grid[0] * ROAD_PIXEL_COUNT, self.view_bottom_right_grid[1] * ROAD_PIXEL_COUNT,
                ROAD_PIXEL_COUNT, ROAD_PIXEL_COUNT), width=4)
            pygame.draw.rect(self.debug_grid_surface, Colors.YELLOW,
                             (self.view_top_left_grid[0] * ROAD_PIXEL_COUNT,
                              self.view_top_left_grid[1] * ROAD_PIXEL_COUNT,
                              ROAD_PIXEL_COUNT, ROAD_PIXEL_COUNT),
                             width=4)

        if D_SHOW_VISIBILITY:
            self.view_port_surface.blit(self.view_port_visibility_surface, (0, 0))

        if DEBUG and D_SHOW_TEXT_INFO:
            self.draw_debug_menus()

        pygame.transform.scale(self.view_port_surface, pygame.display.get_surface().get_size(), self.screen)
        pygame.display.update()

        if self.video_recorder:
            if VIEWPORT_WIDTH != VIDEO_WIDTH or VIEWPORT_HEIGHT != VIDEO_HEIGHT:
                pygame.transform.scale(self.view_port_surface, (VIDEO_WIDTH, VIDEO_HEIGHT), self.video_surface)
                self.video_recorder.capture_frame(self.video_surface)
            else:
                self.video_recorder.capture_frame(self.view_port_surface)

        self.frame_counter += 1

    def draw_debug_menus(self):
        if self.debug_menu == DebugMenu.ENV:
            self.draw_debug_menu_env()
        elif self.debug_menu == DebugMenu.OBJECT:
            self.draw_debug_menu_object()

    def draw_debug_menu_object(self):
        agent = self.sim.agent_list[0]
        draw_texts = self.get_tracking_info().split("\n")

        x = 10
        y = 10
        for text in draw_texts:
            color = Colors.G_TRACKING_OBJECT
            if self.tracking_object == agent and text.startswith("action_") and agent.shield_interrupted:
                color = Colors.RED

            size = self.draw_text(text, color, (x, y))
            y += size[1] + 3

    def draw_debug_menu_env(self):
        draw_texts = ["Collisions: " + str(self.sim.collisions), "Time Remaining: " + str(self.sim.remaining_seconds),
                      "Last Set Seed: " + str(self.sim.last_seed)]

        env_dict = vars(self.sim.config)
        fresh_env_dict = vars(Config())
        for key in env_dict.keys():
            text = ""
            if key not in fresh_env_dict:
                text += "legacy_"
            text += f"env_{key}: {env_dict[key]}"
            draw_texts.append(text)

        x = 10
        y = 10
        for text in draw_texts:
            size = self.draw_text(text, Colors.WHITE, (x, y))
            y += size[1] + 3

    def draw_patch(self, patch, index):
        polygon = patch.polygon

        # self.draw_circle(polygon.exterior.coords[0], Colors.RED, radius=2)
        # self.draw_circle(polygon.exterior.coords[1], Colors.BLUE, radius=2)
        # self.draw_circle(polygon.exterior.coords[2], Colors.YELLOW, radius=2)
        # self.draw_circle(polygon.exterior.coords[3], Colors.LIME, radius=2)

        top_right = pygame.Vector2(polygon.exterior.coords[0])
        bottom_right = pygame.Vector2(polygon.exterior.coords[1])
        bottom_left = pygame.Vector2(polygon.exterior.coords[2])
        top_left = pygame.Vector2(polygon.exterior.coords[3])

        values = [patch.tto_other, patch.ttv_other, patch.tto_other_next, patch.tto_ego]
        if self.sim.config.state_include_intersection:
            values.append(patch.intersection)

        values.append(patch.priority)
        step_percent = 1.0 / float(len(values))
        percent = 0.0

        for value in values:
            new_top_right = top_right + (bottom_right - top_right) * percent
            new_bottom_right = top_right + (bottom_right - top_right) * (percent + step_percent)
            new_top_left = top_left + (bottom_left - top_left) * percent
            new_bottom_left = top_left + (bottom_left - top_left) * (percent + step_percent)

            value_polygon = Polygon(
                [new_top_right.xy, new_bottom_right.xy, new_bottom_left.xy, new_top_left.xy, new_top_right.xy])

            self.draw_polygon(value_polygon,
                              self.get_patch_color_from_value(normalized_value=patch.get_normalized(value)))

            percent += step_percent

        if index == self.tracking_patch_index:
            self.draw_polygon(polygon, Colors.G_TRACKING_OBJECT, width=4)
        else:
            self.draw_polygon(polygon, Colors.BLACK, width=1)

    def draw_agent_info(self, agent):
        if D_SHOW_VISIBILITY:

            if D_SHOW_VISIBILITY_GRID:
                visible_grid = agent.get_visible_grid(self.sim.grid)

                for index, visible_tile in np.ndenumerate(visible_grid):
                    grid_x, grid_y = visible_tile.grid_x, visible_tile.grid_y

                    pygame.draw.rect(self.debug_grid_surface, Colors.GREEN, (
                        grid_x * ROAD_PIXEL_COUNT, grid_y * ROAD_PIXEL_COUNT,
                        ROAD_PIXEL_COUNT, ROAD_PIXEL_COUNT), width=4)

            if D_FANCY_SHADOWS:
                for obstacle, polygon in agent.hidden_polygons_list:
                    if self.is_grid_object_visible(obstacle.grid_x, obstacle.grid_y):
                        self.draw_polygon(polygon, Colors.G_BACKGROUND, width=0,
                                          surface=self.view_port_visibility_surface)
            else:
                visible_polygons = agent.hidden_polygons
                if visible_polygons is not None:
                    if isinstance(visible_polygons, shapely.geometry.base.BaseMultipartGeometry):

                        for polygon in visible_polygons.geoms:
                            pass
                            self.draw_polygon(polygon, Colors.G_BACKGROUND, width=0,
                                              surface=self.view_port_visibility_surface)
                    else:
                        pass
                        self.draw_polygon(visible_polygons, Colors.G_BACKGROUND, width=0,
                                          surface=self.view_port_visibility_surface)

            if D_SHOW_VISIBILITY_RAYS:
                for index, ray_pair in enumerate(agent.visibility_ray_pairs):
                    for ray in ray_pair:
                        self.draw_line(ray, Colors.BLUE, width=2)

            # for index,hidden_polygon in enumerate(agent.hiden_polygons):
            # self.draw_polygon(hidden_polygon, Colors.RED)

        if D_SHOW_AGENT_ROUTE_LANES:
            for lane in agent.route.route_lanes:
                zoomed_coords = self.scale_to_px(lane.global_line).coords
                pygame.draw.lines(self.view_port_surface, Colors.RED, False, zoomed_coords[:],
                                  width=int(4 * self.zoom))
        if D_SHOW_AGENT_ROUTE:
            zoomed_coords = self.scale_to_px(agent.route_line).coords
            pygame.draw.lines(self.view_port_surface, Colors.YELLOW, False, zoomed_coords[:], width=int(4))
            # self.draw_polygon(agent.route_line.buffer((agent.width +1)/2, cap_style=2), Colors.RED, width=1)

        if D_SHOW_AGENT_PATCHES:
            for index, patch in enumerate(agent.state_representation.ier_patches):
                self.draw_patch(patch, index)

        if D_SHOW_INTERSECTING_LANES:
            current_road = agent.current_lane.get_road(self.sim.grid)
            if current_road.is_inter():
                current_lane = agent.current_lane
                for lane in current_road.intersecting_lanes[current_lane]:
                    zoomed_coords = self.scale_to_px(lane.global_line).coords
                    pygame.draw.lines(self.view_port_surface, Colors.RED, False, zoomed_coords[:],
                                      width=int(4 * self.zoom))

    def draw_pedestrians(self):
        view_bottom_right_grid = self.get_road_index_from_viewport_pos(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
        view_top_left_grid = self.get_road_index_from_viewport_pos(0, 0)

        for pedestrian in self.sim.pedestrian_list:

            grid_x, grid_y = pedestrian.grid_pos[0], pedestrian.grid_pos[1]

            visible = True
            if grid_x < view_top_left_grid[0] or grid_x > view_bottom_right_grid[0]:
                visible = False
            elif grid_y < view_top_left_grid[1] or grid_y > view_bottom_right_grid[1]:
                visible = False

            if visible:
                if DEBUG:
                    if D_SHOW_PEDESTRIAN_CROSSING_CHECK and pedestrian.cross_collision_area:
                        self.draw_polygon(pedestrian.cross_collision_area, Colors.BLUE, 4)
                    if D_SHOW_PEDESTRIAN_ROUTE and pedestrian.route_line:
                        self.draw_line(pedestrian.route_line, Colors.YELLOW, 4)
                color = Colors.G_PEDESTRIAN
                if pedestrian is self.tracking_object:
                    color = Colors.G_TRACKING_OBJECT
                self.draw_polygon(pedestrian.rect, color)

                if DEBUG and D_SHOW_MOVEMENT_DIRECTION:
                    front_left = get_vector(pedestrian.rect.exterior.coords[2])
                    front_right = get_vector(pedestrian.rect.exterior.coords[3])
                    back_left = get_vector(pedestrian.rect.exterior.coords[0])
                    back_right = get_vector(pedestrian.rect.exterior.coords[1])

                    front_middle = front_left + (front_right - front_left) / 2
                    back_middle = back_left + (back_right - back_left) / 2

                    center = front_middle + (back_middle - front_middle) / 2

                    direction_vector = get_vector(pedestrian.direction_offset) * pedestrian.size

                    line = shapely.geometry.LineString([center, center + direction_vector * 1.25])

                    adjusted_color = adjust_color_lightness(color, 1.75)

                    self.draw_circle((center[0], center[1]), adjusted_color, radius=int(2 * self.zoom))
                    self.draw_line(line, adjusted_color, width=int(2 * self.zoom))

    def is_grid_object_visible(self, grid_x, grid_y):
        visible = True
        if grid_x < self.view_top_left_grid[0] or grid_x > self.view_bottom_right_grid[0]:
            visible = False
        elif grid_y < self.view_top_left_grid[1] or grid_y > self.view_bottom_right_grid[1]:
            visible = False

        return visible

    def draw_cars(self):

        for index, car in enumerate(self.sim.vehicle_list):
            lane = car.current_lane
            grid_x, grid_y = lane.grid_x, lane.grid_y

            if self.is_grid_object_visible(grid_x, grid_y):
                if car.is_agent() or index == 0:
                    color = Colors.G_AGENT
                else:
                    color = Colors.G_CAR

                if car == self.tracking_object:
                    color = Colors.G_TRACKING_OBJECT
                    if isinstance(car, IDMVehicle):
                        self.draw_line(car.route.get_route_line(), Colors.YELLOW, 4)

                for agent in self.sim.agent_list:
                    pass
                    # if car is agent.leader:
                    #    color = Colors.GREEN
                self.draw_car(car, color)

    def draw_circle(self, point, color, radius):
        zoomed_point = self.scale_to_px(point)

        pygame.draw.circle(self.view_port_surface, color, zoomed_point, radius)

    def draw_text(self, text, color, pos):
        text_surface = self.font.render(text, fgcolor=color, bgcolor=Colors.BLACK)

        self.view_port_surface.blit(text_surface[0], pos)

        return text_surface[1]

    def draw_polygon(self, geometry, color, width=0, surface=None):
        if surface is None:
            surface = self.view_port_surface

        zoomed_coords = self.scale_to_px(geometry).exterior.coords
        if len(zoomed_coords) > 2:
            pygame.draw.polygon(surface, color, zoomed_coords, width=width)

    def draw_line(self, geometry, color, width=0):
        zoomed_coords = self.scale_to_px(geometry).coords
        pygame.draw.lines(self.view_port_surface, color, False, zoomed_coords, width=width)

    def draw_car(self, car, color):
        car_polygon = self.scale_to_px(car.get_info()[0])
        zoomed_coords = car_polygon.exterior.coords

        front_left = get_vector(zoomed_coords[2])
        front_right = get_vector(zoomed_coords[3])
        back_left = get_vector(zoomed_coords[0])
        back_right = get_vector(zoomed_coords[1])

        front_middle = front_left + (front_right - front_left) / 2
        back_middle = back_left + (back_right - back_left) / 2

        direction_vector = (front_middle - back_middle)

        car_center = front_middle - direction_vector * 0.5

        if D_FANCY_GRAPHICS:
            rounded_polygon_coords = car_polygon.buffer(-4 * self.zoom).buffer(4 * self.zoom).exterior.coords
            if len(rounded_polygon_coords) > 2:
                pygame.draw.polygon(self.view_port_surface, color, rounded_polygon_coords)
            else:
                pygame.draw.polygon(self.view_port_surface, color, zoomed_coords)
        else:
            if D_HOLLOW_CARS:
                pygame.draw.polygon(self.view_port_surface, color, zoomed_coords, width=int((4 * self.zoom)))
            else:
                pygame.draw.polygon(self.view_port_surface, color, zoomed_coords)

        if DEBUG and D_SHOW_MOVEMENT_DIRECTION:
            darkened_color = adjust_color_lightness(color, factor=0.5)
            pygame.draw.circle(self.view_port_surface, darkened_color, center=car_center, radius=int(3 * self.zoom))
            front_line = [front_middle + direction_vector * 0.15, car_center]
            pygame.draw.lines(self.view_port_surface, darkened_color, closed=False, points=front_line,
                              width=int(2 * self.zoom))

        road = car.current_lane.get_road(self.sim.grid)
        next_road = car.route.next_lane.get_road(self.sim.grid)

        left_coordinate = zoomed_coords[2]
        right_coordinate = zoomed_coords[3]
        coordinate = None
        if next_road.is_inter():
            if car.route.next_turn is Turn.LEFT:
                coordinate = left_coordinate
            elif car.route.next_turn is Turn.RIGHT:
                coordinate = right_coordinate

        if road.is_inter():
            if car.route.current_turn is Turn.LEFT:
                coordinate = left_coordinate
            elif car.route.current_turn is Turn.RIGHT:
                coordinate = right_coordinate

        if self.frame_counter % self.sim.config.fps / 2 < self.sim.config.fps / 4 and coordinate is not None:
            pygame.draw.circle(self.view_port_surface, Colors.G_INDICATOR, center=coordinate, radius=4 * self.zoom)

    def get_road_index_from_viewport_pos(self, view_x, view_y):
        m_x, m_y = self.scale_to_meters(view_x, view_y)
        grid_x, grid_y = m_x / (ROAD_TILE_COUNT * TILE_SIZE_METERS), m_y / (ROAD_TILE_COUNT * TILE_SIZE_METERS)
        return int(grid_x), int(grid_y)

    def scale_to_px(self, x, y=None, zoomed=True, axis=0):
        """
        Scales a geometric object to pixel size for drawing functions. Works with lists, shapely geometries, float values,
        tuples and pygame vectors.

        :param x: The geometry which should be scaled
        :param y: The second part of the coordinate, if a coordinate should be scaled. Can be None
        :param zoomed: If this flag is true, a zoom will be applied
        :param axis: Only used if y is None. Then this will tell, if the x axis or y axis is being used for the zoom calculation
        :return: The scaled geometry measured in pixels
        """
        ratio = TILE_SIZE_PX / TILE_SIZE_METERS
        if isinstance(x, list):
            return [self.scale_to_px(entry) for entry in x]
        elif isinstance(x, tuple):
            return self.scale_to_px(x[0], x[1])
        elif isinstance(x, shapely.geometry.base.BaseGeometry):
            geometry = affinity.scale(x, xfact=ratio, yfact=ratio,
                                      origin=(0, 0))
            if zoomed:
                geometry = affinity.translate(geometry, xoff=-self.view_port_rect.x, yoff=-self.view_port_rect.y)
                geometry = affinity.scale(geometry, self.zoom, self.zoom, origin=(0, 0))
            return geometry
        elif isinstance(x, pygame.Vector2):
            vector = ratio * x
            if zoomed:
                vector = ((vector - pygame.Vector2(self.view_port_rect.topleft)) * self.zoom)
            return vector
        elif isinstance(x, float) or isinstance(x, int):
            if y is None:
                value = ratio * x
                if zoomed:
                    if axis != 0 and axis != 1:
                        raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis)!")
                    value = (value - self.view_port_rect.topleft[axis]) * self.zoom
                return value
            else:
                x, y = ratio * x, ratio * y
                if zoomed:
                    x = (x - self.view_port_rect.topleft[0]) * self.zoom
                    y = (y - self.view_port_rect.topleft[1]) * self.zoom
                return x, y
        else:
            raise ValueError("This can not be scaled!")

    def scale_to_meters(self, x, y=None, zoomed=True, axis=0):
        """
        Scales a geometric object to meter size for simulation queries. Works with lists, shapely geometries, float values,
        tuples and pygame vectors.

        :param x: The geometry which should be scaled
        :param y: The second part of the coordinate, if a coordinate should be scaled. Can be None
        :param zoomed: If this flag is true, an inverse zoom will be applied
        :param axis: Only used if y is None. Then this will tell, if the x axis or y axis is being used for the zoom calculation
        :return: The scaled geometry measured in metres
        """
        ratio = TILE_SIZE_METERS / TILE_SIZE_PX
        if isinstance(x, list):
            return [self.scale_to_meters(entry) for entry in x]
        elif isinstance(x, tuple):
            return self.scale_to_meters(x[0], x[1])
        elif isinstance(x, shapely.geometry.base.BaseGeometry):
            geometry = x
            if zoomed:
                geometry = affinity.scale(x, xfact=1 / self.zoom, yfact=1 / self.zoom,
                                          origin=(0, 0))
                geometry = affinity.translate(geometry, xoff=self.view_port_rect.x, yoff=self.view_port_rect.y)
            geometry = affinity.scale(geometry, ratio, ratio, origin=(0, 0))
            return geometry
        elif isinstance(x, pygame.Vector2):
            vector = x
            if zoomed:
                vector = ((vector / self.zoom) + pygame.Vector2(self.view_port_rect.topleft))
            vector = ratio * vector
            return vector
        elif isinstance(x, float) or isinstance(x, int):
            if y is None:
                value = x
                if zoomed:
                    if axis != 0 and axis != 1:
                        raise ValueError("Axis must be 0 (x-axis) or 1 (y-axis)!")
                    value = (value / self.zoom) + self.view_port_rect.topleft[axis]
                value = ratio * value
                return value
            else:
                if zoomed:
                    x = (x / self.zoom) + self.view_port_rect.topleft[0]
                    y = (y / self.zoom) + self.view_port_rect.topleft[1]
                return x * ratio, y * ratio
        else:
            raise ValueError("This can not be scaled!")

    def camera_follow(self, car):
        viewport_vector = pygame.Vector2(self.view_port_rect.center)

        if S_CAMERA_OFFSET:
            route_line = car.route.get_route_line()
            distance = min(route_line.length, 40)
            pos = route_line.interpolate(distance / self.zoom, normalized=False)
        else:
            pos = car.pos
        pos = self.scale_to_px(pos, zoomed=False)

        pos_vector = pygame.Vector2(pos.x, pos.y)

        if S_SMOOTH_CAMERA:
            difference_vector = ((pos_vector - viewport_vector) * min(4 * self.dt, 1.0))

            if difference_vector.length() > 0.05:
                viewport_vector = viewport_vector.lerp(difference_vector + viewport_vector, 0.75)
        else:
            viewport_vector = pos_vector

        self.view_port_rect.center = (viewport_vector.x, viewport_vector.y)

    def get_grid_object_surface(self, grid_object: GridObject) -> pygame.Surface:
        surface = pygame.Surface((ROAD_PIXEL_COUNT, ROAD_PIXEL_COUNT))
        surface.fill(Colors.G_BACKGROUND)

        # Draw basic tiles
        for index, tile in np.ndenumerate(grid_object.tiles):
            x, y = index
            draw_x = x * TILE_SIZE_PX
            draw_y = y * TILE_SIZE_PX
            color = Colors.G_BACKGROUND
            if tile == ATiles.NOTHING:
                color = Colors.G_BACKGROUND
            elif tile == ATiles.ROAD_F or tile == ATiles.ROAD_B:
                color = Colors.G_ROAD
            elif tile == ATiles.WALL_F or tile == ATiles.WALL_B:
                color = Colors.G_PATH
            elif tile == ATiles.OBSTACLE:
                color = Colors.G_OBSTACLE

            grid_indicator = None
            if tile == ATiles.WALL_F or tile == ATiles.ROAD_F:
                grid_indicator = Colors.G_LANE_F
            elif tile == ATiles.WALL_B or tile == ATiles.ROAD_B:
                grid_indicator = Colors.G_LANE_B

            pygame.draw.rect(surface, color, (draw_x, draw_y, TILE_SIZE_PX, TILE_SIZE_PX))

            if DEBUG and D_SHOW_GRID:
                if grid_indicator is not None:
                    pygame.draw.rect(surface, grid_indicator, (draw_x, draw_y, TILE_SIZE_PX, TILE_SIZE_PX), width=1)

        if grid_object.is_road():
            for markings in grid_object.lane_markings:
                coords = self.scale_to_px(markings, zoomed=False).coords
                pygame.draw.lines(surface, Colors.WHITE, False, coords[:], width=4)
            if grid_object.is_inter():
                if DEBUG and D_SHOW_INTER_START_LINES:
                    for inter_start in grid_object.local_inter_start_lines:
                        coords = self.scale_to_px(inter_start, zoomed=False).coords
                        pygame.draw.lines(surface, Colors.WHITE, False, coords[:], width=4)
                    for inter_end in grid_object.local_inter_end_lines:
                        coords = self.scale_to_px(inter_end, zoomed=False).coords
                        pygame.draw.lines(surface, Colors.BLUE, False, coords[:], width=4)

        if DEBUG:
            if D_SHOW_GRID:
                self.draw_text_on_grid_object(surface, grid_object.name, (ROAD_PIXEL_COUNT / 2, ROAD_PIXEL_COUNT / 2),
                                              Colors.LIGHT_BLUE)
                self.draw_text_on_grid_object(surface, f"({str(grid_object.grid_x)}, {str(grid_object.grid_y)})",
                                              (ROAD_PIXEL_COUNT / 2, ROAD_PIXEL_COUNT / 2 + 40),
                                              Colors.LIGHT_BLUE)
            if grid_object.is_road():
                lanes = grid_object.lanes

                for key in lanes.keys():
                    for lane in lanes[key]:
                        coords = self.scale_to_px(lane.local_line, zoomed=False).coords

                        if D_SHOW_LANES:
                            if LaneSides.FORWARD == lane.lane_side:
                                pygame.draw.lines(surface, Colors.G_LANE_F, False, coords[:], width=2)
                            else:
                                pygame.draw.lines(surface, Colors.G_LANE_B, False, coords[:], width=2)

                        if D_SHOW_ENTRY_POINTS:
                            for info in [(0, Colors.WHITE, lane.entry_direction, lane.turn),
                                         (-1, Colors.RED, lane.exit_direction, lane.turn)]:
                                coord_index, color, direction, turn = info
                                self.draw_text_on_grid_object(surface, Directions.get_name(direction),
                                                              coords[coord_index], color)

        if grid_object.is_obstacle():
            pygame.draw.rect(surface, Colors.G_OBSTACLE_OUTLINE, (0, 0, ROAD_PIXEL_COUNT, ROAD_PIXEL_COUNT), width=8)

        return surface

    def draw_text_on_grid_object(self, surface, text, position, color):
        text_surface = self.small_font.render(text, False,
                                              color)
        position = pygame.Vector2(position)
        rect_size = pygame.Vector2(text_surface.get_rect().size)
        if position.x + rect_size.x > TILE_SIZE_PX:
            position.x -= rect_size.x
        if position.y + rect_size.y > TILE_SIZE_PX:
            position.y -= rect_size.y
        position.x = max(0.0, position.x)
        position.y = max(0.0, position.y)
        surface.blit(text_surface, position)

    @staticmethod
    def get_unique_path(path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + " (" + str(counter) + ")" + extension
            counter += 1

        return path

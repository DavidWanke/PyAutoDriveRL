"""
The constants for the view.
"""

"""
Determines size of screen.
"""
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

"""
Determines size of viewport.
"""
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
"""
Determines target framerate for rendering the scene.
"""
RENDER_FPS = 24

S_CAMERA_FOLLOW = True
S_CAMERA_OFFSET = True
S_SMOOTH_CAMERA = False
S_VIDEO_USE_ENV_FPS = False
S_RECORD_VIDEO = True

"""
DEBUG SETTINGS
"""
DEBUG = True
D_SHOW_ENTRY_POINTS = False
D_SHOW_LANES = False
D_SHOW_VIEWPORT_SIZE = False
D_SHOW_GRID = False
D_FANCY_GRAPHICS = False
D_FANCY_SHADOWS = False
D_SHOW_CARS = True
D_HOLLOW_CARS = False
D_SHOW_INTERSECTING_LANES = False
D_SAVE_ENV_IMAGE = False
D_SHOW_AGENT_ROUTE = False
D_SHOW_AGENT_ROUTE_LANES = False
D_SHOW_AGENT_PATCHES = True
D_SHOW_VISIBILITY = True
D_SHOW_VISIBILITY_RAYS = False
D_SHOW_VISIBILITY_GRID = False
D_SHOW_PEDESTRIAN_CROSSING_CHECK = False
D_SHOW_PEDESTRIAN_ROUTE = False
D_FORCE_SEED = None
D_SHOW_INTER_START_LINES = False
D_SHOW_TEXT_INFO = True
D_SHOW_MOVEMENT_DIRECTION = True
D_SELECTABLE_OBJECTS = True


class Colors:
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    RED = (255, 0, 0)
    GREEN = (22, 156, 2)
    LIME = (119, 255, 15)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (66, 132, 245)
    BLACK = (0, 0, 0)
    GRAY = (127, 127, 127)
    WHITE = (255, 255, 255)

    G_BACKGROUND = (20, 20, 20)  # The background color of the scene
    G_ROAD = (48, 48, 48)  # The road color
    G_MARKINGS = (255, 255, 255)  # The color of the road markings
    G_PATH = (87, 85, 81)  # The color of the pathways around the roads
    G_LANE_F = LIME
    G_LANE_B = BLUE
    G_CAR = (54, 141, 255)
    G_AGENT = (252, 73, 94)
    G_INDICATOR = (255, 208, 54)
    G_OBSTACLE = (81, 94, 130)
    G_OBSTACLE_OUTLINE = (102, 115, 150)
    G_PEDESTRIAN = (94, 196, 35)
    G_TRACKING_OBJECT = WHITE

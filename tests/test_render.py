import math
from macad_gym.viz.render import Render

def test_resize():
    """Test resize_screen() and get_screen()
    """
    resX, resY = 800, 600
    Render.resize_screen(resX, resY)

    # The screen should be None before calling get_screen() at the first time
    assert Render._screen is None
    assert Render._update_size is True

    screen = Render.get_screen()

    # get_screen will update _screen and _update_size
    assert screen.get_size() == (resX, resY)
    assert Render._update_size is False

def test_window_size_1():
    """Test window size when subwindow's max width
    smaller than main window's width
    """
    resX, resY = 640, 480

    # This is the main image, representing manual_control view
    Render.resize_screen(resX, resY)

    # This is the sub image, representing the view of the agent
    subwindow_num = 4
    unit_dimension = [84, 84]   # 84 * 4 = 336 < 640

    # Generate subwindow_num of pseudo actor_config
    actor_config = {i: {"render": True} for i in range(subwindow_num)}

    _, window_dim = Render.get_surface_poses(unit_dimension, actor_config)

    # The window_dim should be [336, 84], i.e. 1 row
    assert window_dim == [336, 84]


def test_window_size_2():
    """Test window size when subwindow's max width
    larger than main window's width
    """
    resX, resY = 240, 120

    # This is the main image, representing manual_control view
    Render.resize_screen(resX, resY)

    # This is the sub image, representing the view of the agent
    subwindow_num = 4
    unit_dimension = [84, 84]   # 84 * 4 = 336 > 240

    # Generate subwindow_num of pseudo actor_config
    actor_config = {i: {"render": True} for i in range(subwindow_num)}

    _, window_dim = Render.get_surface_poses(unit_dimension, actor_config)

    rows = math.ceil(math.sqrt(subwindow_num))

    # The window_dim should be [168, 168], i.e. 2 row
    assert window_dim == [unit_dimension[0] * rows, unit_dimension[1] * rows]


def test_window_size_3():
    """Test window size when no subwindow
    """
    resX, resY = 640, 480

    # This is the main image, representing manual_control view
    Render.resize_screen(resX, resY)

    # This is the sub image, representing the view of the agent
    unit_dimension = [84, 84]   # 84 * 4 = 336 > 240

    _, window_dim = Render.get_surface_poses(unit_dimension, {})

    # The window_dim should be [0, 0]
    assert window_dim == [0, 0]

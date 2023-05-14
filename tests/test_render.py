import math
from carla_gym.core.utils.multi_view_renderer import MultiViewRenderer as Render
from core.utils.scenario_config import ActorConfiguration


def test_resize():
    """Test resize_screen() and get_screen()."""
    resX, resY = 800, 600
    r = Render(resX, resY)
    r.resize_screen(resX + 10, resY + 10)

    # The screen should be None before calling get_screen() at the first time
    assert r._screen is None
    assert r._update_size is True

    screen = r.get_screen()

    # get_screen will update _screen and _update_size
    assert screen.get_size() == (resX + 10, resY + 10)
    assert r._update_size is False


def test_window_size_0():
    """Test window size with no subwindows"""
    resX, resY = 640, 480

    # This is the main image, representing manual_control view
    r = Render(resX, resY)

    # This is the sub image, representing the view of the agent
    unit_dimension = [84, 84]  # 84 * 4 = 336 > 240

    _, window_dim = r.set_surface_poses(unit_dimension, {})

    # The window_dim should be [0, 0]
    assert window_dim == [0, 0]


def test_window_size_1():
    """Test window size when subwindow's max width smaller than main window's width."""
    resX, resY = 640, 480

    # This is the main image, representing manual_control view
    r = Render(resX, resY)

    # This is the sub image, representing the view of the agent
    subwindow_num = 4
    unit_dimension = [84, 84]  # 84 * 4 = 336 < 640
    # Generate subwindow_num of pseudo actor_config
    a = ActorConfiguration("_", render=True)
    actor_config = {i: a for i in range(subwindow_num)}

    _, window_dim = r.set_surface_poses(unit_dimension, actor_config)

    # The window_dim should be [336, 84], i.e. 1 row
    assert window_dim == [336, 84]


def test_window_size_2():
    """Test window size when subwindow's max width larger than main window's width."""
    resX, resY = 240, 120

    # This is the main image, representing manual_control view
    r = Render(resX, resY)

    # This is the sub image, representing the view of the agent
    subwindow_num = 4
    unit_dimension = [84, 84]  # 84 * 4 = 336 > 240

    # Generate subwindow_num of pseudo actor_config
    a = ActorConfiguration("_", render=True)
    actor_config = {i: a for i in range(subwindow_num)}

    _, window_dim = r.set_surface_poses(unit_dimension, actor_config)

    rows = math.ceil(math.sqrt(subwindow_num))

    # The window_dim should be [168, 168], i.e. 2 row
    assert window_dim == [unit_dimension[0] * rows, unit_dimension[1] * rows]

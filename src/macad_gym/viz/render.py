import math

import cv2
import pygame

i = 0
pygame.init()
pygame.display.set_caption("MACAD-Gym")


def multi_view_render(images, unit_dimension, actor_configs):
    """Render images based on pygame > 1.9.4

    Args:
        images (dict):
        unit_dimension (list): window size, e.g., [84, 84]
        actor_configs (dict): configs of actors

    Returns:
        N/A.
    """
    global i

    surface_seq = ()
    poses, window_dim = get_surface_poses(len(images), unit_dimension, images.keys())

    # Get all surfaces.
    for actor_id, im in images.items():
        if not actor_configs[actor_id]["render"]:
            continue
        surface = pygame.surfarray.make_surface(im)
        surface_seq += ((surface, (poses[actor_id][1], poses[actor_id][0])),)

    multi_surface = pygame.display.set_mode(
        (window_dim[0], window_dim[1]), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    multi_surface.blits(blit_sequence=surface_seq, doreturn=1)
    pygame.display.flip()
    # save to disk
    # pygame.image.save(display,
    #                   "/mnt/DATADRIVE1/pygame_surfs/" + str(i) + ".jpeg")
    i += 1


def get_surface_poses(subwindow_num, unit_dimension, ids):
    """Calculate the poses of sub-windows of actors

    Args:
        subwindow_num (int): number of sub-windows(actors)
        window_dimension (list): E.g., [800, 600]
        ids (list): list of actor ids.

    Returns:
        dict: return position dicts in pygame window (start from left corner).
        E.g., {"vehiclie":[0,0]}
    """
    num = subwindow_num
    unit_x = unit_dimension[0]
    unit_y = unit_dimension[1]

    row_num = math.ceil(math.sqrt(num))
    max_x = row_num * unit_x
    max_y = row_num * unit_y

    poses = {}

    for i, id in enumerate(ids):
        x_pos = math.floor(i / row_num) * unit_x
        y_pos = math.floor(i % row_num) * unit_y
        poses[id] = [x_pos, y_pos]

    return poses, [max_x, max_y]

# noqa
import math

import pygame
from core.controllers.manual_controller import MANUAL_VIEW_RENDER_X, MANUAL_VIEW_RENDER_Y

pygame.init()
pygame.display.set_caption("MACAD-Gym")


class MultiViewRenderer:
    """Handle rendering of multiple camera sensors in a single PyGame window."""

    def __init__(self, screen_width=640, screen_height=480):
        """Constructor.

        Args:
            screen_width: maximum width value of the screen allowed
            screen_height: maximum height value of the screen allowed
        """
        self.width = screen_width
        self.height = screen_height
        self.poses = {}
        self._update_size = False
        self._screen = None
        self.save_counter = 0

    def reset_frame_counter(self):
        """Set the internal frame counter to 0.

        This can cause the overwrite of images saved in the future.
        """
        self.save_counter = 0

    def resize_screen(self, screen_width, screen_height):
        """Set a new size for the screen that will be used in the first new rendering."""
        self.width = screen_width
        self.height = screen_height
        self._update_size = True

    def get_screen(self):
        """Retrieve PyGame screen following internal class sizes.

        Returns:
            PyGame screen instance.
        """
        if self._screen is None or self._update_size:
            self._screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self._update_size = False

        return self._screen

    def set_surface_poses(self, unit_dimension, actor_configs):
        """Calculate the poses of sub-windows of actors.

        Args:
            unit_dimension (list): size of each view, E.g., [84, 84]
            actor_configs (dict): dict of actor configs.

        Returns:
            poses (dict): return position dicts in pygame window (start from left corner). E.g., {"vehicle":[0,0]}
            window_dim (list): return the max [width, height] needed to render all actors.
        """
        unit_x = unit_dimension[0]
        unit_y = unit_dimension[1]

        subwindow_num = 0
        manual_control_view = False
        for _, config in actor_configs.items():
            if config.render:
                subwindow_num += 1
            if config.manual_control:
                manual_control_view = True

        if subwindow_num == 0:
            return {}, [0, 0]

        if unit_dimension[0] * subwindow_num > self.width:
            row_num = math.ceil(math.sqrt(subwindow_num))
            max_x = row_num * unit_x
        else:
            row_num = 1
            max_x = subwindow_num * unit_x
        max_y = row_num * unit_y

        self.poses = {}
        if manual_control_view:
            self.poses["manual"] = [0, max_y]
            max_x = max(max_x, MANUAL_VIEW_RENDER_X)
            max_y = max_y + MANUAL_VIEW_RENDER_Y

        for i, a in enumerate(actor_configs.items()):
            id, config = a
            if not config.render and not config.manual_control:
                continue

            x_pos = math.floor(i / row_num) * unit_x
            y_pos = math.floor(i % row_num) * unit_y
            self.poses[id] = [x_pos, y_pos]

        self.resize_screen(max_x, max_y)

        return max_x, max_y

    def render(self, images, enable_save=False):
        """Render multiple views of actors.

        Args:
            images (dict): e.g. {"vehicle": ndarray | pygame.Surface}
            poses (dict): {"vehicle": [0,0]}
            enable_save (bool): whether to save the rendered image to disk

        Returns:
            N/A.
        """
        # Get all surfaces.
        surface_seq = ()
        for id, im in images.items():
            if isinstance(im, pygame.Surface):
                surface = im
            else:
                surface = pygame.surfarray.make_surface(im.swapaxes(0, 1))
            surface_seq += ((surface, (self.poses[id][0], self.poses[id][1])),)

        self.get_screen().blits(blit_sequence=surface_seq, doreturn=1)
        pygame.display.flip()

        # Save to disk
        if enable_save:
            for surf, pos in surface_seq:
                pygame.image.save(surf, f"./carla_out/{self.save_counter}_{pos[0]}_{pos[1]}.png")
            self.save_counter += 1

    @staticmethod
    def window_event_handler():
        """Dummy event handler.

        This is needed to make pygame window responsive.

        This function will clean all events in the event queue,
        so it should only be called when manual_control is off.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

import math
import pygame

pygame.init()
pygame.display.set_caption("MACAD-Gym")


class Render:
    """Handle rendering of pygame window."""

    _screen = None
    _update_size = False
    resX = 640
    resY = 480

    save_cnt = 0

    @staticmethod
    def reset_frame_cnt():
        Render.save_cnt = 0

    @staticmethod
    def resize_screen(resX, resY):
        """Resize the screen.

        This call doesn't take effect until Render.get_screen() is called.
        """
        Render.resX = resX
        Render.resY = resY
        Render._update_size = True

    @staticmethod
    def get_screen():
        if Render._screen is None or Render._update_size:
            Render._screen = pygame.display.set_mode(
                (Render.resX, Render.resY), pygame.HWSURFACE | pygame.DOUBLEBUF)
            Render._update_size = False

        return Render._screen

    @staticmethod
    def draw(image, render_pose=(0, 0)):
        """Draw the image on the screen.

        Args:
            image (pygame.Surface | numpy.array | list): image to be drawn
            render_pose (tuple): position of the image on the screen (left corner)

        Returns:
            N/A.
        """
        screen = Render.get_screen()

        if isinstance(image, pygame.Surface):
            screen.blit(image, render_pose)
        else:
            surface = pygame.surfarray.make_surface(image)
            screen.blit(surface, render_pose)

        pygame.display.flip()

    @staticmethod
    def get_surface_poses(unit_dimension, actor_configs):
        """Calculate the poses of sub-windows of actors

        Args:
            unit_dimension (list): size of each view, E.g., [84, 84]
            actor_configs (dict): dict of actor configs.

        Returns:
            poses (dict): return position dicts in pygame window (start from left corner). E.g., {"vehiclie":[0,0]}
            window_dim (list): return the max [width, height] needed to render all actors.
        """
        unit_x = unit_dimension[0]
        unit_y = unit_dimension[1]

        subwindow_num = 0
        for _, config in actor_configs.items():
            if config["render"] == True:
                subwindow_num += 1

        if subwindow_num == 0:
            return {}, [0, 0]

        if unit_dimension[0] * subwindow_num > Render.resX:
            row_num = math.ceil(math.sqrt(subwindow_num))
            max_x = row_num * unit_x
        else:
            row_num = 1
            max_x = subwindow_num*unit_x

        max_y = row_num * unit_y

        poses = {}

        i = 0
        for id, config in actor_configs.items():
            if config["render"] == False:
                continue

            x_pos = math.floor(i / row_num) * unit_x
            y_pos = math.floor(i % row_num) * unit_y
            poses[id] = [x_pos, y_pos]
            i += 1

        return poses, [max_x, max_y]

    @staticmethod
    def multi_view_render(images, poses, enable_save=False):
        """Render multiple views of actors.

        Args:
            images (dict): e.g. {"vehicle": ndarray | pygame.Surface}
            poses (dict): {"vehicle": [0,0]}
            enable_save (bool): whether to save the rendered image to disk

        Returns:
            N/A.
        """
        surface_seq = ()

        # Get all surfaces.
        for actor_id, im in images.items():
            if isinstance(im, pygame.Surface):
                surface = im
            else:
                surface = pygame.surfarray.make_surface(im)

            surface_seq += ((surface,
                            (poses[actor_id][0], poses[actor_id][1])),)

        Render.get_screen().blits(blit_sequence=surface_seq, doreturn=1)
        pygame.display.flip()

        # save to disk
        if enable_save:
            for surf, pos in surface_seq:
                pygame.image.save(
                    surf, f"./carla_out/{Render.save_cnt}_{pos[0]}_{pos[1]}.png")

            Render.save_cnt += 1

    @staticmethod
    def dummy_event_handler():
        """Dummy event handler.

        This is needed to make pygame window responsive.

        This function will clean all events in the event queue,
        so it should only be called when manual_control is off.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

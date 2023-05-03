# noqa
import carla
from core.controllers.camera_manager import CAMERA_TYPES, CameraManager
from core.world_objects.hud import HUD

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

MANUAL_VIEW_RENDER_X = 600
MANUAL_VIEW_RENDER_Y = 600


class ManualController:
    """Manual controller attached to an actor object.

    Self-contained class inspired from carla library manual_control.py.
    It instantiates a keyboard handler, HUD and Camera objects to let the dev interact/observe the world.
    """

    def __init__(self, actor_obj, start_in_autopilot):
        """Constructor.

        Args:
            actor_obj: actor object
            start_in_autopilot: boolean flag specifying weather to start with autopilot enabled
        """
        self._vehicle = actor_obj
        self._autopilot_enabled = start_in_autopilot

        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._control_clock = pygame.time.Clock()
        pygame.font.init()  # for HUD
        self.hud = HUD(MANUAL_VIEW_RENDER_X, MANUAL_VIEW_RENDER_Y)
        self.camera_manager = CameraManager(actor_obj, render_dim=(MANUAL_VIEW_RENDER_X, MANUAL_VIEW_RENDER_X))
        self.camera_manager.set_sensor(CAMERA_TYPES["rgb"].value - 1, pos=2)

    def tick(self, fps, world, vehicle, collision_sensor):
        """Tick from the client. Ideally it should be called at each environment step."""
        # Sync with env evolution
        self._control_clock.tick(fps)
        # Update information in the hud
        self.hud.tick(world, vehicle, collision_sensor, self._control_clock)

    def parse_events(self):
        """Parse keyboard events."""
        self._vehicle.set_autopilot(self._autopilot_enabled)
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                elif event.type == pygame.KEYUP:
                    if self._is_quit_shortcut(event.key):
                        return True
                    # elif event.key == K_BACKSPACE:
                    #     world.restart()
                    elif event.key == K_F1:
                        self.hud.toggle_info()
                    elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                        self.hud.help.toggle()
                    elif event.key == K_TAB:
                        self.camera_manager.toggle_camera()
                    # elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    #     world.next_weather(reverse=True)
                    # elif event.key == K_c:
                    #     world.next_weather()
                    elif event.key == K_BACKQUOTE:
                        self.camera_manager.next_sensor()
                    elif event.key > K_0 and event.key <= K_9:
                        self.camera_manager.set_sensor(event.key - 1 - K_0)
                    elif event.key == K_r:
                        self.camera_manager.toggle_recording()
                    elif event.key == K_q:
                        self._control.reverse = not self._control.reverse
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        self._vehicle.set_autopilot(self._autopilot_enabled)
            if not self._autopilot_enabled:
                self._parse_keys(pygame.key.get_pressed(), self._control_clock.get_time())
                self._vehicle.apply_control(self._control)

    def _parse_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.0)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1.0)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_keys1(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_keys2(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def render(self, main_screen=None, render_pose=(0, 0)):
        """Render the internal camera view.

        If a screen is provided the image will be rendered in there in the specified position without triggering
        the screen pixel effective render update. If the screen is not provided the rendering process will be executed
        internally, obtaining a screen object and updating the screen pixels.
        Args:
            main_screen: (optional) PyGame screen object
            render_pose: position where to render the image in the screen

        Returns:
            N/A.
        """
        screen = main_screen if main_screen is not None else self.camera_manager.get_screen()
        self.camera_manager.render(screen, render_pose)
        self.hud.render(screen, render_pose)
        if main_screen is None:
            pygame.display.flip()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

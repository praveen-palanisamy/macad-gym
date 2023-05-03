"""Overlay interface Objects."""
import datetime
import logging
import math

import pygame

logger = logging.getLogger(__name__)


def get_actor_display_name(actor, truncate=250):
    """Prettified actor name.

    Args:
        actor: actor object
        truncate: maximum str output length.

    Returns:
        String representing the actor.
    """
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


class HUD:
    """Overlay Interface."""

    def __init__(self, width, height):
        """Constructor.

        Args:
            width: width size
            height: height size
        """
        self.dim = (width, height)
        # font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        #  the _notifications and help are not needed for multi_env,
        #   they depends on other classes.
        # self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_server_tick(self, timestamp):
        """Method to sync client with server."""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, vehicle, collision_sensor, clock):
        """Step the world objects with provided clock updating the information in overlay."""
        if not self._show_info:
            return

        t = vehicle.get_transform()
        v = vehicle.get_velocity()
        c = vehicle.get_control()

        heading = "N" if abs(t.rotation.yaw) < 89.5 else ""
        heading += "S" if abs(t.rotation.yaw) > 90.5 else ""
        heading += "E" if 179.5 > t.rotation.yaw > 0.5 else ""
        heading += "W" if -0.5 > t.rotation.yaw > -179.5 else ""
        colhist = collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.get_actors().filter("vehicle.*")
        self._info_text = [
            "Server:  % 16d FPS" % self.server_fps,
            "Client:  % 16d FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(vehicle, truncate=20),
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            "Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (t.rotation.yaw, heading),
            "Location:% 20s" % (f"({t.location.x: 5.1f}, {t.location.y: 5.1f})"),
            "Height:  % 18.0f m" % t.location.z,
            "",
            ("Throttle:", c.throttle, 0.0, 1.0),
            ("Steer:", c.steer, -1.0, 1.0),
            ("Brake:", c.brake, 0.0, 1.0),
            ("Reverse:", c.reverse),
            ("Hand brake:", c.hand_brake),
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            # distance = lambda l: math.sqrt((l.x - t.location.x)**2 +
            #                               (l.y - t.location.y)**2 +
            #                               (l.z - t.location.z)**2)
            vehicles = [(self.distance(x.get_location(), t), x) for x in vehicles if x.id != vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def distance(self, l, t):
        """Get distance between two points."""
        return math.sqrt((l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)

    def toggle_info(self):
        """Toggle for the overlay information layer."""
        self._show_info = not self._show_info

    # def notification(self, text, seconds=2.0):
    #     logger.info("Notification disabled: " + text)
    #     # self._notifications.set_text(text, seconds=seconds)

    # def error(self, text):
    #     logger.info("Notification error disabled: " + text)
    #     # self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, screen, render_pose=(0, 0)):
        """Render method."""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            screen.blit(info_surface, render_pose)
            v_offset = 4 + render_pose[1]
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(screen, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(screen, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(screen, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(screen, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    screen.blit(surface, (8, v_offset))
                v_offset += 18
        self.help.render(screen, render_pose)


class HelpText:
    """Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change camera position
    `            : next camera sensor
    [1-9]        : change to camera sensor [1-9]

    R            : toggle recording images to disk

    H/?          : toggle help
    ESC          : quit
    """

    def __init__(self, font, width, height):
        """Constructor for the overlay help interface.

        Args:
            font: font type
            width: width of the screen
            height: height of the screen
        """
        lines = self.__doc__.split("\n")
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle for the overlay help layer."""
        self._render = not self._render

    def render(self, display, render_pose):
        """Render method."""
        if self._render:
            display.blit(self.surface, (self.pos[0] + render_pose[0], self.pos[1] + render_pose[1]))

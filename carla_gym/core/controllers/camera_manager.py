import os
import weakref
from enum import Enum

import carla
import numpy as np
import pygame

CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

CAMERA_TYPES = Enum("CameraType", ["rgb", "depth", "depth_raw", "semseg", "semseg_raw"])

DEPTH_CAMERAS = ["depth", "depth_raw"]


class CameraManager:
    """Controller for camera objects."""

    def __init__(self, parent_actor, render_dim, record=False):
        """Constructor.

        Args:
            parent_actor: actor configuration
            render_dim: world object
            record: flag if the images should saved
        """
        self._parent = parent_actor
        self._render_dim = render_dim
        self.sensor = None
        self.image = None  # need image to encode obs.
        self.image_list = []  # for save images later.
        self._surface = None
        self._img_array = None
        self._recording = record
        self._screen = None
        self._buffered_recording = False
        # supported through toggle_camera
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.8, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(
                carla.Location(
                    x=-2.0*(0.5 + self._parent.bounding_box.extent.x),
                    y=0.0,
                    z=2.0*(0.5 + self._parent.bounding_box.extent.z)
                ), carla.Rotation(pitch=8.0)
            )
        ]
        # 0 is dashcam view; 1 is tethered view; 2 for spring arm view (manual_control)
        self._transform_index = 0
        self._sensors = [
            ["sensor.camera.rgb", carla.ColorConverter.Raw, "Camera RGB"],
            ["sensor.camera.depth", carla.ColorConverter.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", carla.ColorConverter.Depth, "Camera Depth (Gray Scale)"],
            ["sensor.camera.depth", carla.ColorConverter.LogarithmicDepth, "Camera Depth (Logarithmic Gray Scale)"],
            ["sensor.camera.semantic_segmentation", carla.ColorConverter.Raw, "Camera Semantic Segmentation (Raw)"],
            ["sensor.camera.semantic_segmentation", carla.ColorConverter.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)"],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(render_dim[0]))
                bp.set_attribute("image_size_y", str(render_dim[1]))
            item.append(bp)
        self._index = None
        self.callback_count = 0

    @property
    def surface(self):
        return self._surface

    @property
    def img_array(self):
        return self._img_array

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, pos=0):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else list(CAMERA_TYPES)[index] != list(CAMERA_TYPES)[self._index]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self._transform_index = pos % len(self._camera_transforms)
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent,
                attachment_type=carla.AttachmentType.Rigid if pos != 2 else carla.AttachmentType.SpringArm)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        """Toggle for the internal images recording flag.

        Returns:
            N/A.
        """
        self._recording = not self._recording
        print("Recording %s" % ("On" if self._recording else "Off"))

    def get_screen(self):
        """Retrieve PyGame screen following internal class sizes.

        Returns:
            PyGame screen instance.
        """
        if self._screen is None:
            self._screen = pygame.display.set_mode(self._render_dim, pygame.HWSURFACE | pygame.DOUBLEBUF)

        return self._screen

    def render(self, main_screen=None, render_pose=(0, 0)):
        """Render in place, otherwise draw a surface without triggering the rendering if a screen is provided.

        Args:
            main_screen: pygame screen object.
            render_pose: tuple representing the coordinate where draw the internal surface on the screen.

        Returns:
            N/A.
        """
        screen = main_screen if main_screen is not None else self.get_screen()
        if self._surface is not None:
            screen.blit(self._surface, render_pose)
        if main_screen is None:
            pygame.display.flip()

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        self.image = image
        self.callback_count += 1
        if not self:
            return
        if self._sensors[self._index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._img_array = lidar_img
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._img_array = array
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image_dir = os.path.join(CARLA_OUT_PATH, 'images/{}/%04d.png'.format(self._parent.id) % image.frame_number)
            image.save_to_disk(image_dir)
        elif self._buffered_recording:
            self.image_list.append(image)
        else:
            pass

    def __del__(self):
        """Delete instantiated sub-elements."""
        if self.sensor is not None:
            self.sensor.destroy()

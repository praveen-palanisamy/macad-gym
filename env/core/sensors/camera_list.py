import os


class CameraList(object):
    """Maintains a camera list and functions on it.
    """

    def __init__(self, output_address):
        self.cam_list = {}
        self.out = output_address

    def save_images_to_disk(self):
        """Save images from memory.

        Returns:
            N/A: no returns. This func print images to folders.

        """

        # Save images from actors from their corresponding camera manager.
        for cam_manager in self.cam_list.values():
            if not cam_manager._memory_record:
                continue
            actor_id = cam_manager._parent.id
            for image in cam_manager.image_list:
                image_dir = os.path.join(
                    self.out,
                    'images/{}/%04d.png'.format(actor_id) % image.frame_number)
                image.save_to_disk(image_dir)  #, env.cc

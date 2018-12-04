import numpy as np
import pygame
import cv2
import sys
"""
# Defs _parse_image and get_image are not in use now they are used in vechile_manager now.
# However, will keep them for a while and then delete 
def _parse_image(env, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        env._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# not in use now
def get_image(env, image, i, cc):
        #image_dir = os.path.join(CARLA_OUT_PATH, 'images/{}/{}_%04d.png'.format(i,self.episode_id) % image.frame_number)
        #image.save_to_disk(image_dir, cc)
        env.image_pool[i].append(image)
        env.original_image = image
        _parse_image(env, image)  # py_game render use
        env.image = preprocess_image(env, image, i)
"""


def preprocess_image(env, image, i):
    config = env.config_list[str(i)]
    if config["use_depth_camera"]:
        assert config["use_depth_camera"]
        data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        data = np.reshape(data, (env.render_y_res, env.render_x_res, 4))
        data = data[:, :, :1]
        data = data[:, :, ::-1]
        data = cv2.resize(
            data, (env.x_res, env.y_res), interpolation=cv2.INTER_AREA)
        data = np.expand_dims(data, 2)
    else:
        data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        data = np.reshape(data, (env.render_y_res, env.render_x_res, 4))
        data = data[:, :, :3]
        data = data[:, :, ::-1]
        data = cv2.resize(
            data, (env.x_res, env.y_res), interpolation=cv2.INTER_AREA)
        data = (data.astype(np.float32) - 128) / 128

    return data


def get_transform_from_nearest_way_point(env, vehicle_id, pos):
    vehcile = env.actor_list[vehicle_id]
    way_points = env.cur_map.get_waypoint(vehcile.get_location())
    nexts = list(way_points.next(1.0))
    print('Next(1.0) --> %d waypoints' % len(nexts))
    if not nexts:
        raise RuntimeError("No more waypoints!")
    smallest_dist = sys.maxsize
    for p in nexts:
        trans = p.transform.location
        diff_x = trans.x - pos[vehicle_id][0]
        diff_y = trans.y - pos[vehicle_id][1]
        diff_z = trans.z - pos[vehicle_id][2]
        cur_dist = np.linalg.norm([diff_x, diff_y, diff_z])
        if cur_dist < smallest_dist:
            next_point = p
    text = "road id = %d, lane id = %d"
    print(type(next_point))
    print(text % (next_point.road_id, next_point.lane_id))

    #debugger = self.client.get_world().debug
    #debugger.draw_point(next_point.transform.location, size=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)
    return next_point.transform


# used in run_multi_env currently
def images_to_video(env):
    videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)

    ffmpeg_cmd = (
        "ffmpeg -loglevel -8 -r 20 -f image2 -s {x_res}x{y_res} "
        "-pattern_type glob "
        "-i '{img}/*.png' -vcodec libx264 {vid}.mp4 && rm -f {img}/*.png"  #&& rm -f {img}/*.png
    ).format(
        x_res=env.render_x_res,
        y_res=env.render_y_res,
        #first_frame_num = self.first_frame_num,
        vid=os.path.join(videos_dir, self.episode_id),
        img=os.path.join(CARLA_OUT_PATH, "images"))
    print("Executing ffmpeg command", ffmpeg_cmd)
    subprocess.call(ffmpeg_cmd, shell=True)

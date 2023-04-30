


# ===============================================================================
# -- VehicleManager ------------------------------------------------------------
# ===============================================================================
class VehicleManager(object):
    def __init__(self, vehicle, autopilot_enabled=False):
        self._vehicle = vehicle
        self._autopilot_enabled = autopilot_enabled
        self._hud = None  # TODO
        self._collision_sensor = CollisionSensor(self._vehicle, self._hud)
        self._lane_invasion_sensor = LaneInvasionSensor(
            self._vehicle, self._hud)
        self._start_pos = None
        self._end_pos = None
        self._start_coord = None
        self._end_coord = None

    def get_location(self):
        return self._vehicle.get_location()

    def get_velocity(self):
        return self._vehicle.get_velocity()

    def draw_waypoints(self, helper, wp):
        nexts = list(wp.next(1.0))
        if not nexts:
            raise RuntimeError("No more waypoints")
        wp_next = random.choice(nexts)
        # text = "road id = %d, lane id = %d, transform = %s"
        # print(text % (wp_next.road_id, wp_next.lane_id, wp_next.transform))
        self.inner_wp_draw(helper, wp_next)

    def inner_wp_draw(self, helper, wp, depth=4):
        if depth < 0:
            return
        for w in wp.next(4.0):
            t = w.transform
            begin = t.location + carla.Location(
                z=40)  # TODO, the wp Z-coord is set as 0, not visiable
            angle = math.radians(t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            helper.draw_arrow(begin, end, arrow_size=0.1, life_time=1.0)
            self.inner_wp_draw(helper, w, depth - 1)

    def set_autopilot(self, autopilot_enabled):
        self._autopilot_enabled = autopilot_enabled
        self._vehicle.set_autopilot(self._autopilot_enabled)

    def get_autopilot(self):
        return self._autopilot_enabled

    def apply_control(self, control):
        self._vehicle.apply_control(control)

    def dynamic_collided(self):
        return self._collision_sensor.dynamic_collided()

    def offlane_invasion(self):
        return self._lane_invasion_sensor.get_offlane_percentage()

    # TODO: this routine need interect with road map data
    # issue#17, CPP code can be viewed at
    #   ACarlaVehicleController:IntersectPlayerWithRoadMap
    def offroad_invasion(self):
        return 0

    # TODO: for demo, all vehicles has same start_pos & end_pos
    # but in reality, need find the nearest pos at each spawn location
    def _nearest_pos(self, vid):
        pass

    def _pos_coord(self, scenario):
        city = ENV_CONFIG["server_map"].split("/")[-1]
        if city == "Town01":
            POS_COOR_MAP = TOWN01
        elif city == "Town02":
            POS_COOR_MAP = TOWN02
        # TODO: failure due to start_id type maybe list or int
        start_id = scenario["start_pos_id"]
        end_id = scenario["end_pos_id"]
        self._start_pos = POS_COOR_MAP[str(start_id)]
        self._end_pos = POS_COOR_MAP[str(end_id)]
        self._start_coord = [
            self._start_pos[0] // 100, self._start_pos[1] // 100
        ]
        self._end_coord = [self._end_pos[0] // 100, self._end_pos[1] // 100]
        return (self._start_pos, self._end_pos, self._start_coord,
                self._end_coord)

    def read_observation(self, scenario, config, step=0):
        c_vehicles, c_pedestrains, c_other = self.dynamic_collided()
        c_offline = self.offlane_invasion()
        c_offroad = self.offroad_invasion()
        start_pos, end_pos, start_coord, end_coord = self._pos_coord(scenario)
        cur_ = self._vehicle.get_transform()
        cur_x = cur_.location.x
        cur_y = cur_.location.y
        x_orient = cur_.rotation
        y_orient = cur_.rotation
        distance_to_goal_euclidean = float(
            np.linalg.norm([cur_x - end_pos[0], cur_y - end_pos[1]]) / 100)
        distance_to_goal = distance_to_goal_euclidean
        endcondition = atomic_scenario_behavior.InTriggerRegion(
            self._vehicle, 294, 304, 193, 203, name="reach end position")
        if endcondition.update() != py_trees.common.Status.SUCCESS:
            next_command = "LANE_FOLLOW"
        else:
            next_command = "REACH_GOAL"

        previous_action = "LANE_FOLLOW"

        vehicle_data = {
            "episode_id": 0,
            "step": step,
            "x": cur_x,
            "y": cur_y,
            "x_orient": x_orient,
            "y_orient": y_orient,
            "forward_speed": self._vehicle.get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": c_vehicles,
            "collision_pedestrians": c_pedestrains,
            "collision_other": c_other,
            "intersection_offroad": c_offroad,
            "intersection_otherlane": c_offline,
            "weather": None,
            "map": config["server_map"],
            "start_coord": start_coord,
            "end_coord": end_coord,
            "current_scenario": scenario,
            "x_res": config["x_res"],
            "y_res": config["y_res"],
            "num_vehicles": config["num_vehicles"],
            "num_pedestrians": config["num_pedestrians"],
            "max_steps": 10000,
            "next_command": next_command,
            "previous_action": previous_action,
            "previous_reward": 0
        }

        return vehicle_data

    def __del__(self):
        for actor in [
                self._collision_sensor.sensor,
                self._lane_invasion_sensor.sensor
        ]:
            if actor is not None:
                actor.destroy()


# ===============================================================================
# -- CarlaMap ------------------------------------------------------------------
# ===============================================================================
"""
from .carla.converter import Converter
class CarlaMap(object):
    IsRoadRow = 15
    HasDirectionRow = 14
    AngleMask = (0xFFFF >> 2)
    MaximumEncodeAngle = (1<<14) - 1
    PixelPerCentimeter =  6   # 1/0.1643
    CheckPerCentimeter = 0.1
    MAP_WIDTH = 800 #TODO
    MAP_HEIGHT = 600 #TODO


    def __init__(self, city, world, pixel_density=0.1643, node_density=50):
        dir_path = os.path.dirname(__file__)
        city_file = os.path.join(dir_path, city + '.txt')
        self._converter = Converter(city_file, pixel_density, node_density)
        self._map = world.get_map()

    def convert_to_pixel(self, input_data):
        "" "
        Receives a data type (Can Be Node or World )
        :param input_data: position in some coordinate
        :return: A node object
        "" "
        return self._converter.convert_to_pixel(input_data)

    def convert_to_world(self, input_data):
        "" "
        Receives a data type (Can Be Pixel or Node )
        :param input_data: position in some coordinate
        :return: A node object
        "" "
        return self._converter.convert_to_world(input_data)

    def get_map_topology(self):
        self._map.get_topology()

    def get_waypoint(self, location):
        return self._map.get_waypoint(location)

    def isRoad(self, pixelValue):
        return ( pixelValue & (1 << self.IsRoadRow)) != 0

    def clampFloatToInt(self, input_data, bmin, bmax):
        if input_data >= bmin or input_data <= bmax:
            return int(input_data)
        elif input_data < bmin:
            return bmin
        else :
            return bmax

    def get_data_at(self, WorldLocation):
        # additionally MapOffset
        pixle = self.convert_to_pixel(WorldLocation)
        indexx = clampFloatToInt(self.PixelPerCentimeter
            * pixle.x, 0, MAP_WIDTH-1);
        indexy = clampFloatToInt(self.PixelPerCentimeter
            * pixle.y, 0, MAP_HEIGHT-1);
        #TODO : access raw data from OpenDrive
        # return rawData[(indexx, indexy)]
        pass

    def get_offroad_percent(vehicle):
        bbox = vehicle.bounding_box
        bbox_rot = bbox.rotation
        bbox_ext = bbox.extent
        bbox_trans = carla.Transform(bbox.location)
        # TODO: bbox_rot * bbox_trans to get the forward direction
        step = 1.0 / self.CheckPerCentimeter
        checkCount = 0
        offRoad = 0.0
        for i in range(-bbox_ext.x , bbox_ext.x, step):
            for j in range(-bbox_ext.y,  bbox_ext.y, step):
                checkCount += 1
                pixel_data = self.get_data_at(
                    carla.Location(x=i, y=j, z=0))
                if not self.isRoad(pixel_data):
                    offRoad += 1.0
        if checkCount > 0:
            offRoad /= checkCount

        return offRoad
"""

# ===============================================================================
# -- Spawn Initial Location Detecter -------------------------------------------
# ===============================================================================


class Detecter(object):
    MAX_ITERATION = 20
    SPAWN_COLLISION = True

    def __init__(self, location, actor_list):
        self._first_center = location
        self._location = location
        self._actors = actor_list

# first transform the 8bbox vertices respect to the bbox transform
# then transform the vertices respect to vehicle(relative to the world coord)
# TODO: carla.Transform.transform_points() deprecated from v0.9.x

    def _bbox_vertices(self, vehicle):
        ext = vehicle.bounding_box.extent
        # 8bbox vertices relative to (0,0,0) locally
        bbox = np.array([[ext.x, ext.y, ext.z], [-ext.x, ext.y, ext.z],
                         [ext.x, -ext.y, ext.z], [-ext.x, -ext.y, ext.z],
                         [ext.x, ext.y, -ext.z], [-ext.x, ext.y, -ext.z],
                         [ext.x, -ext.y, -ext.z], [-ext.x, -ext.y, -ext.z]])

        vehicle_transform = carla.Transform(vehicle.get_location())
        bbox_transform = carla.Transform(vehicle.bounding_box.location)
        bbox = transform_points(bbox_transform, bbox)
        bbox = transform_points(vehicle_transform, bbox)

        return bbox

    def _min_max(self, a, b):
        min_ = min(a, b)
        max_ = max(a, b)
        return [min_, max_]

# TODO: need reimplement, not familar with Python numpy

    def _cubic(self, bbox):
        n1 = np.squeeze(np.asarray(bbox[0] - bbox[1]))  # x direction
        n2 = np.squeeze(np.asarray(bbox[0] - bbox[2]))  # y direction
        n3 = np.squeeze(np.asarray(bbox[0] - bbox[4]))  # z direction

        bbox0 = np.squeeze(np.asarray(bbox[0]))
        bbox1 = np.squeeze(np.asarray(bbox[1]))
        bbox2 = np.squeeze(np.asarray(bbox[2]))
        bbox4 = np.squeeze(np.asarray(bbox[4]))
        min1 = np.dot(bbox0, n1)
        max1 = np.dot(bbox1, n1)
        min1, max1 = self._min_max(min1, max1)
        min2 = np.dot(bbox0, n2)
        max2 = np.dot(bbox2, n2)
        min2, max2 = self._min_max(min2, max2)
        min3 = np.dot(bbox0, n3)
        max3 = np.dot(bbox4, n3)
        min3, max3 = self._min_max(min3, max3)
        return min1, max1, min2, max2, min3, max3, n1, n2, n3

    def _vel_update(self, vel):
        if vel < -0.1 or vel > 0.1:
            pass
        if -0.1 < vel <= 0:
            vel = -0.1
        elif 0.1 > vel >= 0:
            vel = 0.1
        return vel

    def collision(self):
        for vehicle in self._actors:
            vel = vehicle.get_velocity()
            bbox = self._bbox_vertices(vehicle)
            min1, max1, min2, max2, min3, max3, n1, n2, n3 = self._cubic(bbox)
            collision_flag = True

            iteration = 0
            try_location = np.zeros(3)
            while collision_flag and iteration < self.MAX_ITERATION:
                try_location[0] = self._location.x
                try_location[1] = self._location.y
                try_location[2] = self._location.z
                p1 = np.dot(n1, try_location)
                p2 = np.dot(n2, try_location)
                p3 = np.dot(n3, try_location)
                iteration += 1
                # adding bounding box size first
                ext = vehicle.bounding_box.extent
                self._location.x += ext.x
                self._location.y += ext.y
                self._location.z += ext.z
                if p1 >= min1 and p1 <= max1:
                    self._location.x += self._vel_update(vel.x)
                    print('collision happens in x direction,  '
                          'adding spwan x-location by %4.2f' %
                          self._vel_update(vel.x))
                    continue
                elif p2 >= min2 and p2 <= max2:
                    self._location.y += self._vel_update(vel.y)
                    print("collision happens in y direction, "
                          "adding spwan y-location by %4.2f" %
                          self._vel_update(vel.y))
                    continue
                elif p3 >= min3 and p3 <= max3:
                    self._location.z += self._vel_update(vel.z)
                    print("collision happens in z direction, "
                          "adding spwan z-location by %4.2f" %
                          self._vel_update(vel.z))
                    continue
                else:
                    break
                print("no collision with %2d" % vehicle.id)

            print('  will spawn a vehicle at location: '
                  '(%4.2f, %4.2f, %4.2f)' %
                  (self._location.x, self._location.y, self._location.z))
            return self._location


# ===============================================================================
# -- TrafficLight----------------------------------------------------------
# ===============================================================================
# TODO: import update


class TrafficLight(object):
    def __init__(self, parent_actor):
        self.light = None
        self._parent = parent_actor
        self.world = self._parent.get_world()
        # These values are fixed for every city.
        self._node_density = 50.0
        self._pixel_density = 0.1643
        city_name = 'Town01'
        self._map = CarlaMap(city_name, self._pixel_density,
                             self._node_density)
        self.lights = []
        actors = self.world.get_actors()
        for actor in actors:
            if actor.type_id == 'traffic.traffic_light':
                self.lights.append(actor)

    # https://github.com/carla-simulator/driving-benchmarks/
    #   blob/master/version084/benchmark_tools/benchmark_runner.py

    def is_traffic_light_active(self, light, orientation):
        x_agent = light.transform.location.x
        y_agent = light.transform.location.y

        def search_closest_lane_point(x_agent, y_agent, depth):
            step_size = 4
            if depth > 1:
                return None
            # try:
            degrees = self._map.get_lane_orientation_degrees(
                [x_agent, y_agent, 38])
            # print (degrees)
            # except:
            #    return None

            if not self._map.is_point_on_lane([x_agent, y_agent, 38]):
                # print (" Not on lane ")
                result = search_closest_lane_point(x_agent + step_size,
                                                   y_agent, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent,
                                                   y_agent + step_size,
                                                   depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent + step_size,
                                                   y_agent + step_size,
                                                   depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent + step_size,
                                                   y_agent - step_size,
                                                   depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size,
                                                   y_agent + step_size,
                                                   depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size,
                                                   y_agent, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent,
                                                   y_agent - step_size,
                                                   depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size,
                                                   y_agent - step_size,
                                                   depth + 1)
                if result is not None:
                    return result
            else:
                # print(" ON Lane ")
                if degrees < 6:
                    return [x_agent, y_agent]
                else:
                    return None

        closest_lane_point = search_closest_lane_point(x_agent, y_agent, 0)
        car_direction = math.atan2(orientation.y, orientation.x) + 3.1415
        if car_direction > 6.0:
            car_direction -= 6.0

        return math.fabs(
            car_direction - self._map.get_lane_orientation_degrees(
                [closest_lane_point[0], closest_lane_point[1], 38])) < 1

    def _test_for_traffic_light(self, vehicle):
        """
        test if car passed into a traffic light,
            returning 'red' if it crossed a red light
            returnning 'green' if it crossed a green light
            or none otherwise
        """
        def is_on_burning_point(_map, location):  # location of vehicle
            # We get the current lane orientation
            ori_x, ori_y = _map.get_lane_orientation(
                [location.x, location.y, 38])

            # We test to walk in direction of the lane
            future_location_x = location.x
            future_location_y = location.y

            for i in range(3):
                future_location_x += ori_x
                future_location_y += ori_y
            # Take a point on a intersection in the future
            location_on_intersection_x = future_location_x + 2 * ori_x
            location_on_intersection_y = future_location_y + 2 * ori_y

            if not _map.is_point_on_intersection([future_location_x,
                                                  future_location_y,
                                                  38]) and \
                    _map.is_point_on_intersection([location_on_intersection_x,
                                                   location_on_intersection_y,
                                                   38]):
                return True

            return False

        # check nearest traffic light with the correct orientation state
        player_x = vehicle.get_location().x
        player_y = vehicle.get_location().y

        for light in self.lights:
            if light is not None:
                if not self._map.is_point_on_intersection(
                    [player_x, player_y, 38]):  # noqa: E125 yapf bug
                    #  light_x and light_y never used
                    # light_x = light.transform.location.x
                    # light_y = light.transform.location.y

                    #  unknow func get_vec_dist
                    # t1_vector, t1_dist = get_vec_dist(light_x, light_y,
                    #                                 player_x, player_y)
                    if self.is_traffic_light_active(
                            light, vehicle.transform.orientation):
                        if is_on_burning_point(self._map):
                            #  t1_dist never defined
                            # vehicle.transform.location) and t1_dist < 6.0:
                            if light.state != 0:  # not green
                                return 'red'
                            else:
                                return 'green'
        return None


# ==============================================================================
# -- LaneInvasionSensor -------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp,
                                        carla.Transform(),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._offlane = 0
        self._off_lane_percentage = 0
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_offlane_percentage(self):
        return self._off_lane_percentage

    # TODO: calc percentage of vehicle offlane
    def calc_percentage(self):
        return self._off_lane_percentage

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

    # text = ['%r' % str(x).split()[-1]
    #        for x in set(event.crossed_lane_markings)]
    #   self._hud.notification('Crossed line %s' % ' and '.join(text))
        text = [
            '%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)
        ]
        print('VEHICLE %s' % (self._parent).id +
              ' crossed line %s' % ' and '.join(text))
        self._offlane += 1
        self._off_lane_percentage = self._offlane / event.frame_number * 100
        print('off lane percentage %6.4f' % self._off_lane_percentage)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self.collision_vehicles = 0
        self.collision_pedestrains = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp,
                                        carla.Transform(),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
#        actor_type = get_actor_display_name(event.other_actor)
#        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)

        print('vehicle %s ' % (self._parent).id +
              ' collision with %2d vehicles, %2d people, %2d others' %
              self.dynamic_collided())
        _cur = event.other_actor
        if _cur.id == 0:  # the static world objects
            if _cur.type_id in self.collision_type_id_set:
                return
            else:
                self.collision_type_id_set.add(_cur.type_id)
        else:
            if _cur.id in self.collision_id_set:
                return
            else:
                self.collision_id_set.add(_cur.id)

        collided_type = type(_cur).__name__
        if collided_type == 'Vehicle':
            self.collision_vehicles += 1
        elif collided_type == 'Pedestrain':
            self.collision_pedestrains += 1
        elif collided_type == 'Actor':
            self.collision_other += 1
        else:
            pass

    def _reset(self):
        self.collision_vehicles = 0
        self.collision_pedestrains = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()

    def dynamic_collided(self):
        return (self.collision_vehicles, self.collision_pedestrains,
                self.collision_other)


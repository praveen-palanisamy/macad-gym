# noqa
import math


class TrafficLightManager:
    """Controller for traffic light objects."""

    def __init__(self, tl_config, tl_object):
        """Constructor.

        Args:
            tl_config: actor configuration
            tl_object: world object
        """
        self._config = tl_config
        self._tl = tl_object

    def read_observation(self):
        """Read observation and return measurement.

        Returns:
            dict: measurement data.
        """
        pass

    def apply_control(self, traffic_light_state):
        """Apply new state to a traffic light object.

        Args:
            traffic_light_state: new state

        Returns:
            N/A.
        """
        pass

    # def is_traffic_light_active(self, light, orientation):
    #     x_agent = light.transform.location.x
    #     y_agent = light.transform.location.y
    #
    #     def search_closest_lane_point(x_agent, y_agent, depth):
    #         step_size = 4
    #         if depth > 1:
    #             return None
    #         # try:
    #         degrees = self._map.get_lane_orientation_degrees([x_agent, y_agent, 38])
    #         # print (degrees)
    #         # except:
    #         #    return None
    #
    #         if not self._map.is_point_on_lane([x_agent, y_agent, 38]):
    #             # print (" Not on lane ")
    #             result = search_closest_lane_point(x_agent + step_size, y_agent, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent, y_agent + step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent + step_size, y_agent + step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent + step_size, y_agent - step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent - step_size, y_agent + step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent - step_size, y_agent, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent, y_agent - step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #             result = search_closest_lane_point(x_agent - step_size, y_agent - step_size, depth + 1)
    #             if result is not None:
    #                 return result
    #         else:
    #             # print(" ON Lane ")
    #             if degrees < 6:
    #                 return [x_agent, y_agent]
    #             else:
    #                 return None
    #
    #     closest_lane_point = search_closest_lane_point(x_agent, y_agent, 0)
    #     car_direction = math.atan2(orientation.y, orientation.x) + 3.1415
    #     if car_direction > 6.0:
    #         car_direction -= 6.0
    #
    #     return math.fabs(car_direction - self._map.get_lane_orientation_degrees(
    #         [closest_lane_point[0], closest_lane_point[1], 38])) < 1
    #
    # def _test_for_traffic_light(self, vehicle):
    #     """
    #     test if car passed into a traffic light,
    #         returning 'red' if it crossed a red light
    #         returning 'green' if it crossed a green light
    #         or none otherwise
    #     """
    #
    #     def is_on_burning_point(_map, location):  # location of vehicle
    #         # We get the current lane orientation
    #         ori_x, ori_y = _map.get_lane_orientation([location.x, location.y, 38])
    #
    #         # We test to walk in direction of the lane
    #         future_location_x = location.x
    #         future_location_y = location.y
    #
    #         for i in range(3):
    #             future_location_x += ori_x
    #             future_location_y += ori_y
    #         # Take a point on a intersection in the future
    #         location_on_intersection_x = future_location_x + 2 * ori_x
    #         location_on_intersection_y = future_location_y + 2 * ori_y
    #
    #         if not _map.is_point_on_intersection(
    #                 [future_location_x, future_location_y, 38]) and _map.is_point_on_intersection(
    #             [location_on_intersection_x, location_on_intersection_y, 38]):
    #             return True
    #
    #         return False
    #
    #     # check nearest traffic light with the correct orientation state
    #     player_x = vehicle.get_location().x
    #     player_y = vehicle.get_location().y
    #
    #     for light in self.lights:
    #         if light is not None:
    #             if not self._map.is_point_on_intersection([player_x, player_y, 38]):  # noqa: E125 yapf bug
    #                 #  light_x and light_y never used
    #                 # light_x = light.transform.location.x
    #                 # light_y = light.transform.location.y
    #
    #                 #  unknown func get_vec_dist
    #                 # t1_vector, t1_dist = get_vec_dist(light_x, light_y,
    #                 #                                 player_x, player_y)
    #                 if self.is_traffic_light_active(light, vehicle.transform.orientation):
    #                     if is_on_burning_point(self._map):
    #                         #  t1_dist never defined
    #                         # vehicle.transform.location) and t1_dist < 6.0:
    #                         if light.state != 0:  # not green
    #                             return 'red'
    #                         else:
    #                             return 'green'
    #     return None

    def __del__(self):
        """Delete instantiated sub-elements."""
        pass

    @staticmethod
    def get_traffic_light(
        world, candidate_transform, sort=False, distance_threshold=50.0, angle_threshold=math.pi / 4.0
    ):
        """Get a list of traffic lights that will affect an actor.

        Helpful when the actor is approaching a signal controlled intersection. May
        not work well when the actor is turning unless the angle_threshold is
        appropriate.

        Args:
            world (carla.world): Carla World object
            candidate_transform (carla.Transform): Pose (location & orientation) of
                interest for which relevant traffic lights are to be found
            sort (bool): Return a sorted list of TrafficLights based on L2 distance
                and angle if True.
            distance_threshold: Maximum L2 distance to search for the lights
            angle_threshold:

        Returns:
            list: Containing carla.TrafficLights that affect the actor
        """
        tls = {}
        ax, ay = candidate_transform.location.x, candidate_transform.location.y

        for t in world.get_actors().filter("traffic.traffic_light"):
            tx, ty = t.get_location().x, t.get_location().y
            dist = math.sqrt((ax - tx) * (ax - tx) + (ay - ty) * (ay - ty))
            if dist < distance_threshold:
                actor_orientation = math.radians(candidate_transform.rotation.yaw)
                traffic_light_orientation = math.radians(t.get_transform().rotation.yaw)
                angle = math.fabs(
                    ((traffic_light_orientation - actor_orientation + math.pi) % (math.pi * 2.0)) - math.pi
                )

                if math.fabs(angle) > angle_threshold and math.fabs(angle) < math.pi - angle_threshold:
                    tls[t] = (dist, angle)
        if sort:
            # Return a sorted list sorted first based on dist & then angle
            return sorted(tls.items(), key=lambda kv: kv[1])
        else:
            return list(tls.keys())

    @staticmethod
    def get_all_traffic_lights(world):
        """Get a list of all traffic lights in the map.

        Args:
            world (carla.world): Carla World object
        Returns:
            list: Containing carla.TrafficLights

        """
        tls = []
        for t in world.get_actors().filter("traffic.traffic_light"):
            tls.append(t)
        return tls

    @staticmethod
    def set_traffic_light_tate(traffic_lights, traffic_light_state):
        """Sets all traffic lights in the `traffic_lights` list to the given state.

        Args:
            traffic_lights (list): List of carla.TrafficLight actors
            traffic_light_state (carla.TrafficLightState):  The state to set the
                lights

        Returns:
            N/A.

        """
        for traffic_light in traffic_lights:
            traffic_light.set_state(traffic_light_state)

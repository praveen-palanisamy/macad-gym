import math

from .utils import get_vec_dist, get_angle
from carla.planner.map import CarlaMap

class ObstacleAvoidance(object):

    def __init__(self, param, city_name):

        print (" Map Name ", city_name)
        self._map = CarlaMap(city_name)
        self.param = param
        # Select WP Number


    def is_traffic_light_visible(self, location, agent):

        x_agent = agent.traffic_light.transform.location.x
        y_agent = agent.traffic_light.transform.location.y

        _, tl_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)

        return tl_dist > (self.param['tl_min_dist_thres'])

    def is_traffic_light_active(self, location, agent, orientation):

        x_agent = agent.traffic_light.transform.location.x
        y_agent = agent.traffic_light.transform.location.y
        #_, tl_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
        def search_closest_lane_point(x_agent, y_agent, depth):
            step_size = 4
            #print ('depth ', depth, 'x_agent', x_agent, 'y_agent', y_agent)
            if depth > 1:
                return None
            try:
                degrees = self._map.get_lane_orientation_degrees([x_agent, y_agent, 38])
                #print (degrees)
            except:
                return None

            if not self._map.is_point_on_lane([x_agent, y_agent, 38]):
                #print (" Not on lane ")
                result = search_closest_lane_point(x_agent + step_size, y_agent, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent, y_agent + step_size, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent + step_size, y_agent + step_size, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent + step_size, y_agent - step_size, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size, y_agent + step_size, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size, y_agent, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent, y_agent - step_size, depth+1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size, y_agent - step_size, depth+1)
                if result is not None:
                    return result
            else:
                #print(" ON Lane ")
                if degrees < 6:
                    return [x_agent, y_agent]
                else:
                    return None


        #print (" Start ")

        closest_lane_point = search_closest_lane_point(x_agent, y_agent, 0)


        # math.fabs(self._map.get_lane_orientation_degrees([location.x, location.y, 38])

        print("  Angle ", math.atan2(orientation.y, orientation.x) + 3.1415)

        car_direction = math.atan2(orientation.y, orientation.x) + 3.1415
        if car_direction > 6.0:
            car_direction -= 6.0

        return math.fabs(car_direction -
        self._map.get_lane_orientation_degrees([closest_lane_point[0], closest_lane_point[1], 38])
                         ) < 1


    def stop_traffic_light(self, location, agent, wp_vector, wp_angle, speed_factor_tl):

        speed_factor_tl_temp = 1

        if agent.traffic_light.state != 0:  # Not green
            x_agent = agent.traffic_light.transform.location.x
            y_agent = agent.traffic_light.transform.location.y
            tl_vector, tl_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
            # tl_angle = self.get_angle(tl_vector,[ori_x_player,ori_y_player])
            tl_angle = get_angle(tl_vector, wp_vector)
            # print ('Traffic Light: ', tl_vector, tl_dist, tl_angle)


            if (0 < tl_angle < self.param['tl_angle_thres'] / self.param['coast_factor']
                and tl_dist < self.param['tl_max_dist_thres'] * self.param['coast_factor']) \
                    or (
                    0 < tl_angle < self.param['tl_angle_thres'] and tl_dist < self.param['tl_max_dist_thres']) and math.fabs(
                wp_angle) < 0.2:

                #print ()
                #print (' case 1 Traffic Light')
                #print ()

                speed_factor_tl_temp = tl_dist / (self.param['coast_factor'] * self.param['tl_max_dist_thres'])

            if (0 < tl_angle < self.param['tl_angle_thres'] * self.param['coast_factor'] and tl_dist < self.param['tl_max_dist_thres'] / self.param['coast_factor']) and math.fabs(
                wp_angle) < 0.2:
                speed_factor_tl_temp = 0
                #print ()
                #print (' case 2 Traffic Light')
                #print ()

            if (speed_factor_tl_temp < speed_factor_tl):
                speed_factor_tl = speed_factor_tl_temp

        return speed_factor_tl

    def has_burned_traffic_light(self, location, agent, wp_vector, orientation):

        def is_on_burning_point(_map, location):

            # We get the current lane orientation
            ori_x, ori_y = _map.get_lane_orientation([location.x, location.y, 38])

            print("orientation ", ori_x, ori_y)
            # We test to walk in direction of the lane
            future_location_x = location.x
            future_location_y = location.y

            print ("future ", future_location_x, future_location_y)

            for i in range(3):
                future_location_x += ori_x
                future_location_y += ori_y
            # Take a point on a intersection in the future
            location_on_intersection_x = future_location_x + 2*ori_x
            location_on_intersection_y = future_location_y + 2*ori_y
            print ("location ", location_on_intersection_x, location_on_intersection_y)

            if not _map.is_point_on_intersection([future_location_x,
                                                  future_location_y,
                                                  38]) and \
               _map.is_point_on_intersection([location_on_intersection_x,
                                              location_on_intersection_y,
                                              38]):
               return [[future_location_x, future_location_y],
                       [location_on_intersection_x, location_on_intersection_y]
                       ], True

            return [[future_location_x, future_location_y],
                    [location_on_intersection_x, location_on_intersection_y]
                    ], False

        positions = []
        # The vehicle is on not an intersection
        if not self._map.is_point_on_intersection([location.x, location.y, 38]):
            x_agent = agent.traffic_light.transform.location.x
            y_agent = agent.traffic_light.transform.location.y
            tl_vector, tl_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
            # tl_angle = self.get_angle(tl_vector,[ori_x_player,ori_y_player])
            tl_angle = get_angle(tl_vector, wp_vector)

            if agent.traffic_light.state != 0:  # Not green

                if self.is_traffic_light_active(location, agent, orientation):
                    print("ORIENTATION ", orientation.x, "  ", orientation.y)
                    print("  Angle ", math.atan2(orientation.y, orientation.x) + 3.1415)
                    print( " LOCATION ", location.x, location.y)
                    print('Traffic Light: ', tl_dist, tl_angle)
                    positions, burned = is_on_burning_point(self._map, location)
                    if burned and tl_dist < 6.0:
                        print("ORIENTATION ", orientation.x, "  ", orientation.y)
                        print(" Angle ", math.atan2(orientation.y, orientation.x) + 3.1415)
                        print(" LOCATION ", location.x, location.y)
                        print('Traffic Light: ', tl_dist, tl_angle)
                        print(" \n\n BUUURNNNNNNN \n\n")
                        exit(1)
                        return positions

        return positions


    def is_pedestrian_hitable(self, pedestrian):

        """
        Determine if a certain pedestrian is in a hitable zone or it is pasible
        to be hit in a near future.
        Should check if pedestrians are on lane and or if the pedestrians are out
        of the lane but with velocity vector pointing to lane.
        :return:
        """


        # TODO: for now showing only if pedestrians are on the road

        x_agent = pedestrian.transform.location.x
        y_agent = pedestrian.transform.location.y

        return self._map.is_point_on_lane([x_agent, y_agent, 38])


    def is_vehicle_on_same_lane(self, player, vehicle):
        """
            Check if the vehicle is on the same lane as the player
        :return:
        """

        x_agent = vehicle.transform.location.x
        y_agent = vehicle.transform.location.y

        if self._map.is_point_on_intersection([x_agent, y_agent, 38]):
            return True


        return math.fabs(self._map.get_lane_orientation_degrees([player.x, player.y, 38]) -
        self._map.get_lane_orientation_degrees([x_agent, y_agent, 38])) < 1






    def is_pedestrian_on_hit_zone(self, p_dist, p_angle):
        """
        Draw a semi circle with a big radius but small period from the circunference.
        Pedestrians on this zone will cause the agent to reduce the speed

        """

        return math.fabs(p_angle) < self.param['p_angle_hit_thres'] and p_dist < self.param['p_dist_hit_thres']


    def is_pedestrian_on_near_hit_zone(self, p_dist, p_angle):

        return math.fabs(p_angle) < self.param['p_angle_eme_thres'] and p_dist < self.param['p_dist_eme_thres']


    def stop_pedestrian(self, location, agent, wp_vector, speed_factor_p):



        """
        if is_pedestrian_on_near_hit_zone():

            return 0

        if is_pedestrian_on_hit_zone():
            # return  some proportional to distance deacceleration constant.
            pass
        """
        speed_factor_p_temp = 1

        x_agent = agent.pedestrian.transform.location.x
        y_agent = agent.pedestrian.transform.location.y
        p_vector, p_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
        # p_angle = self.get_angle(p_vector,[ori_x_player,ori_y_player])
        p_angle = get_angle(p_vector, wp_vector)

        # Define flag, if pedestrian is outside the sidewalk ?
        # print('Pedestrian: ', p_vector, p_dist, p_angle)

        if self.is_pedestrian_on_hit_zone(p_dist, p_angle):

            speed_factor_p_temp = p_dist / (self.param['coast_factor'] * self.param['p_dist_hit_thres'])



        if self.is_pedestrian_on_near_hit_zone(p_dist, p_angle):

            speed_factor_p_temp = 0
            #print()
            #print(" Case 2 Pedestrian")
            #print()



        if (speed_factor_p_temp < speed_factor_p):
            speed_factor_p = speed_factor_p_temp


        return speed_factor_p


    def stop_vehicle(self, location, agent, wp_vector, speed_factor_v):


        speed_factor_v_temp = 1
        x_agent = agent.vehicle.transform.location.x
        y_agent = agent.vehicle.transform.location.y
        v_vector, v_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
        # v_angle = self.get_angle(v_vector,[ori_x_player,ori_y_player])
        v_angle = get_angle(v_vector, wp_vector)
        # print ('Vehicle: ', v_vector, v_dist, v_angle)
        # print (v_angle, self.param['v_angle_thres'], self.param['coast_factor'])
        if (
                -0.5 * self.param['v_angle_thres'] / self.param['coast_factor'] < v_angle < self.param['v_angle_thres'] / self.param['coast_factor'] and v_dist < self.param['v_dist_thres'] * self.param['coast_factor']) or (
                -0.5 * self.param['v_angle_thres'] / self.param['coast_factor'] < v_angle < self.param['v_angle_thres'] and v_dist < self.param['v_dist_thres']):
            speed_factor_v_temp = v_dist / (self.param['coast_factor'] * self.param['v_dist_thres'])
            #print()
            #print(' case 1 Vehicle ')
            #print()

        if (
                -0.5 * self.param['v_angle_thres'] * self.param['coast_factor'] < v_angle < self.param['v_angle_thres'] * self.param['coast_factor'] and v_dist < self.param['v_dist_thres'] / self.param['coast_factor']):
            speed_factor_v_temp = 0
            #print()
            #print(' case 1 Vehicle ')
            #print()

        if (speed_factor_v_temp < speed_factor_v):
            speed_factor_v = speed_factor_v_temp

        return speed_factor_v

    def stop_for_agents(self, location, orientation, wp_angle, wp_vector, agents):

        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1
        hitable_pedestrians = []    # The list of pedestrians that are on roads or nearly on roads
        out_pos = []

        for agent in agents:

            positions = self.has_burned_traffic_light( location, agent, wp_vector, orientation)

            if len(positions) > 0:

                out_pos = positions

            if agent.HasField('traffic_light') and self.param['stop4TL']:
                if self.is_traffic_light_active(location, agent, orientation) and self.is_traffic_light_visible(location, agent):

                    speed_factor_tl = self.stop_traffic_light(location, agent, wp_vector,
                                                              wp_angle, speed_factor_tl)


                    hitable_pedestrians.append(agent.id)

            if agent.HasField('pedestrian') and self.param['stop4P']:
                if self.is_pedestrian_hitable(agent.pedestrian):

                    speed_factor_p = self.stop_pedestrian(location, agent, wp_vector, speed_factor_p)


            if agent.HasField('vehicle') and self.param['stop4V']:
                if self.is_vehicle_on_same_lane(player=location, vehicle=agent.vehicle):
                    speed_factor_v = self.stop_vehicle(location, agent, wp_vector, speed_factor_v)


            speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)


        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }

        return speed_factor, hitable_pedestrians, state, out_pos

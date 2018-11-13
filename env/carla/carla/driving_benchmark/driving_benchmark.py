# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import abc
import logging
import math
import time
import numpy as np

from carla.client import VehicleControl
from carla.client import make_carla_client
from carla.driving_benchmark.metrics import Metrics
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from . import results_printer
from .recording import Recording

def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / dist, dist


def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class DrivingBenchmark(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.


    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            save_images=False,
            distance_for_success=2.0
    ):
        """
        Args
            city_name:
            name_to_save:
            continue_experiment:
            save_images:
            distance_for_success:
            collisions_as_failure: if this flag is set to true, episodes will terminate as failure, when the car collides.
        """

        self.__metaclass__ = abc.ABCMeta

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images
                                    )

        # We have a default planner instantiated that produces high level commands
        self._planner = Planner(city_name)
        self._map = self._planner._city_track.get_map()

        # TO keep track of the previous collisions
        self._previous_pedestrian_collision = 0
        self._previous_vehicle_collision = 0
        self._previous_other_collision = 0



    def benchmark_agent(self, experiment_suite, agent, client):
        """
        Function to benchmark the agent.
        It first check the log file for this benchmark.
        if it exist it continues from the experiment where it stopped.


        Args:
            experiment_suite
            agent: an agent object with the run step class implemented.
            client:


        Return:
            A dictionary with all the metrics computed from the
            agent running the set of experiments.
        """

        # Instantiate a metric object that will be used to compute the metrics for
        # the benchmark afterwards.
        metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

        # Function return the current pose and task for this benchmark.
        start_pose, start_experiment = self._recording.get_pose_and_experiment(
            experiment_suite.get_number_of_poses_task())

        logging.info('START')

        for experiment in experiment_suite.get_experiments()[int(start_experiment):]:

            positions = client.load_settings(
                experiment.conditions).player_start_spots

            self._recording.log_start(experiment.task)

            for pose in experiment.poses[start_pose:]:
                for rep in range(experiment.repetitions):

                    start_index = pose[0]
                    end_index = pose[1]

                    client.start_episode(start_index)
                    # Print information on
                    logging.info('======== !!!! ==========')
                    logging.info(' Start Position %d End Position %d ',
                                 start_index, end_index)

                    self._recording.log_poses(start_index, end_index,
                                              experiment.Conditions.WeatherId)

                    # Calculate the initial distance for this episode
                    initial_distance = \
                        sldist(
                            [positions[start_index].location.x, positions[start_index].location.y],
                            [positions[end_index].location.x, positions[end_index].location.y])

                    time_out = experiment_suite.calculate_time_out(
                        self._get_shortest_path(positions[start_index], positions[end_index]))

                    logging.info('Timeout for Episode: %f', time_out)
                    # running the agent
                    (result, reward_vec, control_vec, final_time, remaining_distance, col_ped,
                     col_veh, col_oth, number_of_red_lights, number_of_green_lights) = \
                        self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index), experiment_suite.metrics_parameters,
                            experiment_suite.collision_as_failure,
                            experiment_suite.traffic_light_as_failure)

                    # Write the general status of the just ran episode
                    self._recording.write_summary_results(
                        experiment, pose, rep, initial_distance,
                        remaining_distance, final_time, time_out, result, col_ped, col_veh, col_oth,
                        number_of_red_lights, number_of_green_lights)

                    # Write the details of this episode.
                    self._recording.write_measurements_results(experiment, rep, pose, reward_vec,
                                                               control_vec)
                    if result > 0:
                        logging.info('+++++ Target achieved in %f seconds! +++++',
                                     final_time)
                    else:
                        logging.info('----- Timeout! -----')

            start_pose = 0

        self._recording.log_end()

        return metrics_object.compute(self._recording.path)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _has_agent_collided(self, measurement, metrics_parameters):

        """
            This function must have a certain state and only look to one measurement.
        """
        collided_veh = 0
        collided_ped = 0
        collided_oth = 0

        if (measurement.collision_vehicles - self._previous_vehicle_collision) \
                > metrics_parameters['collision_vehicles']['threshold']/2.0:
            collided_veh = 1
        if (measurement.collision_pedestrians - self._previous_pedestrian_collision) \
                > metrics_parameters['collision_pedestrians']['threshold']/2.0:
            collided_ped = 1
        if (measurement.collision_other - self._previous_other_collision) \
                > metrics_parameters['collision_other']['threshold']/2.0:
            collided_oth = 1

        self._previous_pedestrian_collision = measurement.collision_pedestrians
        self._previous_vehicle_collision = measurement.collision_vehicles
        self._previous_other_collision = measurement.collision_other

        return collided_ped, collided_veh, collided_oth

    def _is_traffic_light_active(self, agent, orientation):

        x_agent = agent.traffic_light.transform.location.x
        y_agent = agent.traffic_light.transform.location.y

        def search_closest_lane_point(x_agent, y_agent, depth):
            step_size = 4
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


        closest_lane_point = search_closest_lane_point(x_agent, y_agent, 0)
        car_direction = math.atan2(orientation.y, orientation.x) + 3.1415
        if car_direction > 6.0:
            car_direction -= 6.0

        return math.fabs(car_direction -
            self._map.get_lane_orientation_degrees([closest_lane_point[0], closest_lane_point[1], 38])
                         ) < 1

    def _test_for_traffic_lights(self, measurement):
        """

        This function tests if the car passed into a traffic light, returning 'red'
        if it crossed a red light , 'green' if it crossed a green light or none otherwise

        Args:
            measurement: all the measurements collected by carla 0.8.4

        Returns:

        """

        def is_on_burning_point(_map, location):

            # We get the current lane orientation
            ori_x, ori_y = _map.get_lane_orientation([location.x, location.y, 38])

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

        # Check nearest traffic light with the correct orientation state.

        player_x = measurement.player_measurements.transform.location.x
        player_y = measurement.player_measurements.transform.location.y

        # The vehicle is on an intersection
        # THIS IS THE PLACE TO VERIFY FOR A TL BURN

        for agent in measurement.non_player_agents:
            if agent.HasField('traffic_light'):
                if not self._map.is_point_on_intersection([player_x, player_y, 38]):
                    x_agent = agent.traffic_light.transform.location.x
                    y_agent = agent.traffic_light.transform.location.y
                    tl_vector, tl_dist = get_vec_dist(x_agent, y_agent, player_x, player_y)
                    if self._is_traffic_light_active(agent,
                                                     measurement.player_measurements.
                                                             transform.orientation):
                        if is_on_burning_point(self._map,
                                               measurement.player_measurements.transform.location)\
                                and tl_dist < 6.0:
                            if agent.traffic_light.state != 0:  # Not green
                                return 'red'
                            else:
                                return 'green'

        return None

    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name,
            metrics_parameters,
            collision_as_failure,
            traffic_light_as_failure):
        """
         Run one episode of the benchmark (Pose) for a certain agent.


        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target to reach
            episode_name: The name for saving images of this episode
            metrics_object: The metrics object to check for collisions

        """

        # Send an initial command.
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        measurement_vec = []
        # The vector containing all controls produced on this episode
        control_vec = []
        frame = 0
        distance = 10000
        col_ped, col_veh, col_oth = 0, 0, 0
        traffic_light_state, number_red_lights, number_green_lights = None, 0, 0
        fail = False
        success = False
        not_count = 0

        while not fail and not success:

            # Read data from server with the client
            measurements, sensor_data = client.read_data()
            # The directions to reach the goal are calculated.
            directions = self._get_directions(measurements.player_measurements.transform, target)
            # Agent process the data.
            control = agent.run_step(measurements, sensor_data, directions, target)
            # Send the control commands to the vehicle
            client.send_control(control)

            # save images if the flag is activated
            self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y

            logging.info("Controller is Inputting:")
            logging.info('Steer = %f Throttle = %f Brake = %f ',
                         control.steer, control.throttle, control.brake)

            current_timestamp = measurements.game_timestamp
            logging.info('Timestamp %f', current_timestamp)
            # Get the distance travelled until now

            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])
            # Write status of the run on verbose mode
            logging.info('Status:')
            logging.info(
                '[d=%f] c_x = %f, c_y = %f ---> t_x = %f, t_y = %f',
                float(distance), current_x, current_y, target.location.x,
                target.location.y)
            # Check if reach the target
            col_ped, col_veh, col_oth = self._has_agent_collided(measurements.player_measurements,
                                                                 metrics_parameters)
            # test if car crossed the traffic light
            traffic_light_state = self._test_for_traffic_lights(measurements)

            if traffic_light_state == 'red' and not_count == 0:
                number_red_lights += 1
                not_count = 20

            elif traffic_light_state == 'green' and not_count == 0:
                number_green_lights += 1
                not_count = 20

            else:
                not_count -= 1
                not_count = max(0, not_count)

            if distance < self._distance_for_success:
                success = True
            elif (current_timestamp - initial_timestamp) > (time_out * 1000):
                fail = True
            elif collision_as_failure and (col_ped or col_veh or col_oth):
                fail = True
            elif traffic_light_as_failure and traffic_light_state == 'red':
                fail = True
            logging.info('Traffic Lights:')
            logging.info(
                'red %f green %f, total %f',
                number_red_lights, number_green_lights, number_red_lights + number_green_lights)
            # Increment the vectors and append the measurements and controls.
            frame += 1
            measurement_vec.append(measurements.player_measurements)
            control_vec.append(control)

        if success:
            return 1, measurement_vec, control_vec, float(
                current_timestamp - initial_timestamp) / 1000.0, distance,  col_ped, col_veh, col_oth, \
                   number_red_lights, number_green_lights
        return 0, measurement_vec, control_vec, time_out, distance, col_ped, col_veh, col_oth, \
            number_red_lights, number_green_lights


def run_driving_benchmark(agent,
                          experiment_suite,
                          city_name='Town01',
                          log_name='Test',
                          continue_experiment=False,
                          host='127.0.0.1',
                          port=2000
                          ):
    while True:
        try:

            with make_carla_client(host, port) as client:
                # Hack to fix for the issue 310, we force a reset, so it does not get
                #  the positions on first server reset.
                client.load_settings(CarlaSettings())
                client.start_episode(0)

                # We instantiate the driving benchmark, that is the engine used to
                # benchmark an agent. The instantiation starts the log process, sets

                benchmark = DrivingBenchmark(city_name=city_name,
                                             name_to_save=log_name + '_'
                                                          + type(experiment_suite).__name__
                                                          + '_' + city_name,
                                             continue_experiment=continue_experiment)
                # This function performs the benchmark. It returns a dictionary summarizing
                # the entire execution.

                benchmark_summary = benchmark.benchmark_agent(experiment_suite, agent, client)

                print("")
                print("")
                print("----- Printing results for training weathers (Seen in Training) -----")
                print("")
                print("")
                results_printer.print_summary(benchmark_summary, experiment_suite.train_weathers,
                                              benchmark.get_path())

                print("")
                print("")
                print("----- Printing results for test weathers (Unseen in Training) -----")
                print("")
                print("")

                results_printer.print_summary(benchmark_summary, experiment_suite.test_weathers,
                                              benchmark.get_path())

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

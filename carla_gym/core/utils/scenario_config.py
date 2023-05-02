#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for an XML-based scenario
"""
import warnings
from collections.abc import Iterable
from typing import List, Dict
import xml.etree.ElementTree as ET

import carla
from carla_gym.core.controllers.camera_manager import CAMERA_TYPES

from carla_gym.core.constants import WEATHERS


def strtobool(s):
    return str(s).lower() == "true"


class ActorConfiguration(object):
    def __init__(self, name, render=False, enable_planner=True, camera_type="rgb", camera_position=0, framestack=1, lane_sensor=True, collision_sensor=True, early_terminate_on_collision=True,
                 manual_control=False, auto_control=False, squash_action_logits=False, reward_function="corl2017", send_measurements=False, log_images=False, log_measurements=False, verbose=False):
        self.name = name
        self.render = render
        self.enable_planner = enable_planner
        self.camera_type = camera_type
        self.camera_position = camera_position
        self.framestack = framestack
        self.lane_sensor = lane_sensor
        self.collision_sensor = collision_sensor
        self.early_terminate_on_collision = early_terminate_on_collision
        self.manual_control = manual_control
        self.auto_control = auto_control
        self.squash_action_logits = squash_action_logits
        self.reward_function = reward_function
        self.send_measurements = send_measurements
        self.log_images = log_images
        self.log_measurements = log_measurements
        self.verbose = verbose

    @staticmethod
    def parse_xml_node(node):
        """
        static method to initialize an ActorConfiguration from a given ET tree
        """
        assert node.attrib.get("name", None) is not None, "XML attribute error. The 'actor' elements require a 'name' key."
        assert node.attrib.get('framestack', 1) in [1, 2], "XML attribute error. Only a framestack in [1,2] is supported."
        assert node.attrib.get('camera_type', 'rgb') in [ct.name for ct in CAMERA_TYPES], f"XML attribute error. Camera type `{node.attrib['camera_type']}` not available. Choose one between {[ct.name for ct in CAMERA_TYPES]}."

        name = node.attrib.get('name', None)
        config = {
            'render': node.attrib.get('render', None),
            'enable_planner': node.attrib.get('enable_planner', None),
            'camera_type': node.attrib.get('camera_type', None),
            'camera_position': node.attrib.get('camera_position', None),
            'framestack': node.attrib.get('framestack', None),
            'lane_sensor': node.attrib.get('lane_sensor', None),
            'collision_sensor': node.attrib.get('collision_sensor', None),
            'early_terminate_on_collision': node.attrib.get('early_terminate_on_collision', None),
            'manual_control': node.attrib.get('manual_control', None),
            'auto_control': node.attrib.get('auto_control', None),
            'squash_action_logits': node.attrib.get('squash_action_logits', None),
            'reward_function': node.attrib.get('reward_function', None),
            'send_measurements': node.attrib.get('send_measurements', None),
            'log_images': node.attrib.get('log_images', None),
            'log_measurements': node.attrib.get('log_measurements', None),
            'verbose': node.attrib.get('verbose', None)
        }

        return ActorConfiguration(name).update(config)

    def update(self, conf):
        assert conf.get('framestack', None) is None or int(conf['framestack']) in [1, 2], "Only a framestack in [1,2] is supported."
        assert conf.get('camera_type', None) is None or conf['camera_type'] in [ct.name for ct in CAMERA_TYPES], f"Camera type `{conf['camera_type']}` not available. Choose one between {[ct.name for ct in CAMERA_TYPES]}."

        if conf.get("render", None) is not None: self.render = strtobool(conf["render"])
        if conf.get("enable_planner", None) is not None: self.enable_planner = strtobool(conf["enable_planner"])
        if conf.get("camera_type", None) is not None: self.camera_type = conf["camera_type"]
        if conf.get("camera_position", None) is not None: self.camera_position = int(conf["camera_position"])
        if conf.get("framestack", None) is not None: self.framestack = int(conf["framestack"])
        if conf.get("lane_sensor", None) is not None: self.lane_sensor = strtobool(conf["lane_sensor"])
        if conf.get("collision_sensor", None) is not None: self.collision_sensor = strtobool(conf["collision_sensor"])
        if conf.get("early_terminate_on_collision", None) is not None: self.early_terminate_on_collision = strtobool(conf["early_terminate_on_collision"])
        if conf.get("manual_control", None) is not None: self.manual_control = strtobool(conf["manual_control"])
        if conf.get("auto_control", None) is not None: self.auto_control = strtobool(conf["auto_control"])
        if conf.get("squash_action_logits", None) is not None: self.squash_action_logits = strtobool(conf["squash_action_logits"])
        if conf.get("reward_function", None) is not None: self.reward_function = conf["reward_function"]
        if conf.get("send_measurements", None) is not None: self.send_measurements = strtobool(conf["send_measurements"])
        if conf.get("log_images", None) is not None: self.log_images = strtobool(conf["log_images"])
        if conf.get("log_measurements", None) is not None: self.log_measurements = strtobool(conf["log_measurements"])
        if conf.get("verbose", None) is not None: self.verbose = strtobool(conf["verbose"])

        return self


class ObjectsConfiguration(object):
    """
    This is a configuration base class to hold model and transform attributes
    """
    def __init__(self, name, type="vehicle_4W", model=None, start=None, end=None, speed=0, autopilot=False, color=None):
        self.name = name
        self.type = type
        self.model = model
        self.start = start if start is not None else [0,0,0,0]
        self.end = end if end is not None else [10,10,0]
        self.speed = speed
        self.autopilot = autopilot
        self.color = color

    @staticmethod
    def parse_xml_node(node):
        """
        static method to initialize an ActorConfigurationData from a given ET tree
        """
        assert node.attrib.get("name", None) is not None, "XML attribute error. The 'object' elements require a 'name' key."

        name = node.attrib.get('name', None)
        config = {
            "type": node.attrib.get('type', None),
            "model": node.attrib.get('model', None),
            "start_x": node.attrib.get('start_x', None),
            "start_y": node.attrib.get('start_y', None),
            "start_z": node.attrib.get('start_z', None),
            "yaw": node.attrib.get('yaw', None),
            "end_x": node.attrib.get('end_x', None),
            "end_y": node.attrib.get('end_y', None),
            "end_z": node.attrib.get('end_z', None),
            "speed": node.attrib.get('speed', None),
            "autopilot": node.attrib.get('autopilot', None),
            "color": node.attrib.get('color', None),
        }

        return ObjectsConfiguration(name).update(config)

    def update(self, conf):
        if conf.get("type", None) is not None: self.type = conf["type"]
        if conf.get("model", None) is not None: self.model = conf["model"]
        if conf.get("start", None) is not None: self.start = conf["start"]
        if conf.get("start_x", None) is not None: self.start[0] = float(conf["start_x"])
        if conf.get("start_y", None) is not None: self.start[1] = float(conf["start_y"])
        if conf.get("start_z", None) is not None: self.start[2] = float(conf["start_z"])
        if conf.get("yaw", None) is not None: self.start[3] = float(conf["yaw"])
        if conf.get("end", None) is not None: self.end = conf["end"]
        if conf.get("end_x", None) is not None: self.end[0] = float(conf["end_x"])
        if conf.get("end_y", None) is not None: self.end[1] = float(conf["end_y"])
        if conf.get("end_z", None) is not None: self.end[2] = float(conf["end_z"])
        if conf.get("speed", None) is not None: self.speed = float(conf["speed"])
        if conf.get("autopilot", None) is not None: self.autopilot = strtobool(conf["autopilot"])
        if conf.get("color", None) is not None: self.color = conf["color"]

        return self


class ScenarioConfiguration(object):
    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario (e.g. ControlLoss_1)
    - type is the class of scenario (e.g. ControlLoss)
    """

    def __init__(self, name, type=None, town="Town01", objects=None, num_pedestrians=0, num_vehicles=0, weathers=None):
        self.name = name
        self.type = type
        self.town = town
        self.objects = objects if objects is not None else {}
        self.num_pedestrians = num_pedestrians
        self.num_vehicles = num_vehicles
        self.weathers = weathers if weathers is not None else [carla.WeatherParameters.Default]

    @staticmethod
    def parse_xml_node(node):
        """
        static method to initialize an ActorConfigurationData from a given ET tree
        """
        assert node.attrib.get("name", None) is not None, "XML attribute error. The 'scenario' elements require a 'name' key."

        name = node.attrib.get('name', None)
        config = {
            "type": node.attrib.get('type', None),
            "town": node.attrib.get('town', None),
            "num_pedestrians": node.attrib.get('npc_pedestrians', None),
            "num_vehicles": node.attrib.get('npc_vehicles', None)
        }

        objects = {}
        for object in node.iter("object"):
            o = ObjectsConfiguration.parse_xml_node(object)
            if o.name in objects:
                warnings.warn("Multiple `object` elements with same name identifier in XML configuration.")
            objects.update({o.name: o})

        weathers = [] if len(list(node.iter("weather"))) > 0 else [WEATHERS[node.attrib.get('weather', "Default")]]
        for weather_node in node.iter("weather"):
            weather = carla.WeatherParameters()
            weather.cloudiness = float(weather_node.attrib.get("cloudiness", 0))
            weather.precipitation = float(weather_node.attrib.get("precipitation", 0))
            weather.precipitation_deposits = float(weather_node.attrib.get("precipitation_deposits", 0))
            weather.wind_intensity = float(weather_node.attrib.get("wind_intensity", 0.35))
            weather.sun_azimuth_angle = float(weather_node.attrib.get("sun_azimuth_angle", 0.0))
            weather.sun_altitude_angle = float(weather_node.attrib.get("sun_altitude_angle", 15.0))
            weather.fog_density = float(weather_node.attrib.get("fog_density", 0.0))
            weather.fog_distance = float(weather_node.attrib.get("fog_distance", 0.0))
            weather.wetness = float(weather_node.attrib.get("wetness", 0.0))
            weathers.append(weather)

        return ScenarioConfiguration(name, objects=objects, weathers=weathers).update(config)

    def update(self, conf):
        if conf.get("type", None) is not None: self.type = conf["type"]
        if conf.get("town", None) is not None: self.town = conf["town"]

        if len(conf.get("objects", [])) > 0:
            self.objects = {}
            if isinstance(conf["objects"], list):
                for new_object_dict in conf["objects"]:
                    assert new_object_dict.get("name", None) is not None, "The 'object' elements require a 'name' key."
                    new_object = ObjectsConfiguration(new_object_dict["name"]).update(new_object_dict)
                    self.objects.update({new_object.name: new_object})
            elif isinstance(conf["objects"], dict):
                for name, new_object_dict in conf["objects"].items():
                    new_object = ObjectsConfiguration(name).update(new_object_dict)
                    self.objects.update({new_object.name: new_object})

        if conf.get("num_pedestrians", None) is not None: self.num_pedestrians = int(conf["num_pedestrians"])
        if conf.get("num_vehicles", None) is not None: self.num_vehicles = int(conf["num_vehicles"])

        if len(conf.get("weathers", []))>0:
            self.weathers = []
            for weather_node in conf["weathers"]:
                weather = carla.WeatherParameters()
                weather.cloudiness = float(weather_node.get("cloudiness", 0))
                weather.precipitation = float(weather_node.get("precipitation", 0))
                weather.precipitation_deposits = float(weather_node.get("precipitation_deposits", 0))
                weather.wind_intensity = float(weather_node.get("wind_intensity", 0.35))
                weather.sun_azimuth_angle = float(weather_node.get("sun_azimuth_angle", 0.0))
                weather.sun_altitude_angle = float(weather_node.get("sun_altitude_angle", 15.0))
                weather.fog_density = float(weather_node.get("fog_density", 0.0))
                weather.fog_distance = float(weather_node.get("fog_distance", 0.0))
                weather.wetness = float(weather_node.get("wetness", 0.0))
                self.weathers.append(weather)

        return self


class Configuration(object):
    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario (e.g. ControlLoss_1)
    - type is the class of scenario (e.g. ControlLoss)
    """
    def __init__(self, actors, scenarios):
        self.actors = actors
        self.scenarios = scenarios
        self._check_actors_consistency()

    def _check_actors_consistency(self):
        for n, s in self.scenarios.items():
            if len(set(self.actors.keys()).difference(set(s.objects.keys()))) != 0:
                raise ValueError(f"The `name` of `actor` elements in `actors` do not correspond the `name` of controllable objects in scenario `{n}`.")

    @staticmethod
    def parse_xml(config_file_name):
        """
        Parse the  provided as argument.
        Args:
          config_file_name (str): Configuration XML file name, srunner compatible.

        Returns:
          A configuration object.
        """

        actors = {}
        scenarios = {}
        tree_node = ET.parse(config_file_name)

        actors_node = tree_node.find("actors")
        scenarios_node = tree_node.find("scenarios")

        for actor_node in actors_node.iter("actor"):
            actor = ActorConfiguration.parse_xml_node(actor_node)
            actors.update({actor.name: actor})
        for scenario_node in scenarios_node.iter("scenario"):
            scenario = ScenarioConfiguration.parse_xml_node(scenario_node)
            scenarios.update({scenario.name: scenario})

        return Configuration(actors, scenarios)

    @staticmethod
    def parse(configs):
        """
        Parse the configuration dictionary provided as argument.
        Args:
          configs (dict): Dictionary of configurations.

        Returns:
          A configuration object.
        """
        assert isinstance(configs.get("actors", {}), Iterable) and isinstance(configs.get("scenarios", {}), Iterable), "'actors' and 'scenarios' attributes in the configuration should be iterable objects."
        assert len(configs.get("scenarios", {})) > 0, "'scenarios' attribute should contain at least one element."

        actors = {}
        scenarios = {}

        if "actors" in configs:
            if isinstance(configs["actors"], list):
                for actor_dict in configs["actors"]:
                    assert actor_dict.get("name", None) is not None, "The 'actor' elements require a 'name' key."
                    actor = ActorConfiguration(actor_dict["name"]).update(actor_dict)
                    actors.update({actor.name: actor})
            elif isinstance(configs["actors"], dict):
                for name, actor_dict in configs["actors"].items():
                    actor = ActorConfiguration(name).update(actor_dict)
                    actors.update({name: actor})
            else: raise TypeError("'actors' node type error.")

        if isinstance(configs["scenarios"], list):
            for scenario_dict in configs["scenarios"]:
                assert scenario_dict.get("name", None) is not None, "The 'scenario' elements require a 'name' key."
                scenario = ScenarioConfiguration(scenario_dict["name"]).update(scenario_dict)
                scenarios.update({scenario.name: scenario})
        elif isinstance(configs["scenarios"], dict):
            for name, scenario_dict in configs["scenarios"].items():
                scenario = ScenarioConfiguration(name).update(scenario_dict)
                scenarios.update({name: scenario})
        else: raise TypeError("'scenarios' node type error.")

        return Configuration(actors, scenarios)

    def update(self, conf):
        assert isinstance(conf.get("actors", {}), Iterable) and isinstance(conf.get("scenarios", []), Iterable), "'actors' and 'scenarios' attributes in the configuration should be iterable objects"

        # update actors configuration inserting new actors or overwriting them individually
        if len(conf.get("actors", [])) > 0:
            # self.actors = []  not intended behaviour
            if isinstance(conf["actors"], list):
                for actor_dict in conf["actors"]:
                    assert actor_dict.get("name", None) is not None, "The 'actor' elements require a 'name' key."
                    if actor_dict["name"] in self.actors:
                        self.actors[actor_dict["name"]].update(actor_dict)
                    else:
                        actor = ActorConfiguration(actor_dict["name"]).update(actor_dict)
                        self.actors.update({actor.name: actor})
            elif isinstance(conf["actors"], dict):
                for name, actor_dict in conf["actors"].items():
                    if name in self.actors:
                        self.actors[name].update(actor_dict)
                    else:
                        actor = ActorConfiguration(name).update(actor_dict)
                        self.actors.update({name: actor})
            else: raise TypeError("'actors' node type error.")

        # update scenarios configuration inserting new scenarios or overwriting them individually
        if len(conf.get("scenarios", [])) > 0:
            # self.actors = []  not intended behaviour
            if isinstance(conf["scenarios"], list):
                for scenario_dict in conf["scenarios"]:
                    assert scenario_dict.get("name", None) is not None, "The 'scenario' elements require a 'name' key."
                    if scenario_dict["name"] in self.scenarios:
                        self.scenarios[scenario_dict["name"]].update(scenario_dict)
                    else:
                        scenario = ScenarioConfiguration(scenario_dict["name"]).update(scenario_dict)
                        self.scenarios.update({scenario.name: scenario})
            elif isinstance(conf["scenarios"], dict):
                for name, scenario_dict in conf["scenarios"].items():
                    if name in self.scenarios:
                        self.scenarios[name].update(scenario_dict)
                    else:
                        scenario = ScenarioConfiguration(name).update(scenario_dict)
                        self.scenarios.update({name: scenario})
            else: raise TypeError("'scenarios' node type error.")

        self._check_actors_consistency()

        return self

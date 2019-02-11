#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import itertools

try:
    sys.path.append(
        glob.glob('../../carla/PythonAPI/**/*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import math
import random
import json
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "-v",
    "--viz-map",
    action="store_true",
    default=False,
    help="Show map topology")
parser.add_argument(
    "-e",
    "--export-node-coord-map",
    help="Export the map between spawn_points and node_ids"
    " to the JSON file")

args = parser.parse_args()


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a),
                              2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(
        yaw=180 + angle, pitch=-15))


def show_map_topology(world):
    import matplotlib.pyplot as plt
    topology = world.get_map().get_topology()
    for segment in topology:
        x1, y1 = segment[0].transform.location.x, segment[
            0].transform.location.y
        x2, y2 = segment[1].transform.location.x, segment[
            1].transform.location.y
        plt.plot([x1, x2], [y1, y2], marker='o')
    plt.gca().invert_yaxis()
    plt.show()
    input()


def map_spawn_point_to_node(world) -> dict:
    node_coord_map = dict()
    node_id = itertools.count()

    for location in world.get_map().get_spawn_points():

        node_coord_map[next(node_id)] = [
            location.location.x, location.location.y, location.location.z
        ]
    return node_coord_map


def start_walker(ped1):
    """Set a walker in forward motion"""
    ped_cmd = carla.WalkerControl()
    ped_cmd.speed = 1.778
    ped1.apply_control(ped_cmd)


def stop_walker(ped1):
    """Halt a walker"""
    ped_cmd = carla.WalkerControl()
    ped_cmd.speed = 0.0
    ped1.apply_control(ped_cmd)


spawn_locs = {
    "car1": {
        "S": [19, -133, 0.3],
        "E": [104, -132, 8]
    },
    "car2": {
        "S": [84, -123, 8],
        "E": [41, -137, 8]
    },
    "ped1": {
        "S": [74, -126, 8],
        "E": [92, -125, 8]
    }
}


def start_scenario():
    car_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
    car_bp.set_attribute("role_name", "hero")
    car1_loc_s = carla.Location(*spawn_locs["car1"]["S"])
    car2_loc_s = carla.Location(*spawn_locs["car2"]["S"])
    ped1_loc_s = carla.Location(*spawn_locs["ped1"]["S"])

    car1 = world.spawn_actor(
        car_bp, carla.Transform(car1_loc_s, carla.Rotation(yaw=0)))
    car1.set_autopilot(True)
    car2 = world.spawn_actor(
        car_bp, carla.Transform(car2_loc_s, carla.Rotation(yaw=-90)))
    car2.set_autopilot(True)

    ped_bp = random.choice(world.get_blueprint_library().filter('walker'))
    ped1 = world.spawn_actor(
        ped_bp, carla.Transform(ped1_loc_s, carla.Rotation(yaw=0)))
    start_walker(ped1)


def get_traffic_lights(loc=carla.Location(0, 0, 0)):
    tls = {}
    for a in world.get_actors().filter("traffic.traffic_light"):
        tls[a.id] = [
            a.get_location().x,
            a.get_location().y,
            a.get_location().z
        ]
        print("ID:", a.id, "loc:",
              a.get_location().x,
              a.get_location().y,
              a.get_location().z)
    # Sort traffic lights by their location.x
    ans = sorted(tls.items(), key=lambda kv: kv[1][0])
    return ans

    # Main:


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
if args.export_node_coord_map:
    node_coord_map = map_spawn_point_to_node(world)
    json.dump(
        node_coord_map, open("TOWN04.json", 'w'), indent=2, sort_keys=True)

if args.viz_map:
    show_map_topology(world.get_map())

spectator = world.get_spectator()
spectator_loc = carla.Location(70, -123, 9)
spectator.set_transform(get_transform(spectator_loc, angle=160.0))
start_scenario()
"""
    try:


        angle = 0
        while angle < 90:
            timestamp = world.wait_for_tick()
            angle += timestamp.delta_seconds * 90.0
            spectator.set_transform(get_transform(vehicle.get_location(),
             angle - 90))
        # spectator.set_transform(get_transform(vehicle.get_location(), angle))
        # input("Enter Key")

    finally:

        vehicle.destroy()
"""

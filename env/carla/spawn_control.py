#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change camera position
    `            : next camera sensor
    [1-9]        : swtich camera view among vehicles
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('env/carla/**/**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("IndexErr loading egg")
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc
from carla import World as CarlaWorld
from carla import DebugHelper

import argparse
import logging
import random
import time
import re
import weakref
import json
import math
import collections


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')
       
#===============================================================================
#-- VehicleManager -------------------------------------------------------------
#===============================================================================
class VehicleManager(object):
    def __init__(self, vehicle, autopilot_enabled =False):
        self._vehicle = vehicle
        self._autopilot_enabled = autopilot_enabled
        self._hud = None  #TODO 
        self._collision_sensor = CollisionSensor(self._vehicle, self._hud)       
        self._lane_invasion_sensor = LaneInvasionSensor(self._vehicle, self._hud)
        self._start_pos = None
        self._end_pos = None
        self._start_coord = None
        self._end_coord = None
        self.prev_measurement = None
       
    def get_location(self):
        return self._vehicle.get_location() 
    
    def get_velocity(self):
        return self._vehicle.get_velocity() 
    
    #TODO: refresh in each render frame 
    #TODO: waypoints doesn't refresh when restart a new pygame
    #self.map.get_waypoint(loc1)      
    def draw_waypoints(self, helper):           
            loc = self.get_location()
            vel = self.get_velocity()
            abs_vel = (vel.x)**2 + (vel.y)**2 + (vel.z)**2
            if abs_vel < 0.5 :
                print('draw a point')
                helper.draw_point(loc)
            else:
                print('draw a line')
                loc2 = loc + carla.Location(x=vel.x , y=vel.y,  z=0)                 
                helper.draw_line(loc, loc2)    

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
    # issue#17, CPP code can be viewed at ACarlaVehicleController:IntersectPlayerWithRoadMap 
    def offroad_invasion(self):
        return 0          
   
    #TODO: for demo, all vehicles has same start_pos & end_pos 
    #but in reality, need find the nearest pos at each spawn location 
    def _nearest_pos(self, vid):
        pass 
        
    def _pos_coord(self, scenario):
        POS_COOR_MAP = json.load(open("env/carla/POS_COOR/pos_cordi_map_town1.txt"))       
        start_id = scenario["start_pos_id"]
        end_id = scenario["end_pos_id"]        
        self._start_pos =  POS_COOR_MAP[str(start_id)]
        self._end_pos = POS_COOR_MAP[str(end_id)]
        self._start_coord =  [ self._start_pos[0]// 100 ,   self._start_pos[1] // 100 ] 
        self._end_coord =  [ self._end_pos[0] // 100 ,   self._end_pos[1] // 100 ]            
        return (self._start_pos, self._end_pos, self._start_coord, self._end_coord)
       
    def read_observation(self, scenario):      
        c_vehicles, c_pedestrains, c_other = self.dynamic_collided()       
        c_offline = self.offlane_invasion()
        c_offroad = self.offroad_invasion()
        start_pos, end_pos, start_coord, end_coord = self._pos_coord(scenario)
        cur_ = self._vehicle.get_transform()
        cur_x =  cur_.location.x
        cur_y = cur_.location.y
        x_orient = cur_.rotation 
        y_orient = cur_.rotation              
        distance_to_goal_euclidean = float(np.linalg.norm( [cur_x -  end_pos[0], cur_y - end_pos[1]]) / 100)        
        distance_to_goal = distance_to_goal_euclidean         
        
        vehicle_data = {
#            "episode_id": 0 ,
#            "step": self.num_steps,
            "x": cur_x,
            "y": cur_y,
            "x_orient": x_orient ,
            "y_orient": y_orient ,
            "forward_speed": self._vehicle.get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": c_vehicles ,
            "collision_pedestrians": c_pedestrains ,
            "collision_other": c_other ,
            "intersection_offroad": c_offroad , 
            "intersection_otherlane": c_offline ,
#            "weather": None ,
#            "map": self.config["server_map"],
            "start_coord": start_coord,
            "end_coord": end_coord
#            "current_scenario": self.scenario , 
#            "x_res": self.config["x_res"],
#            "y_res": self.config["y_res"],
#            "num_vehicles": self.num_vehicles , 
#            "num_pedestrians": self.config["num_pedestrians"],
#            "max_steps": 1000
#            "next_command": next_command,
#            "previous_actions": previous_action,  # Dict of action 1 per actor
#             "previous_rewards": previous_rewards  # Dict of rewards 1 per actor            
        }   
        
        return vehicle_data 

    #TODO    
    def destroy(self):
        pass

#===============================================================================
#-- CarlaMap -------------------------------------------------------------------
#===============================================================================
from .carla.converter import Converter

class CarlaMap(object):
    def __init__(self, city, pixel_density=0.1643, node_density=50):
        dir_path = os.path.dirname(__file__)
        city_file = os.path.join(dir_path, city + '.txt')    
        self._converter = Converter(city_file, pixel_density, node_density)  
    def convert_to_pixel(self, input_data):
        """
        Receives a data type (Can Be Node or World )
        :param input_data: position in some coordinate
        :return: A node object
        """
        return self._converter.convert_to_pixel(input_data)

    def convert_to_world(self, input_data):
        """
        Receives a data type (Can Be Pixel or Node )
        :param input_data: position in some coordinate
        :return: A node object
        """
        return self._converter.convert_to_world(input_data)  

#===============================================================================
#-- Detecter -------------------------------------------------------------------
#===============================================================================
from .Transform import transform_points
class Detecter(object):
    MAX_ITERATION = 20 
    SPAWN_COLLISION = True
    def __init__(self, location, actor_list):
        self._first_center = location 
        self._location = location
        self._actors = actor_list

#first transform the 8bbox vertices respect to the bbox transform    
#then transform the vertices respect to vehicle(relative to the world coord)
#TODO: carla.Transform.transform_points() deprecated from v0.9.x 
    def _bbox_vertices(self, vehicle):
        ext = vehicle.bounding_box.extent    
#8bbox vertices relative to (0,0,0) locally         
        bbox = np.array([
               [  ext.x,   ext.y,   ext.z],
               [- ext.x,   ext.y,   ext.z],
               [  ext.x, - ext.y,   ext.z],
               [- ext.x, - ext.y,   ext.z],
               [  ext.x,   ext.y, - ext.z],
              [- ext.x,   ext.y, - ext.z],
              [  ext.x, - ext.y, - ext.z],
              [- ext.x, - ext.y, - ext.z]    
            ]) 

        vehicle_transform = carla.Transform(vehicle.get_location())
        bbox_transform = carla.Transform(vehicle.bounding_box.location) 
        bbox = transform_points(bbox_transform, bbox)
        bbox = transform_points(vehicle_transform, bbox)   
        
        return bbox 
    
    def _min_max(self, a, b):
        min_ = min(a, b)
        max_ = max(a, b)
        return [min_, max_]
    

#TODO: need reimplement, not familar with Python numpy

    def _cubic(self, bbox):
        n1 = np.squeeze(np.asarray(bbox[0] - bbox[1]) )#x direction
        n2 = np.squeeze(np.asarray(bbox[0] - bbox[2]) )#y direction
        n3 = np.squeeze(np.asarray(bbox[0] - bbox[4]) )#z direction 
                   
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
    
    def _vel_update(self, vel ):
        if vel < -0.1 or vel > 0.1 :
            pass
        if -0.1 < vel <= 0 :
            vel = -0.1 
        elif 0.1 > vel >= 0 :
            vel = 0.1
        return vel 
                      
    def collision(self):
        for vehicle in self._actors:
            vel = vehicle.get_velocity()
            abs_vel = (vel.x)**2 + (vel.y)**2 + (vel.z)**2
#            print('current vehicle %2d' % vehicle.id + '  velocity(%6.4f %6.4f %6.4f)' % (vel.x, vel.y, vel.z) )
            if abs_vel < 0.1 :   
                print('current velocity less than 0.1 Km/h, failed spawn new vehicle')
                return self.SPAWN_COLLISION    
                
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
                #adding bounding box size first
                ext = vehicle.bounding_box.extent 
                self._location.x += ext.x  
                self._location.y += ext.y 
                self._location.z += ext.z 
                if  p1 >= min1 and  p1 <= max1:                    
                    self._location.x += self._vel_update(vel.x) 
                    print('collision happens in x direction,  adding spwan x-location by %4.2f' %  self._vel_update(vel.x) )
                    continue                    
                elif p2 >= min2 and p2 <= max2 :                 
                    self._location.y +=    self._vel_update(vel.y) 
                    print("collision happens in y direction, adding spwan y-location by %4.2f" % self._vel_update(vel.y)  )
                    continue                                        
                elif p3 >= min3 and p3 <= max3 :                   
                    self._location.z +=  self._vel_update(vel.z) 
                    print("collision happens in z direction, adding spwan z-location by %4.2f" %  self._vel_update(vel.z)  )
                    continue      
                else: 
                    break
                print("no collision with %2d" % vehicle.id )
                         
            print('  will spawn a vehicle at location: (%4.2f, %4.2f, %4.2f)' % (self._location.x,  self._location.y, self._location.z))             
            return self._location

                   
# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================   

START_POSITION = carla.Transform(carla.Location(x=180.0, y=199.0, z=40.0)) 
END_POSITION = carla.Transform(carla.Location(x=217.0, y=195.0, z=40.0))
GLOBAL_CAM_POSITION = carla.Transform(carla.Location(x=180.0, y=199.0, z= 45.0))

from .scenarios import DEFAULT_SCENARIO_TOWN1

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name

# Default environment configuration
ENV_CONFIG = {
    "log_images": True,
    "enable_planner": False,
    "render": True,  
    "framestack": 2, 
    "x_res": 80,
    "y_res": 80,
    "server_map": "/Game/Carla/Maps/Town01", 
    "scenarios": DEFAULT_SCENARIO_TOWN1 ,
    "num_pedestrians" : 10 , 
    "discrete_actions": True,
    "manual_control": False,
}

#TODO: replace vehicle_list with vehicle_manager_list
#TODO: remove _vehicle property
class World(object):     
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.map = self.world.get_map() 
        self.hud = hud
        self.num_vehicles = 1 
        blueprint = self._get_random_blueprint()
        self.vehicle_list = [] 
        self.vehicle_manager_list = [] 
        self._vehicle = self.world.spawn_actor(blueprint, START_POSITION)
        self.vehicle_list.append(self._vehicle)
        vmanager = VehicleManager(self._vehicle) 
        self.vehicle_manager_list.append(vmanager)
        self.collision_sensor = CollisionSensor(self._vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self._vehicle, self.hud)
        self.camera_manager_list = []  
        self.global_camera = CameraManager(self.world, self.hud, GLOBAL_CAM_POSITION)
        self.global_camera.set_sensor(0, notify=False)
        self.camera_manager_list.append(self.global_camera) 
        self._camera_manager = CameraManager(self._vehicle, self.hud)
        self._camera_manager.set_sensor(0, notify=False)
        self.camera_manager_list.append(self._camera_manager) 
        self.camera_index = 0  #set global camera
        self.controller = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.draw_waypoints()
        
        #integration with MultiCarlaEnv
        self.config = ENV_CONFIG 
        self.num_steps = 0 
        self.scenario = ENV_CONFIG["scenarios"] 
        self.prev_measurements = {}        
        self.prev_measurements[0] = vmanager.read_observation(self.scenario)
      

                
    def init_global_camera(self):
        self.global_camera = CameraManager(self.world, self.hud)
        self.global_camera.set_sensor(0, notify=False)
        self.camera_manager_list.append(self.global_camera)       
        
    def restart(self):
        cam_index = self._camera_manager._index
        cam_pos_index = self._camera_manager._transform_index
        start_pose = self._vehicle.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        blueprint = self._get_random_blueprint()
        self.camera_manager_list=[]
        self.vehicle_list = [] 
        self.vehicle_manager_list = [] 
        self.prev_measurements = {}        
        self.destroy()
        self.num_vehicles = 1
        self.camera_index = 0 
        self.init_global_camera()
        self._vehicle = self.world.spawn_actor(blueprint, start_pose)
        self.vehicle_list.append(self._vehicle)
        self.vehicle_manager_list.append(VehicleManager(self._vehicle))
        self.prev_measurements[0] = vmanager.read_observation(self.scenario)
        self.collision_sensor = CollisionSensor(self._vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self._vehicle, self.hud) 
        
        self._camera_manager = CameraManager(self.vehicle, self.hud)
        self._camera_manager._transform_index = cam_pos_index
        self._camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager_list.append(self._camera_manager) 
        actor_type = ' '.join(self._vehicle.type_id.replace('_', '.').title().split('.')[1:])
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self._vehicle.get_world().set_weather(preset[0])
        
    def get_num_of_vehicles(self):
        return self.num_vehicles 
    
    def get_vehicle_managers(self):
        return self.vehicle_manager_list
           
    def get_cameras(self):
        return self.camera_manager_list

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display, camera_index=0):  
        if camera_index == 0 : 
            self.global_camera.render(display)  
        else :
            self.camera_manager_list[camera_index].render(display)
          
        self.hud.render(display)
           
    def destroy(self):               
        while len(self.camera_manager_list) != 0:
            _cmanager = self.camera_manager_list.pop()
            _vehicle = self.vehicle_list.pop() if self.vehicle_list else None
            for actor in [_cmanager.sensor,   _vehicle]:
                if actor is not None:
                    actor.destroy() 
        for actor in [self.collision_sensor.sensor, self.lane_invasion_sensor.sensor ]:
            if actor is not None:
                actor.destroy() 
                
        for vm in self.vehicle_manager_list:
            vm.destroy()           

    def _get_random_blueprint(self):
        bp = random.choice(self.world.get_blueprint_library().filter('vehicle'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp
    
    def spawn_new_vehicle(self, location):
        blueprint = self._get_random_blueprint()	
        detector = Detecter(location,  self.vehicle_list)  
        s_location = detector.collision()
        if type(s_location) == bool :
            return None
        #TODO: yaw to along street line
        transform = carla.Transform(carla.Location(x=s_location.x, y=s_location.y, z=s_location.z), carla.Rotation(yaw=0.0))                      
        vehicle = self.world.try_spawn_actor(blueprint, transform)  
        if vehicle is not None:
            self.vehicle_list.append(vehicle)             
            vmanager = VehicleManager(vehicle)                   
            self.prev_measurements[self.num_vehicles] = vmanager.read_observation(self.scenario)      
            self.num_vehicles += 1 
            self.vehicle_manager_list.append(vmanager)
            return vehicle
                       
    def reward_computing(self): 
        for vid in np.arange(self.get_num_of_vehicles()) :
            vmanager = self.vehicle_manager_list[vid]
            current = vmanager.read_observation(self.scenario)
            prev = self.prev_measurements[vid] 
        
            reward = 0.0         

            cur_dist = current["distance_to_goal"]
            prev_dist = prev["distance_to_goal"]

            # Distance travelled toward the goal in m
            reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

            # Speed reward, up 30.0 (km/h)
            reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

            # New collision damage
            new_damage = (
                current["collision_vehicles"] + current["collision_pedestrians"] +
                current["collision_other"] - prev["collision_vehicles"] -
                prev["collision_pedestrians"] - prev["collision_other"])
            if new_damage:
                reward -= 100.0

            # Sidewalk intersection
            reward -= current["intersection_offroad"]         

            # Opposite lane intersection
            reward -= current["intersection_otherlane"] 
            
            # Reached goal
#            if current["next_command"] == "REACH_GOAL":
#                reward += 100.0  
#            print('VEHICLE %d' % vid + 'rewarding is %2d' % reward)
            self.prev_measurements[vid] = current      
            
    def draw_waypoints(self):      
        helper = self.world.debug         
        for vid in np.arange(self.get_num_of_vehicles()) :
            vmanager = self.vehicle_manager_list[vid]
            vmanager.draw_waypoints(helper)
 
                    
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

#TODO: rm  _autopilot_enabled & world._vehicle
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world._vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        
    def _cur_vehicle_manager(self, world):
        vm_list = world.get_vehicle_managers() 
        if world.camera_index != 0 :
            return vm_list[world.camera_index-1] 
        else :
            return None
        
    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world._camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world._camera_manager.next_sensor()
                elif event.key >= K_0 and event.key <= K_9:   #K_0 ==  48 ? 
                    if len(world.camera_manager_list) > event.key-48:
                        world.camera_index = event.key-48
                    else :
                        world.camera_index = 0 
                        pass
                elif event.key == K_r:
                    world._camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.reverse = not self._control.reverse
                elif event.key == K_p:
                    cur_vehicle_m = self._cur_vehicle_manager(world)
                    cur_vehicle_m.set_autopilot(not cur_vehicle_m.get_autopilot())
                    world.hud.notification('Autopilot %s' % ('On' if cur_vehicle_m.get_autopilot() else 'Off'))
            elif event.type == pygame.MOUSEBUTTONUP:
                benchmark_transform = world.vehicle_list[-1].get_transform()
                vehicle = world.spawn_new_vehicle(benchmark_transform.location)
                time.sleep(3)  
                if vehicle is not None:
                    cmanager = CameraManager(vehicle, world.hud)   
                    cmanager.set_sensor(0 , notify=False)
                    world.camera_manager_list.append(cmanager)  
        
            vm = self._cur_vehicle_manager(world)
            if vm is not None:
                if not vm.get_autopilot(): 
                    self._parse_keys(pygame.key.get_pressed(), clock.get_time())
                    vm.apply_control(self._control)   
                else :
                    vm.set_autopilot(vm.get_autopilot())
             
            #print('collision with vehicle %2d, people %2d, others %2d' % world.collision_sensor.dynamic_collided())
            

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-3 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:          
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        mono = next(x for x in pygame.font.get_fonts() if 'mono' in x) # hope for the best...
        mono = pygame.font.match_font(mono, bold=True)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.client_fps = 0
        self.server_fps = 0

    def tick(self, world, clock):
        self.client_fps = clock.get_fps()
        self._notifications.tick(world, clock)

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        self._notifications.render(display)
        self.help.render(display)
        fps_text = 'client: %02d FPS; server: %02d FPS' % (self.client_fps, self.server_fps)
        fps = self._font_mono.render(fps_text, True, (60, 60, 60))
        display.blit(fps, (6, 4))


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
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
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

            
            
# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = [] 
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self._offlane = 0
        self._off_lane_percentage = 0 
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        

    def get_offlane_percentage(self):
        return self._off_lane_percentage  
        
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
#        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
#        self._hud.notification('Crossed line %s' % ' and '.join(text))
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        print('VEHICLE %s' % (self._parent).id + ' crossed line %s' % ' and '.join(text)) 
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
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        
    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history :
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

        print('vehicle %s ' % (self._parent).id + ' collision with %2d vehicles, %2d people, %2d others' %  self.dynamic_collided())            
        _cur = event.other_actor
        if _cur.id == 0 : #the static world objects 
            if _cur.type_id in self.collision_type_id_set :
                return
            else :
                self.collision_type_id_set.add(_cur.type_id)
        else :
            if _cur.id in self.collision_id_set :
                return 
            else :
                self.collision_id_set.add(_cur.id)
                
        collided_type = type(_cur).__name__
        if collided_type == 'Vehicle' :
            self.collision_vehicles += 1 
        elif collided_type == 'Pedestrain' :   
            self.collision_pedestrains += 1 
        elif collided_type == 'Actor' :
            self.collision_other += 1 
        else :
            pass
    
    def _reset(self):
        self.collision_vehicles = 0 
        self.collision_pedestrains = 0 
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()
     
    def dynamic_collided(self):
        return (self.collision_vehicles, self.collision_pedestrains, self.collision_other) 
            
                      

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, transform=None):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            transform ]
        if transform is not None:
            self._transform_index = 2
        else:
            self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)']]
        if transform is not None:
            self._world = self._parent
        else:
            self._world = self._parent.get_world()
        bp_library = self._world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None
        self._server_clock = pygame.time.Clock()

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            if self._world == self._parent:
                self.sensor = self._world.spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index])   
            else:
                self.sensor = self._world.spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self._server_clock.tick()
        self._hud.server_fps = self._server_clock.get_fps()
        image.convert(self._sensors[self._index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud)
#        world.draw_waypoints()
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)                  
            world.reward_computing()          
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display, world.camera_index)
            pygame.display.flip()

    finally:

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot at spawn')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()

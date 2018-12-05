#===============================================================================
#-- VehicleManager -------------------------------------------------------------
#===============================================================================
class VehicleManager(object):
    def __init__(self, vehicle, autopilot_enabled=False):
        self._vehicle = vehicle
        self._autopilot_enabled = autopilot_enabled
        self._hud = None  #TODO
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
        text = "road id = %d, lane id = %d, transform = %s"
        #            print(text % (wp_next.road_id, wp_next.lane_id, wp_next.transform))
        self.inner_wp_draw(helper, wp_next)

    def inner_wp_draw(self, helper, wp, depth=4):
        if depth < 0:
            return
        for w in wp.next(4.0):
            t = w.transform
            begin = t.location + carla.Location(
                z=40)  #TODO, the wp Z-coord is set as 0, not visiable
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
    # issue#17, CPP code can be viewed at ACarlaVehicleController:IntersectPlayerWithRoadMap
    def offroad_invasion(self):
        return 0

    #TODO: for demo, all vehicles has same start_pos & end_pos
    #but in reality, need find the nearest pos at each spawn location
    def _nearest_pos(self, vid):
        pass

    def _pos_coord(self, scenario):
        POS_COOR_MAP = json.load(
            open("env/carla/POS_COOR/pos_cordi_map_town1.txt"))
        #TODO: failure due to start_id type maybe list or int
        start_id = scenario["start_pos_id"]
        end_id = scenario["end_pos_id"]
        self._start_pos = POS_COOR_MAP[str(start_id[0])]
        self._end_pos = POS_COOR_MAP[str(end_id[0])]
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

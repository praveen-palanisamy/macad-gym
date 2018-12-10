import numpy as np


class Reward(object):
    def __init__(self):
        self.reward = 0.0
        self.prev = None
        self.curr = None

    def compute_reward(self, prev_measurement, curr_measurement, flag):
        self.prev = prev_measurement
        self.curr = curr_measurement

        if flag == "corl2017":
            return self.compute_reward_corl2017()
        elif flag == "lane_keep":
            return self.compute_reward_lane_keep()
        elif flag == "custom":
            return self.compute_reward_custom()

    def compute_reward_custom(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0

        self.reward -= self.curr["intersection_offroad"] * 0.05
        self.reward -= self.curr["intersection_otherlane"] * 0.05

        if self.curr["next_command"] == "REACH_GOAL":
            self.reward += 100

        return self.reward

    def compute_reward_corl2017(self):
        self.reward = 0.0
        cur_dist = self.curr["distance_to_goal"]
        prev_dist = self.prev["distance_to_goal"]
        # Distance travelled toward the goal in m
        self.reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)
        # Change in speed (km/h)
        self.reward += 0.05 * (
            self.curr["forward_speed"] - self.prev["forward_speed"])
        # New collision damage
        self.reward -= .00002 * (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])

        # New sidewalk intersection
        self.reward -= 2 * (self.curr["intersection_offroad"] -
                            self.prev["intersection_offroad"])

        # New opposite lane intersection
        self.reward -= 2 * (self.curr["intersection_otherlane"] -
                            self.prev["intersection_otherlane"])

        return self.reward

    def compute_reward_lane_keep(self):
        self.reward = 0.0
        # Speed reward, up 30.0 (km/h)
        self.reward += np.clip(self.curr["forward_speed"], 0.0, 30.0) / 10
        # New collision damage
        new_damage = (
            self.curr["collision_vehicles"] +
            self.curr["collision_pedestrians"] + self.curr["collision_other"] -
            self.prev["collision_vehicles"] -
            self.prev["collision_pedestrians"] - self.prev["collision_other"])
        if new_damage:
            self.reward -= 100.0
        # Sidewalk intersection
        self.reward -= self.curr["intersection_offroad"]
        # Opposite lane intersection
        self.reward -= self.curr["intersection_otherlane"]

        return self.reward

    def destory(self):
        pass

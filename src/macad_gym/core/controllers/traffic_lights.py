import math


def get_tls(world,
            candidate_transform,
            sort=False,
            distance_threshold=50.0,
            angle_threshold=math.pi / 4.0):
    """Get a list of traffic lights that will affect an actor

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
            traffic_light_orientation = math.radians(
                t.get_transform().rotation.yaw)
            angle = math.fabs((
                (traffic_light_orientation - actor_orientation + math.pi) %
                (math.pi * 2.0)) - math.pi)

            if math.fabs(angle) > angle_threshold and math.fabs(
                    angle) < math.pi - angle_threshold:
                tls[t] = (dist, angle)
    if sort:
        # Return a sorted list sorted first based on dist & then angle
        return sorted(tls.items(), key=lambda kv: kv[1])
    else:
        return list(tls.keys())


def set_tl_state(traffic_lights, traffic_light_state):
    """Sets all traffic lights in the `traffic_lights` list to the given state.

    Args:
        traffic_lights (list): List of carla.TrafficLight actors
        traffic_light_state (carla.TrafficLightState):  The state to set the
            lights

    Returns:
        None

    """
    for traffic_light in traffic_lights:
        traffic_light.set_state(traffic_light_state)


# Note: may not work due to inconsistent behavior of actor.get_traffic_light()
def change_tl_of_actor(actor, traffic_light_state):
    traffic_light = actor.get_traffic_light()
    if traffic_light is not None:
        traffic_light.set_state(traffic_light_state)
        return True
    return False

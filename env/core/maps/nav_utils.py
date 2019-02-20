import carla


def get_next_waypoint(world, location, distance=1.0):
    """ Return the location of the waypoint distance meters away

    Args:
        world (carla.World): Carla world to navigate in
        location (tuple): The (x, y, z) coordinates of the origin waypoint

    Returns:
        next waypoint as a list of coordinates (x,y,z)
    """
    location = carla.Location(*location)
    current_waypoint = world.get_map().get_waypoint(location)
    current_coords = current_waypoint.transform.location
    next_waypoints = current_waypoint.next(distance)
    if len(next_waypoints) > 0:
        current_coords = next_waypoints[0].transform.location
    return [current_coords.x, current_coords.y, current_coords.z]


def get_shortest_path_distance(planner, origin, destination):
    """
    This function calculates the distance of the shortest path connecting
    origin and destination using A* search with distance heuristic.
    planner     :   the route planner
    origin      :   tuple containing x, y co-ordinates of start position
    destination :   tuple containing x, y co-ordinates of end position
    return      :   the distance of the shortest path found
    """

    graph, _ = planner.build_graph()
    path = planner.path_search(origin, destination)
    distance = 0.0
    if len(path) > 0:
        first_node = graph.nodes[path[0]]["vertex"]
        distance = planner.distance(origin, first_node)
        for i in range(1, len(path)):
            distance += planner.distance(graph.nodes[path[i - 1]]["vertex"],
                                         graph.nodes[path[i]]["vertex"])

    return distance

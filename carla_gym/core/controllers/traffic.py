import random
import logging

import carla

logger = logging.getLogger(__name__)


def apply_traffic(world, traffic_manager, num_vehicles, num_pedestrians, safe=False):
    """Spawn npc pedestrians and vehicles with random targets.

    Args:
        world: world object
        traffic_manager: traffic manager attached to the world
        num_vehicles: number of NPC vehicles to spawn
        num_pedestrians: number of NPC pedestrians to spawn
        safe: boolean flag if True avoid to use problematic blueprints for NPC

    Returns:
        Tuple composed of list of vehicles and pedestrians spawned.
    """
    # --------------
    # Spawn vehicles
    # --------------
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    if safe:
        blueprints = list(filter(lambda x: int(x.get_attribute('number_of_wheels')) == 4 and not
                (x.id.endswith('microlino') or
                 x.id.endswith('carlacola') or
                 x.id.endswith('cybertruck') or
                 x.id.endswith('t2') or
                 x.id.endswith('sprinter') or
                 x.id.endswith('firetruck') or
                 x.id.endswith('ambulance')), blueprints))

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    random.shuffle(spawn_points)
    if num_vehicles <= number_of_spawn_points:
        spawn_points = random.sample(spawn_points, num_vehicles)
    else:
        msg = "requested %d vehicles, but could only find %d spawn points"
        logger.warning(msg, num_vehicles, number_of_spawn_points)
        num_vehicles = number_of_spawn_points

    vehicles_list = []
    failed_v = 0
    for n, transform in enumerate(spawn_points):
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        blueprint.set_attribute("role_name", "autopilot")

        # spawn the cars and set their autopilot and light state all together
        vehicle = world.try_spawn_actor(blueprint, transform)
        world.tick()
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles_list.append(vehicle)
        else:
            failed_v += 1

    logger.info(f"{num_vehicles-failed_v}/{num_vehicles} vehicles correctly spawned.")

    # Set automatic vehicle lights update if specified
    # if args.car_lights_on:
    #     all_vehicle_actors = world.get_actors(vehicles_id_list)
    #     for actor in all_vehicle_actors:
    #         traffic_manager.update_vehicle_lights(actor, True)

    # -------------
    # Spawn Walkers
    # -------------
    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    pedestrian_controller_bp = world.get_blueprint_library().find("controller.ai.walker")

    # Take all the random locations to spawn
    spawn_points = []
    for i in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # Spawn the walker object
    pedestrians_list = []
    pedestrians_speed = []
    failed_p = 0
    for spawn_point in spawn_points:
        pedestrian_bp = random.choice(blueprints)
        # set as not invincible
        if pedestrian_bp.has_attribute("is_invincible"):
            pedestrian_bp.set_attribute("is_invincible", "false")
        # set the max speed
        if pedestrian_bp.has_attribute("speed"):
            if random.random() <= percentagePedestriansRunning:
                speed = pedestrian_bp.get_attribute("speed").recommended_values[2]  # running
            else:
                speed = pedestrian_bp.get_attribute("speed").recommended_values[1]  # walking
        else:
            speed = 0.0
        pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
        world.tick()
        if pedestrian is not None:
            controller = world.try_spawn_actor(pedestrian_controller_bp, carla.Transform(), pedestrian)
            world.tick()
            if controller is not None:
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(float(speed))
                pedestrian.controller = controller
                pedestrians_list.append(pedestrian)
            else:
                pedestrian.destroy()
                failed_p += 1
        else:
            failed_p += 1

    logger.info(f"{num_pedestrians-failed_p}/{num_pedestrians} pedestrians correctly spawned.")

    # Initialize each controller and set target to walk
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for pedestrian in pedestrians_list:
        pedestrian.controller.start()  # start walker

    traffic_manager.global_percentage_speed_difference(30.0)

    return vehicles_list, pedestrians_list

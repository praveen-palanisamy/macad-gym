import carla
import weakref
import math
import collections


class LaneInvasionSensor(object):
    """Lane Invasion class from carla manual_control.py
    """

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self.offlane = 0  # count of off lane
        self.offroad = 0  # count of off road
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_invasion_history(self):
        history = collections.defaultdict(int)
        for frame, text in self._history:
            history[frame] = text
        return history

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # text = ['%r' % str(x).split()[-1]
        #  for x in set(event.crossed_lane_markings)]
        # self._hud.notification('Crossed line %s' % ' and '.join(text))
        text = [
            '%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)
        ]
        self.offlane += 1
        """
        info_str = ('VEHICLE %s' % self._parent.id +
                    ' crossed line %s' % ' and '.join(text))
        logging.info(info_str)
        """

        if len(set(event.crossed_lane_markings)) == 1:
            self.offroad += 1
            """
            info_str = ('VEHICLE %s' % self._parent.id +
                        ' crossed road %s' % ' and '.join(text))
            logging.info(info_str)
            """

        self._history.append((event.frame_number, text))
        if len(self._history) > 4000:
            self._history.pop(0)

    def _reset(self):
        """Reset off-lane and off-road counts"""
        self.offlane = 0
        self.offroad = 0
        self._history = []


class CollisionSensor(object):
    """Collision sensor class from carla manual_control.py
    """

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self.collision_vehicles = 0
        self.collision_pedestrians = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # actor_type = get_actor_display_name(event.other_actor)
        # self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)
        """
        info_str = ('vehicle %s ' % self._parent.id +
                    ' collision with %2d vehicles, %2d people, %2d others' %
                    self.dynamic_collided())
        logging.info(info_str)
        """

        _cur = event.other_actor
        if _cur.id == 0:  # the static world objects
            if _cur.type_id in self.collision_type_id_set:
                return
            else:
                self.collision_type_id_set.add(_cur.type_id)
        else:
            if _cur.id in self.collision_id_set:
                return
            else:
                self.collision_id_set.add(_cur.id)

        collided_type = type(_cur).__name__
        if collided_type == 'Vehicle':
            self.collision_vehicles += 1
        elif collided_type == 'Walker':
            self.collision_pedestrians += 1
        elif collided_type == 'Actor':
            self.collision_other += 1
        else:
            pass

    def _reset(self):
        self.collision_vehicles = 0
        self.collision_pedestrians = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()

    def dynamic_collided(self):
        return (self.collision_vehicles, self.collision_pedestrians,
                self.collision_other)

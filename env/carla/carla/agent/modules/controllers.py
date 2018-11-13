import math

from pid_controller.pid import PID
from carla.client import VehicleControl
# PID based controller, we can have different ones


class Controller(object):

    # The vehicle controller, it receives waypoints and applies a PID control in order
    # to get the action.


    def __init__(self, params):

        # The parameters for this controller, set by the agent
        self.params = params
        # PID speed controller
        self.pid = PID(p=params['pid_p'], i=params['pid_i'], d=params['pid_d'])



    def get_control(self, wp_angle, wp_angle_speed, speed_factor, current_speed):
        control = VehicleControl()
        current_speed = max(current_speed, 0)

        steer = self.params['steer_gain'] * wp_angle
        if steer > 0:
            control.steer = min(steer, 1)
        else:
            control.steer = max(steer, -1)

        # Don't go0 to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = self.params['target_speed'] * speed_factor
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 20 * speed_factor
        else:
            target_speed_adjusted = 15 * speed_factor

        self.pid.target = target_speed_adjusted
        pid_gain = self.pid(feedback=current_speed)
        print ('Target: ', self.pid.target, 'Error: ', self.pid.error, 'Gain: ', pid_gain)
        print ('Target Speed: ', target_speed_adjusted, 'Current Speed: ', current_speed, 'Speed Factor: ',
               speed_factor)

        throttle = min(max(self.params['default_throttle'] - 1.3 * pid_gain, 0),
                       self.params['throttle_max'])

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self.params['brake_strength'], 1)
        else:
            brake = 0


        control.throttle = max(throttle, 0)  # Prevent N by putting at least 0.01
        control.brake = brake

        print ('Throttle: ', control.throttle, 'Brake: ', control.brake, 'Steering Angle: ', control.steer)

        return control
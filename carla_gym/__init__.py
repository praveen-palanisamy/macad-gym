"""Import all the necessary modules for the Multi Actor Carla package."""
import logging
import os
import sys

from gymnasium.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

__version__ = "0.0.1"

# Init and setup the root logger
logging.basicConfig(filename=LOG_DIR + "/carla-gym.log", level=logging.DEBUG)

# Fix path issues with included CARLA API
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "carla/PythonAPI"))

register(id="carla-v0", entry_point="carla_gym.multi_env:MultiActorCarlaEnvPZ")

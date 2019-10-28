import os
import sys
import logging

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Init and setup the root logger
logging.basicConfig(filename=LOG_DIR + '/macad-gym.log', level=logging.DEBUG)

# Fix path issues with included CARLA API
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carla/PythonAPI"))

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'HomoNcomIndePOIntrxMASS3CTWN3-v0': {
        "entry_point":
        "macad_gym.envs:HomoNcomIndePOIntrxMASS3CTWN3",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    'HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0': {
        "entry_point":
        "macad_gym.envs:HeteNcomIndePOIntrxMATLS1B2C1PTWN3",
        "description":
        "Heterogeneous, Non-communicating, Independent,"
        "Partially-Observable Intersection Multi-Agent"
        " scenario with Traffic-Light Signal, 1-Bike, 2-Car,"
        "1-Pedestrian in Town3, version 0"
    }
}

for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)

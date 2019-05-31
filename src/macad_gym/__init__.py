import os
import logging

from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = "logs"
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
# Init and setup the root logger
logging.basicConfig(filename=LOG_DIR + '/macad-gym.log', level=logging.DEBUG)

register(
    id='HomoNcomIndePOIntrxMASS3CTWN3-v0',
    entry_point="macad_gym.envs:HomoNcomIndePOIntrxMASS3CTWN3",
)

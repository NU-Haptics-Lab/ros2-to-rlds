from envlogger.backends.tfds_backend_writer import *
from envlogger.step_data import *
import numpy as np
import dm_env
import tensorflow as tf
from os.path import expanduser
import os
import ros2_to_rlds.utils as utils

from sensor_msgs.msg import Image

SHAPE = (3, 3, 3)
HOME = expanduser("~")
DIRECTORY = HOME + "/data/ros2_to_rlds/test/image_msg"

# make the directory if it doesn't exist
os.makedirs(DIRECTORY, exist_ok=True)

# msg example
"""
header:
  stamp:
    sec: 1739222899
    nanosec: 922223964
  frame_id: leftcam
height: 480
width: 640
encoding: bgr8
is_bigendian: 0
step: 1920
data:
- 7
- 8
- 9
- 7
- 8
- 9
- 4
- 4
- 5
"""
msg = Image()
msg.header.stamp.sec = 1739222899
msg.header.stamp.nanosec = 922223964
msg.header.frame_id = "leftcam"
msg.height = 480
msg.width = 640
msg.encoding = "bgr8"
msg.is_bigendian = 0
msg.step = 1920
msg.data = [7, 8, 9, 7, 8, 9, 4, 4, 5]

# make the topic -> msg dictionary
d1 = {'/leftcam/image_resized': msg}

# make the python dict
d2 = utils.ConvertRos2DictToDict(d1)

# make the tfds dict
d3 = utils.ConvertDictToTFDS(d2)

# a sequence of images, NB_FRAME long
tfds_features = d3

observation = d2

ds_config = tfds.rlds.rlds_base.DatasetConfig(
    name='test',
    observation_info=tfds_features,
    action_info=tf.float64,
    reward_info=tf.float64,
    discount_info=tf.float64  # default python type for 0.
)


writer = TFDSBackendWriter(data_directory=DIRECTORY,
                           split_name='train', # required
                           max_episodes_per_file=500,
                           ds_config=ds_config)
zero_float64 = 0.0  # np.array(0.0, dtype="float64")

# start episode
timestep = dm_env.restart(observation=observation)
data = StepData(timestep, action=zero_float64)
writer.record_step(data, True)

# transition episode
timestep = dm_env.transition(reward=zero_float64, observation=observation)
data = StepData(timestep, action=zero_float64)
writer.record_step(data, False)

# end episode
timestep = dm_env.termination(reward=zero_float64, observation=observation)
data = StepData(timestep, action=zero_float64)
writer.record_step(data, False)

# close
writer.close()

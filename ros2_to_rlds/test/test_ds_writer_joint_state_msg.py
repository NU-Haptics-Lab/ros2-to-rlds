from envlogger.backends.tfds_backend_writer import *
from envlogger.step_data import *
import numpy as np
import dm_env
import tensorflow as tf
from os.path import expanduser
import os
import ros2_to_rlds.utils as utils

from sensor_msgs.msg import JointState

HOME = expanduser("~")
DIRECTORY = HOME + "/data/ros2_to_rlds/test/joint_state_msg"
ZERO = 0.0  # np.array(0.0, dtype="float64")
DEFAULT = tf.float64

# make the directory if it doesn't exist
os.makedirs(DIRECTORY, exist_ok=True)

# msg example
"""
header:
  stamp:
    sec: 1739221968
    nanosec: 451654260
  frame_id: ''
name:
- gofa1_joint_1
- gofa1_joint_2
- gofa1_joint_3
- gofa1_joint_4
- gofa1_joint_5
- gofa1_joint_6
position:
- 1.6979367490662591
- 0.6480197960326936
- 0.07246066588184955
- 0.9965027166935533
- -0.6
- 0.11814198098224574
velocity: []
effort: []
"""
msg = JointState()
msg.header.stamp.sec = 1739221968
msg.header.stamp.nanosec = 451654260
msg.header.frame_id = ""
msg.name = [] # ['gofa1_joint_1', 'gofa1_joint_2', 'gofa1_joint_3', 'gofa1_joint_4', 'gofa1_joint_5', 'gofa1_joint_6']
msg.position = [1.6979367490662591, 0.6480197960326936, 0.07246066588184955, 0.9965027166935533, -0.6, 0.11814198098224574]
msg.velocity = []
msg.effort = []


# make the topic -> msg dictionary
d1 = {'/example/joint_states': msg}

# make the python dict
d2 = utils.ConvertRos2DictToDict(d1)

# make the tfds dict
d3 = utils.ConvertDictToTFDS(d2)

# a sequence of images, NB_FRAME long
tfds_features = d3

# test for both action and observation
observation = d2
action = ZERO

ds_config = tfds.rlds.rlds_base.DatasetConfig(
    name='test',
    observation_info=tfds_features,
    action_info=DEFAULT,
    reward_info=DEFAULT,
    discount_info=DEFAULT  # default python type for 0.
)


writer = TFDSBackendWriter(data_directory=DIRECTORY,
                           split_name='train', # required
                           max_episodes_per_file=500,
                           ds_config=ds_config)

# start episode
timestep = dm_env.restart(observation=observation)
data = StepData(timestep, action=action)
writer.record_step(data, True)

# transition episode
timestep = dm_env.transition(reward=ZERO, observation=observation)
data = StepData(timestep, action=action)
writer.record_step(data, False)

# end episode
timestep = dm_env.termination(reward=ZERO, observation=observation)
data = StepData(timestep, action=action)
writer.record_step(data, False)

# close
writer.close()

print("\n\nClosed Successfully\n")
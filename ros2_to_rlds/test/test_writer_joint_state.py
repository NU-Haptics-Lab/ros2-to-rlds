from envlogger.backends.tfds_backend_writer import *
from envlogger.step_data import *
import numpy as np
import dm_env
import tensorflow as tf
from os.path import expanduser
from sensor_msgs.msg import JointState
from ros2_to_rlds.utils import ConvertRos2DictToDict, ConvertDictToTFDS

LENGTH = 3
HOME = expanduser("~")
DIRECTORY = HOME + "/data/test2"

"""
Composite FeatureConnector for a dict where each value is a list.
"""
msg = JointState()
msg.name = ["test", "ing"]

observation = ConvertRos2DictToDict({"test": msg})
tfds_features = ConvertDictToTFDS(observation)

print(observation)
print(tfds_features)

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
data = StepData(timestep, zero_float64)
writer.record_step(data, True)

# transition episode
timestep = dm_env.transition(reward=zero_float64, observation=observation)
data = StepData(timestep, zero_float64)
writer.record_step(data, False)

# end episode
timestep = dm_env.termination(reward=zero_float64, observation=observation)
data = StepData(timestep, zero_float64)
writer.record_step(data, False)

# close
writer.close()

from dexterity_master_py.dexterity_utils import *
from std_msgs.msg import *

from envlogger.backends.tfds_backend_writer import *
from envlogger.step_data import *

import dm_env
import tensorflow as tf

"""
restart = _environment.restart
termination = _environment.termination
transition = _environment.transition
truncation = _environment.truncation
"""

msg = Float64()
print(convert(msg))

data_directory = "/home/omnid/data/test_ds_writer"

msg_np = convert(msg)
print("msg np: ")
print(msg_np)

topic_and_msg = {'test_topic': msg_np}

print(ConvertNumpyToTFDSFeatures(topic_and_msg))

ds_config = tfds.rlds.rlds_base.DatasetConfig(
    name='test',
    observation_info=ConvertNumpyToTFDSFeatures(topic_and_msg),
    action_info=tf.float64,
    reward_info=tf.float64,
    discount_info=tf.float64  # default python type for 0.
)


writer = TFDSBackendWriter(data_directory=data_directory,
                           split_name='train', # required
                           max_episodes_per_file=500,
                           ds_config=ds_config)
zero_float64 = 0.0  # np.array(0.0, dtype="float64")

# start episode
timestep = dm_env.restart(observation=topic_and_msg)
data = StepData(timestep, zero_float64)
writer.record_step(data, True)

# transition episode
timestep = dm_env.transition(reward=zero_float64, observation=topic_and_msg)
data = StepData(timestep, zero_float64)
writer.record_step(data, False)

# end episode
timestep = dm_env.termination(reward=zero_float64, observation=topic_and_msg)
data = StepData(timestep, zero_float64)
writer.record_step(data, False)

# close
writer.close()

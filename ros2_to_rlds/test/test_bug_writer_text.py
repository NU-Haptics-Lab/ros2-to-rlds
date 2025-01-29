from envlogger.backends.tfds_backend_writer import *
from envlogger.step_data import *
import dm_env
import tensorflow as tf
from os.path import expanduser

HOME = expanduser("~")
DIRECTORY = HOME + "/data/test2"

"""
TEST:
If action includes a tfds.features.Text, it'll crash.
"""


action = "test_string"


ds_config = tfds.rlds.rlds_base.DatasetConfig(
    name='test',
    observation_info=tf.float64,
    action_info=tfds.features.Text(),
    reward_info=tf.float64,
    discount_info=tf.float64  # default python type for 0.
)


writer = TFDSBackendWriter(data_directory=DIRECTORY,
                           split_name='train', # required
                           max_episodes_per_file=500,
                           ds_config=ds_config)
zero_float64 = 0.0  # np.array(0.0, dtype="float64")

# start episode
timestep = dm_env.restart(observation=zero_float64)
data = StepData(timestep, action)
writer.record_step(data, True)

# transition episode
timestep = dm_env.transition(reward=zero_float64, observation=zero_float64)
data = StepData(timestep, action)
writer.record_step(data, False)

# end episode
timestep = dm_env.termination(reward=zero_float64, observation=zero_float64)
data = StepData(timestep, action)
writer.record_step(data, False)

print("Data wrote. Attempting to close.")

# close
writer.close()

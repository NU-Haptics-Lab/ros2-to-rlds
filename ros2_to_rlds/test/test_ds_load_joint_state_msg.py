from os.path import expanduser
import unittest

import tensorflow_datasets as tfds
import rlds

from sensor_msgs.msg import Image

"""
Feb 10, 2025 - Fails when loading. I think it has something to do with having any empty arrays in the original msg (check out test_ds_writer_joint_state_msg.py)
"""

""" Parameters """
HOME = expanduser("~")
DIRECTORY = HOME + "/data/ros2_to_rlds/test/joint_state_msg"
        
def print_tensor(dataset, topic_name):
    for ep in dataset.take(1):
        steps = ep[rlds.STEPS]
        
        for step in steps.take(2):
            obs = step[rlds.OBSERVATION]
            
            msg = obs[topic_name]
            
            print(msg)
            
            print(msg['position'])
            print(msg['encoding'])
        
    return True
        
class test_load_joint_state_msg(unittest.TestCase):
    def test_load(self):
        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        
        # test
        self.assertTrue(print_tensor(dataset, '/example/joint_states'))
    
        


if __name__ == '__main__':
    unittest.main()

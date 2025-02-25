from os.path import expanduser
import unittest

import tensorflow_datasets as tfds
import rlds

from sensor_msgs.msg import Image


""" Parameters """
SHAPE = (3, 3, 3)
HOME = expanduser("~")
DIRECTORY = HOME + "/data/ros2_to_rlds/test/image_msg"

# DIRECTORY = HOME + "/data/test_std_msgs"

# # load the dataset
# builder = tfds.builder_from_directory(DIRECTORY)
# dataset = builder.as_dataset(split='all')

# print("Nb episode: ", len(dataset))
# print(dataset)

# nb_steps = 0

# for ep in dataset:
#     print(ep)
    
#     steps = ep[rlds.STEPS]
    
#     for step in steps:
#         print("\nStep: ", step)
#         action = step[rlds.ACTION]
#         print("Action: ", action)
#         val = action['/test/data_collection/int32']['data']
#         print(val)
#         val = action['/test/data_collection/float64']['data']
#         print(val)
#         val = action['/test/data_collection/float64_multi_array']['data']
#         print(val)
#         val = action['/test/data_collection/int32_multi_array']['data']
#         print(val)
        
def print_tensor(dataset, topic_name):
    for ep in dataset.take(1):
        steps = ep[rlds.STEPS]
        
        for step in steps.take(2):
            obs = step[rlds.OBSERVATION]
            
            msg = obs[topic_name]
            
            print(msg)
            
            print(msg['data'])
            print(msg['encoding'])
        
    return True
        
        
        
class test_load_image_msg(unittest.TestCase):
    def test_load(self):
        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        
        # test
        self.assertTrue(print_tensor(dataset, '/leftcam/image_resized'))
    
        


if __name__ == '__main__':
    unittest.main()

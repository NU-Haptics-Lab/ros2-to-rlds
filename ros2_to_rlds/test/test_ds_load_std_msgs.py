from os.path import expanduser
import unittest

import tensorflow_datasets as tfds
import rlds

""" Parameters """
HOME = expanduser("~")
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
        
        for step in steps.take(1):
            action = step[rlds.ACTION]
            val = action[topic_name]['data']
        print("\n\nPrint Tensor value: \n", val)
        
    return True
        
        
        
class test_std_msgs(unittest.TestCase):
    def test_int32(self):
        DIRECTORY = HOME + "/data/test_int32"

        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        
        # test
        self.assertTrue(print_tensor(dataset, '/test/data_collection/int32'))
        
    def test_int32_array(self):
        DIRECTORY = HOME + "/data/test_int32array"

        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        self.assertTrue(print_tensor(dataset, '/test/data_collection/int32array'))
        
    def test_float64(self):
        DIRECTORY = HOME + "/data/test_float64"

        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        self.assertTrue(print_tensor(dataset, '/test/data_collection/float64'))
        
    def test_float64_array(self):
        DIRECTORY = HOME + "/data/test_float64array"

        # load the dataset
        builder = tfds.builder_from_directory(DIRECTORY)
        dataset = builder.as_dataset(split='all')
        self.assertTrue(print_tensor(dataset, '/test/data_collection/float64array'))
        
    # def test_bool(self):
    #     DIRECTORY = HOME + "/data/test_bool"

    #     # load the dataset
    #     builder = tfds.builder_from_directory(DIRECTORY)
    #     dataset = builder.as_dataset(split='all')
    #     self.assertTrue(print_tensor(dataset, '/test/data_collection/bool'))
        


if __name__ == '__main__':
    unittest.main()

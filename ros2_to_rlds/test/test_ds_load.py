from os.path import expanduser

import tensorflow_datasets as tfds
import rlds

""" Parameters """
HOME = expanduser("~")
DIRECTORY = HOME + "/data/test2"

# load the dataset
builder = tfds.builder_from_directory(DIRECTORY)
dataset = builder.as_dataset(split='all')

print("Nb episode: ", len(dataset))

nb_steps = 0
for ep in dataset:
    steps = ep[rlds.STEPS]
    
    nb_steps = nb_steps + rlds.transformations.episode_length(steps).numpy()
    
    for step in steps.take(nb_steps):
        print(step)
    
    
print("Total number of steps: {}".format(nb_steps))
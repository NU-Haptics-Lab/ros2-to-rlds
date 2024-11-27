from os.path import expanduser

import tensorflow_datasets as tfds
import rlds

""" Parameters """
HOME = expanduser("~")
DIRECTORY = HOME + "/data/test"

# load the dataset
builder = tfds.builder_from_directory(DIRECTORY)
dataset = builder.as_dataset(split='all')

print("Nb episode: ", len(dataset))

# flatten dataset
dataset = dataset.flat_map(lambda episode: episode[rlds.STEPS])

nb_steps = rlds.transformations.episode_length(dataset).numpy()
print("Nb steps: ", nb_steps)
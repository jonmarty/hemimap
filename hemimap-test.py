import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt

# Input the filenames for input and query data
parser = ArgumentParser()
parser.add_argument('--input', help="a binary numpy array indicating the position of the neuron in the input space")
parser.add_argument('--query', help="a binary numpy array indicating the position of the set of neurons that the input is meant to represent")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
parser.add_argument('--step', help="Checkpoint step to load from")
parser.add_argument('--outdir', help="Directory to deposit output image in")
args = parser.parse_args()

# Load in the input
input_bits = np.load(args.input)
input_data = np.stack(np.nonzero(input_bits)).transpose()

print(input_data.shape)

# Load in the query
query_bits = np.load(args.query)
query_data = np.stack(np.nonzero(query_bits)).transpose()

print(query_data.shape)

# Load in the tensorflow model and compute output points
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(f"{args.checkdir}/{args.name}-{args.step}.meta")
    saver.restore(sess, f"{args.checkdir}/{args.name}-{args.step}")
    graph = tf.get_default_graph()
    prediction = graph.get_tensor_by_name("predictions:0")
    x = graph.get_tensor_by_name("x:0")
    output_data = sess.run(prediction, {x: input_data})

# Create visual of output mappings vs the query
output_image = np.zeros(list(query_bits.shape) + [3])
output_image[:,:,2] = query_bits
output_indices = np.round(output_data).astype(np.int)
for index in output_indices:
    output_image[index[0], index[1], 0] = 1

# Save this visual to file
plt.imsave(f"{args.outdir}/{args.name}-{args.step}.png", output_image)
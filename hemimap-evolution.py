import numpy as np
import tensorflow as tf
from tensorflow_graphics.nn.loss import chamfer_distance
from argparse import ArgumentParser
import os
import tensorflow_probability as tfp

# Input the filenames for input and query data
parser = ArgumentParser()
parser.add_argument('--input', help="a binary numpy array indicating the position of the neuron in the input space")
parser.add_argument('--query', help="a binary numpy array indicating the position of the set of neurons that the input is meant to represent")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
parser.add_argument('--is2d', action="store_true", help="If used, input is expected to be 2d")
args = parser.parse_args()

# Set the number of inputs and outputs to our model
n_input = 2 if args.is2d else 3
n_output = 2

# Load in the input
input_bits = np.load(args.input)
input_data = np.stack(np.nonzero(input_bits)).transpose()
input_tensor = tf.convert_to_tensor(input_data)

# Load in the query
query_bits = np.load(args.query)
query_data = np.stack(np.nonzero(query_bits)).transpose()
query_tensor = tf.convert_to_tensor(query_data)

# Some assertions to check correctness of the data
assert input_data.shape[1] == n_input
assert input_data.shape[1] == n_output

# Check that eager execution is currently being used
assert tf.executing_eagerly()

#TODO: Try this https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad/

def predictions(x, weights, biases):
    return tf.add(tf.matmul(x, weights), biases)

def loss(pred, y):
    return chamfer_distance.evaluate(pred, y)

population_size = 40

def make_weights(dims):
    W = np.random.uniform(size = dims)
    return tf.convert_to_tensor(W / np.linalg.det(W))

def make_biases(dims):
    b = np.random.uniform(size=dims)
    return tf.convert_to_tensor(b / np.linalg.norm(b))

initial_population = [tf.tuple([make_weights([n_input, n_output]), make_biases([n_output])]) for _ in range(population_size)]

def objective_function(*items):
    return [loss(predictions(input_tensor, item[0], item[1]), query_tensor) for item in items]

#optim_results = tfp.optimizer.differential_evolution_minimize(
#    objective_function,
#    initial_population = initial_population,
#    population_size = population_size,
#    max_iterations = 100
#)

tfp.optimizer.differential_evolution_one_step(objective_function, initial_population)

weights, biases = optim_results.position
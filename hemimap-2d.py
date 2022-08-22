import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow_graphics.nn.loss import chamfer_distance
from argparse import ArgumentParser
import os

# Input the filenames for input and query data
parser = ArgumentParser()
parser.add_argument('--input', help="a binary numpy array indicating the position of the neuron in the input space")
parser.add_argument('--query', help="a binary numpy array indicating the position of the set of neurons that the input is meant to represent")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
args = parser.parse_args()

# Load in the input
input_bits = np.load(args.input)
input_data = np.stack(np.nonzero(input_bits)).transpose()

# Load in the query
query_bits = np.load(args.query)
query_data = np.stack(np.nonzero(query_bits)).transpose()

#TODO: Give names to these tensors, loading them is really confusing
def perceptron(x, weights, biases, name):
    return tf.add(tf.matmul(x, weights), biases, name=name)

n_input = 2
n_output = 2

weights_init = tf.convert_to_tensor(np.eye(2) + (np.random.rand(n_input, n_output) - 0.5), dtype = tf.float32)
biases_init = tf.convert_to_tensor(np.random.rand(n_output) - 0.5, dtype = tf.float32)

weights = tf.Variable(weights_init)
biases = tf.Variable(biases_init)

training_epochs = 5000
display_step = 10
save_step = 100
batch_size = 320
learning_rate = 0.0001

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

predictions = perceptron(x, weights, biases, name="predictions")

cost = chamfer_distance.evaluate(predictions, y)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    for epoch in range(training_epochs):
        x_train = input_data.copy()
        np.random.shuffle(x_train)
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        for i in range(total_batch):
            batch_x = x_batches[i]
            _, c = sess.run([optimizer, cost],
                           feed_dict = {
                               x: batch_x,
                               y: query_data
                           })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % epoch, "cost=", "{:.9f}".format(avg_cost))
        if epoch % save_step == 0:
            saver.save(sess, os.path.join(args.checkdir, "{name}").format(name = args.name), global_step = epoch)
    
    print("Optimization Finished")
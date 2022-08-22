import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow_graphics.nn.loss import chamfer_distance
from argparse import ArgumentParser
import os

#from tf_ot import dmat, sink

parser = ArgumentParser()
parser.add_argument('--input', help="a binary numpy array indicating the position of the neuron in the input space")
parser.add_argument('--target', help="a binary numpy array indicating the position of the set of neurons that the input is meant to represent (the target)")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
parser.add_argument('--is2d', action="store_true", help="If used, input is expected to be 2d")
parser.add_argument('--useGPU', action="store_true", help="Whether to use the GPU")
parser.add_argument('--epochs', default = 5000, type = int, help="Number of epochs to run for")
parser.add_argument('--ispoints', action="store_true", help="If used, will treat the data as points")
args = parser.parse_args()

# Don't use GPU devices unless specified
if not args.useGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set the number of inputs and outputs to our model
n_input = 2 if args.is2d else 3
n_output = 2 if args.is2d else 3

# Load the input
if args.ispoints:
    input_data = np.load(args.input)
else:
    input_bits = np.load(args.input)
    input_data = np.stack(np.nonzero(input_bits)).transpose()

# Load in the target
if args.ispoints:
    target_data = np.load(args.target)
else:
    target_bits = np.load(args.target)
    target_data = np.stack(np.nonzero(target_bits)).transpose()

# Some assertions to check that the data has the correct shape
assert input_data.shape[1] == n_input
assert input_data.shape[1] == n_output

def perceptron(x, weights, biases, name="perceptron"):
    return tf.add(tf.matmul(x, weights), biases, name = name)

weights_init = tf.random_normal([n_input, n_output])
biases_init = tf.random_normal([n_output])

weights = tf.Variable(weights_init, name = "weights")
biases = tf.Variable(biases_init, name = "biases")

training_epochs = args.epochs
display_step = 10
save_step = 100
batch_size = 320
learning_rate = 0.0001

x = tf.placeholder("float", [None, n_input], name = "x")
y = tf.placeholder("float", [None, n_output], name = "y")

predictions = perceptron(x, weights, biases, name = "predictions")

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
                               y: target_data
                           })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % epoch, "cost=", "{:.9f}".format(avg_cost))
        if epoch % save_step == 0:
            saver.save(sess, os.path.join(args.checkdir, "{name}").format(name = args.name), global_step = epoch)
    
    print("Optimization Finished")

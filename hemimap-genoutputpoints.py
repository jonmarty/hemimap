import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', help="File containing input pointcloud")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
parser.add_argument('--step', help="Checkpoint step to load from")
parser.add_argument('--output', help="File to put output pointcloud in")
parser.add_argument('--useGPU', action="store_true", help="Whether to employ the GPU")
args = parser.parse_args()

if not args.useGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_points = np.load(args.input)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(f"{args.checkdir}/{args.name}-{args.step}.meta")
    saver.restore(sess, f"{args.checkdir}/{args.name}-{args.step}")
    graph = tf.get_default_graph()
    prediction = graph.get_tensor_by_name("predictions:0")
    x = graph.get_tensor_by_name("x:0")
    output_points = sess.run(prediction, {x: input_points})
      
np.save(args.output, output_points)
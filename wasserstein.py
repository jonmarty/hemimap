import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Wasserstein(object):
    #TODO: Code the Wasserstein distance so we can use it as an alternative to the Chamfer distance
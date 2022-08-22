import numpy as np
from scipy.spatial.distance import cdist
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", help="Input file (.npy)")
args = parser.parse_args()

points = np.load(args.input)
dist = cdist(points, points)
mean_dist = np.mean(dist, axis=0)
mean_point = points[np.argmin(mean_dist)]

print(mean_point)
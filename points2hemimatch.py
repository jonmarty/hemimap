import numpy as np
from argparse import ArgumentParser
from PIL import Image

parser = ArgumentParser()
parser.add_argument("--points", help=".npy file containing points")
parser.add_argument("--output", help="Output file to put png image in")
args = parser.parse_args()

points = np.load(args.points).astype(np.int)

hemimatch_dims = (566, 1210, 3)

output = np.zeros(hemimatch_dims).astype(np.uint8)

for p in points:
    output[p[0], p[1], :] = 255

output = output[:,200:800,:]

for i in range(1, output.shape[0]-1):
    for j in range(1, output.shape[1]-1):
        if output[i-1,j-1,0]/255 + output[i-1,j+1,0]/255 + output[i+1,j-1,0]/255 + output[i+1,j+1,0]/255 > 2:
            output[i,j,:] = 255

output = Image.fromarray(output)

output.save(args.output)
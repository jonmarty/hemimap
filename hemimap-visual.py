import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def points2layer(points, dims):
    layer = np.zeros(dims)
    for p in points:
        if p[0] < dims[0] and p[1] < dims[1]:
            layer[p[0], p[1]] = 1
    return layer
    
def input_output_target_visual(input_points, output_points, target_points, dims):
    input_layer = points2layer(input_points, dims) if input_points is not None else np.zeros(dims)
    output_layer = points2layer(output_points, dims) if output_points is not None else np.zeros(dims)
    target_layer = points2layer(target_points, dims) if target_points is not None else np.zeros(dims)
    return np.stack((output_layer, input_layer, target_layer), axis=-1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", help="File containing the input points")
    parser.add_argument("--output", help="File containing the output points")
    parser.add_argument("--target", help="File containing the ")
    parser.add_argument("--dimX", help="Pixel length of X axis of output image", type=int)
    parser.add_argument("--dimY", help="Pixel length of Y axis of output image", type=int)
    parser.add_argument("--outfile", help="File (w/o extension) where the output image will be stored")
    args = parser.parse_args()

    input_points = np.load(args.input).astype(np.int) if args.input else None
    output_points = np.load(args.output).astype(np.int) if args.output else None
    target_points = np.load(args.target).astype(np.int) if args.target else None
    dims = (args.dimX, args.dimY)
    
    visual = input_output_target_visual(input_points, output_points, target_points, dims)
    
    # Save this visual to file
    plt.imsave(f"{args.outfile}.png", visual)
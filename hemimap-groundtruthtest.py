from argparse import ArgumentParser
import numpy as np
import os

log = lambda x: print(f"(HemiMapGroundTruthTest) {x}")

parser = ArgumentParser()
parser.add_argument("--groundtruth", help="File where ground-truth points are stored")
parser.add_argument("--epochs", help="Number of epochs to train for", type=int, default=1000)
parser.add_argument("--notrain", help="Whether to train a model, or just generate visual", action="store_true")
parser.add_argument("--scale", default=1, help="Scale of points", type=float)
parser.add_argument("--translationX", default=0, help="X value by which to translate the data", type = int)
parser.add_argument("--translationY", default=0, help="Y value by which to translate the data", type = int)
parser.add_argument("--rotationAngle", default=0, help="Angle by which to rotate data", type=int)
parser.add_argument("--rotationOriginX", default=None, help="X coordinate of the origin around which coordinates will be rotated", type=int)
parser.add_argument("--rotationOriginY", default=None, help="Y coordinate of the origin around which coordinate will be rotated", type=int)
parser.add_argument("--doReflection", action="store_true", help="Whether to perform a reflection")
parser.add_argument("--reflectionAngle", default=0, help="Angle that the reflection axis lies on", type=int)
parser.add_argument("--reflectionOriginX", default=None, help="X value of the origin of the reflection axis", type=int)
parser.add_argument("--reflectionOriginY", default=None)
args = parser.parse_args()

argnames = [arg for arg in dir(args) if "_" not in arg and arg not in ["groundtruth", "epochs", "notrain"]]
argvals = [eval(f"args.{arg}") for arg in argnames]
arguments = {arg:val for arg, val in zip(argnames, argvals) if type(val) == int or type(val) == float}

affine_arguments = [f"--{arg}={val}" for (arg, val) in arguments.items()]
affine_label = ".".join([f"{arg}_{val}" for (arg, val) in arguments.items()])
affine_arguments.append(f"--input={args.groundtruth}")
affine_file = args.groundtruth.replace("npy","") + f"{affine_label}.npy"
affine_arguments.append(f"--output={affine_file}")
space=" "
log(f"Started affine transform with parameters {space.join(affine_arguments)}")

dir_path = os.path.dirname(os.path.realpath(__file__))

os.system(f"python {dir_path}/affine-transform.py " + " ".join(affine_arguments))

log("Completed affine transform")

if not args.notrain:
    log(f"Running hemimap w/ name groundtruth.{affine_label}")
    os.system(f"python {dir_path}/hemimap.py --input={affine_file} --target={args.groundtruth} --checkdir=/home/jonathan/circadian/data/checkdir --name=groundtruth.{affine_label} --is2d --epochs={args.epochs + 10}")
    log("Finished running hemimap")
else:
    log("No train option applied, skipping hemimap")

log("Generating output points")

outfile = f"{dir_path}/../data/groundtruth.{affine_label}.step_{args.epochs}.output.npy"

os.system(f"python {dir_path}/hemimap-genoutputpoints.py --input {affine_file} --checkdir=/home/jonathan/circadian/data/checkdir --name=groundtruth.{affine_label} --step={args.epochs} --output={outfile}")

log("Generated output points")

log("Calculating bounds of groundtruth pointcloud")

groundtruth_points = np.load(args.groundtruth)
groundtruth_maxX = np.max(groundtruth_points[:,0])
groundtruth_maxY = np.max(groundtruth_points[:,1])
del(groundtruth_points)

log("Calculated bounds of groundtruth pointcloud")

log("Calculating bounds of input pointcloud (groundtruth with an affine linear transform applied)")

input_points = np.load(affine_file)
input_maxX = np.max(input_points[:,0])
input_maxY = np.max(input_points[:,1])
del(input_points)

log("Calculated bounds of input pointcloud")

log("Calculating bounds of output pointcloud (output from hemimap)")

output_points = np.load(outfile)
output_maxX = np.max(output_points[:,0])
output_maxY = np.max(output_points[:,1])
del(output_points)

log("Calculated bounds of output pointcloud")

log(f"Generating visual for groundtruth.{affine_label}")

visual_file = f"{dir_path}/../Images/groundtruth.{affine_label}.step_{args.epochs}.output"

os.system(f"python {dir_path}/hemimap-visual.py --input={affine_file} --output={outfile} --target={args.groundtruth} --dimX={int(1.5*max(groundtruth_maxX, input_maxX, output_maxX))} --dimY={int(1.5*max(groundtruth_maxY, input_maxY, output_maxY))} --outfile={visual_file}")

log("Generated visual")

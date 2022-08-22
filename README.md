# Hemimap

During my research in the Bionet Lab, I was given a dataset of neurons and their 2d locations in the fruit fly brain (coordinates projected onto a cross-section of the brain), as well as 3d microscopy images of neurons from a different study. My goal with HemiMap was to be able to map between the 3d microscopy images of neurons and their 2d locations in the fruit fly brain in order to verify that both sources were talking about the same neurons. I took the binary representations of the 2d locations of a subset of neurons that corresponded with the neurons that were supposed to be in the microscopy images, OR-ed across each pixel, and turned the resulting binary matrix into a 2d point cloud (1 point for each 1 in the binary representation). I also preprocessed the 3d microscopy images, turning them into binary representations indicating where the neuron was located, and proceeded to turn these binary representations into 3d point clouds. From there, I formulated the problem of finding a good mapping as an optimal transport problem, trying to learn an affine linear transform between 3d point clouds from my microscopy data and the 2d point cloud from my dataset.

The model itself was rather simple, just a perceptron taking in the coordinates of 3d points and outputting coordinates for 2d points. The more involved part of the model was the choice of distance measure between point clouds. Some metrics I tried were: Chamfer distance, Wasserstein distance, Sinkhorn loss, and Samples loss. I ultimately settled on the Chamfer distance. Below is a summary of each file (files containing either Tensorflow or PyTorch code are in red)
- hemimap.py - Trains the standard HemiMap model using Chamfer distance as the loss
- hemimap-test.py - Loads in trained HemiMap model and runs it against an inputted 3d neurons pointcloud, then generates a visual where the mapped (2d) output is overlaid on top of the neuron the input was supposed to be mapped onto
- hemimap-torch.py - An implementation of the HemiMap model in PyTorch, contains code to use both Samples Loss and Sinkhorn Loss.
- hemimap-evolution.py - An attempt to use Tensorflow's differential evolution functionality to optimize the model
- hemimap-groundtruthtest.py - Performs an affine linear transform on the 3d input points. Tests if the model can learn the inverse affine linear mapping between the 3d input points and the transformed input points 
- hemimap-genoutputpoints.py - Given a set of 3d input points and a Hemimap model, this script runs the HemiMap model on the input points and saves the 2d output points to file
- affine-transform.py - Performs an affine linear transform on a set of points. Is called by hemimap-groundtruthtest.py
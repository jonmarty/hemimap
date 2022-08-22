import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
import os

#import sinkhorn_pointcloud as spc
from geomloss import SamplesLoss

# Input the filenames for input and query data
parser = ArgumentParser()
parser.add_argument('--input', help="a binary numpy array indicating the position of the neuron in the input space")
parser.add_argument('--query', help="a binary numpy array indicating the position of the set of neurons that the input is meant to represent")
parser.add_argument('--checkdir', help="Directory to save checkpoints to")
parser.add_argument('--name', help="The name for this run")
parser.add_argument('--is2d', action="store_true", help="If used, input is expected to be 2d")
parser.add_argument('--epochs', default = 5000, type = int, help="Number of epochs to run for")
parser.add_argument('--display-step', default = 10, type = int, help = "The number of epochs between printouts")
parser.add_argument('--save-step', default = 100, type = int, help = "The number of epochs between saves")
parser.add_argument('--batch-size', default = 320, type = int, help = "Points per batch")
parser.add_argument('--learning-rate', default = 0.0001, type = float, help = "Learning rate of optimizer")
parser.add_argument('--momentum', default = 0.9, type = float, help = "Momentum of the optimizer")
args = parser.parse_args()

n_input = 2 if args.is2d else 3
n_output = 2

#TODO: Replace these with the argument above
display_step = 10
save_step = 100
batch_size = 320
learning_rate = 0.0001
momentum = 0.9
epsilon = 0.01
sinkhorn_iter = 100


# Load in the input
input_bits = np.load(args.input)
input_data = np.stack(np.nonzero(input_bits)).transpose()
input_tensor = torch.FloatTensor(input_data)

# Load in the query
query_bits = np.load(args.query)
query_data = np.stack(np.nonzero(query_bits)).transpose()
query_tensor = torch.FloatTensor(query_data)

assert query_tensor.size(0) >= input_tensor.size(0)

class Perceptron(nn.Module):
    def __init__(self, n_input, n_output):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(n_input, n_output)
    def forward(self, x):
        return self.layer(x)

net = Perceptron(n_input, n_output)
criterion = SamplesLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)

for epoch in range(args.epochs):
    x_perm = torch.randperm(input_tensor.size(0))
    y_perm = torch.randperm(query_tensor.size(0))[:input_tensor.size(0)]
    x_train = input_tensor[x_perm]
    y_train = query_tensor[y_perm]
    #x_batches = torch.split(x_train, batch_size)
    #y_batches = torch.split(y_train, batch_size)
    #avg_loss = 0.0
    
    #for x_batch, y_batch in zip(x_batches, y_batches):
    #    output_batch = net(x_batch)
    #    loss = criterion(output_batch, y_batch)
    #    #loss = spc.sinkhorn_loss(output_batch, y_batch, epsilon, batch_size, sinkhorn_iter)
    #    loss.backward()
    #    optimizer.step()
    #    avg_loss += loss.item() / x_batch.size(0)
    
    yhat_train = net(x_train)
    loss = criterion(yhat_train, y_train)
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    
    if epoch % display_step == 0:
        print("Epoch:", "%04d" % epoch, "cost=", "{:.9f}".format(running_loss))
    if epoch % save_step == 0:
        PATH = f"{args.checkdir}/{args.name}-{epoch}.pth"
        torch.save(net.state_dict(), PATH)

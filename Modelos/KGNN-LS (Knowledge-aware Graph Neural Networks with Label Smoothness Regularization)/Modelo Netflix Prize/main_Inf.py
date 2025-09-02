import argparse
import numpy as np
from time import time
from data_loader import load_data
from train_Inf import train

try:
    import psutil
except ImportError:
    print("Warning: psutil module not found. System metrics tracking might be limited.")
    print("Please install using: pip install psutil")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib module not found. Graphs will not be generated.")
    print("Please install using: pip install matplotlib")

np.random.seed(24)

parser = argparse.ArgumentParser()


# netflix
parser.add_argument('--dataset', type=str, default='netflix', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--track_emissions', type=bool, default=True, help='track CO2 emissions')


show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter



class Net(nn.Module):
    def __init__(self,num_state,num_action):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        self.fc4 = nn.Linear(100, num_action)

    def forward(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = F.sigmoid(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        action_prob = self.fc4(x3)
        return action_prob
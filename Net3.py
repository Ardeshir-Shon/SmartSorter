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
        self.fc1 = nn.Linear(num_state, max(int(2*num_state),int((1.5*num_state)+num_action)))
        self.fc2 = nn.Linear(max(int(2*num_state),int((1.5*num_state)+num_action)), min(int(2*num_state),int((1.1*num_state)+num_action)))
        self.fc3 = nn.Linear(min(int(2*num_state),int((1.1*num_state)+num_action)), int(1.5*num_action))
        self.fc4 = nn.Linear(int(1.5*num_action), num_action)

    def forward(self, x):
        x1 = F.tanh(self.fc1(x))
        x2 = F.sigmoid(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        action_prob = self.fc4(x3)
        return action_prob
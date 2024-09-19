import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAdaptor(nn.Module):
    def __init__(self, in_dim):
        super(FeatureAdaptor, self).__init__()

        self.fc1 = nn.Linear(in_dim, in_dim // 2)
        self.fc2 = nn.Linear(in_dim // 2, in_dim // 4)
        self.fc3 = nn.Linear(in_dim // 4, in_dim // 2)
        self.fc4 = nn.Linear(in_dim // 2, in_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
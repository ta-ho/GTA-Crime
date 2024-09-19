import torch
import torch.nn as nn
import torch.nn.functional as F 


class FeatureAdaptor(nn.Module):
    def __init__(self, embed_dim):
        super(FeatureAdaptor, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear(embed_dim//2, embed_dim//4)
        
        self.fc3 = nn.Linear(embed_dim//4, embed_dim//2)
        self.fc4 = nn.Linear(embed_dim//2, embed_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        
        return out


class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear(embed_dim//2, embed_dim//4)
        self.fc3 = nn.Linear(embed_dim//4, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out  

    
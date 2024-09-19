import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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
    
def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.shape[1], 1, device=real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).to(real.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates, 
        inputs=interpolates, 
        grad_outputs=torch.ones(d_interpolates.size(), device=real.device).to(real.device),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
        )[0]
    
    #gradients = gradients.view(real.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty
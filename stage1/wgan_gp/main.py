import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.dataset import GTADataset, UCFDataset
from torch.utils.data import DataLoader

import numpy as np
import os
import random
import wandb
import option
from wgan_gp import *

def train(gta_normal_loader, gta_shooting_loader, gta_fighting_loader, ucf_normal_loader, ucf_shooting_loader, ucf_fighting_loader, args, device):
    
    wandb.init(project='WGAN-GP', 
               name='wgan_each', 
               config=args, 
               settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
               save_code=True)
    
    netF = FeatureAdaptor(in_dim=args.embed_dim).to(device)
    netD = Discriminator(in_dim=args.embed_dim).to(device)
    optimizerF = torch.optim.Adam(netF.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr)

    one = torch.tensor(1.0, device=device)
    mone = torch.tensor(-1.0, device=device)
  
    min_w_distance = 1000
    total_iter = 0

    for epoch in range(args.total_epoch):
        epoch_iter = 0
        ucf_normal_iter = iter(ucf_normal_loader)
        ucf_shooting_iter = iter(ucf_shooting_loader)
        ucf_fighting_iter = iter(ucf_fighting_loader)
        gta_normal_iter = iter(gta_normal_loader)
        gta_shooting_iter = iter(gta_shooting_loader)
        gta_fighting_iter = iter(gta_fighting_loader)
        
        for i in range(min(len(gta_normal_loader), len(gta_shooting_loader), len(ucf_normal_loader), len(ucf_shooting_loader), len(gta_fighting_loader), len(ucf_fighting_loader))):
            ucf_normal_features, _, _ = next(ucf_normal_iter) 
            ucf_shooting_features, _, _ = next(ucf_shooting_iter)
            ucf_fighting_features, _, _ = next(ucf_fighting_iter)
            gta_normal_features, _, _ = next(gta_normal_iter)
            gta_shooting_features, _, _ = next(gta_shooting_iter)
            gta_fighting_features, _, _ = next(gta_fighting_iter)

            ucf_normal_features, ucf_shooting_features, ucf_fighting_features, gta_normal_features, gta_shooting_features, gta_fighting_features = ucf_normal_features.to(device), ucf_shooting_features.to(device), ucf_fighting_features.to(device), gta_normal_features.to(device), gta_shooting_features.to(device), gta_fighting_features.to(device)
            ucf_normal_features, ucf_shooting_features, ucf_fighting_features, gta_normal_features, gta_shooting_features, gta_fighting_features = ucf_normal_features.float(), ucf_shooting_features.float(), ucf_fighting_features.float(), gta_normal_features.float(), gta_shooting_features.float(), gta_fighting_features.float()

            ucf_gta_pair = [(ucf_normal_features, gta_normal_features), (ucf_shooting_features, gta_shooting_features), (ucf_fighting_features, gta_fighting_features)]
            
            Wasserstein_D_list = []
            D_loss_list = []
            F_loss_list = []
            gradient_penalty_list = []
            cosine_similarity_list = []

            for ucf_features, gta_features in ucf_gta_pair:

                # 1. update D network
                for p in netD.parameters():
                    p.requires_grad = True

                for j in range(args.critic_iter):
                    real_data_v = autograd.Variable(ucf_features)
                    netD.zero_grad()
                    
                    D_real = netD(real_data_v)
                    D_real = D_real.mean()
                    D_real.backward(mone)

                    fake = netF(gta_features)
                    inputv = fake
                    D_fake = netD(inputv)
                    D_fake = D_fake.mean()
                    D_fake.backward(one)

                    gp = gradient_penalty(netD, real_data_v.data, fake.data)
                    gp.backward()

                    D_cost = D_fake - D_real + gp 

                    Wasserstein_D = D_real - D_fake 
                    optimizerD.step()

                # 2. update F network
                for p in netD.parameters():
                    p.requires_grad = False

                netF.zero_grad()
                fake = netF(gta_features)
                FA = netD(fake)
                FA = FA.mean()
                cos_sim = F.cosine_similarity(fake, ucf_features, dim=2).mean()
                FA.backward(mone)
                F_cost = -FA
                optimizerF.step()

                Wasserstein_D_list.append(Wasserstein_D.item())
                D_loss_list.append(D_cost.item())
                F_loss_list.append(F_cost.item())
                gradient_penalty_list.append(gp.item())
                cosine_similarity_list.append(cos_sim.item())

            epoch_iter += args.batch_size
            total_iter += args.batch_size
            
            Wasserstein_D_average = sum(Wasserstein_D_list) / 3
            D_cost_average = sum(D_loss_list) / 3
            F_cost_average = sum(F_loss_list) / 3
            gp_average = sum(gradient_penalty_list) / 3
            cos_sim_average = sum(cosine_similarity_list) / 3

            if abs(Wasserstein_D_average) < min_w_distance:
                min_w_distance = abs(Wasserstein_D_average)
                torch.save(netF.state_dict(), f'model/wgan_each_best.pth')

            wandb.log({
            "min_w_distance": min_w_distance,
            "Wasserstein Distance": Wasserstein_D_average,
            "epoch": epoch,
            "F loss": F_cost_average,
            "D loss": D_cost_average,
            "gradient penalty": gp_average,
            "Cosine Similarity": cos_sim_average
            })

    torch.save(netF.state_dict(), f'model/wgan_each.pth')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    seed_everything(args.seed)

    gta_normal_dataset = GTADataset(args.visual_length, args.gta_list, False, None, "Normal")
    gta_normal_loader = DataLoader(gta_normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    gta_shooting_dataset = GTADataset(args.visual_length, args.gta_list, False, None, "Shooting")
    gta_shooting_loader = DataLoader(gta_shooting_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    gta_fighting_dataset = GTADataset(args.visual_length, args.gta_list, False, None, "Fighting")
    gta_fighting_loader = DataLoader(gta_fighting_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    ucf_normal_dataset = UCFDataset(args.visual_length, args.ucf_list, False, None, "Normal")
    ucf_normal_loader = DataLoader(ucf_normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ucf_shooting_dataset = UCFDataset(args.visual_length, args.ucf_list, False, None, "Shooting")
    ucf_shooting_loader = DataLoader(ucf_shooting_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    ucf_fighting_dataset = UCFDataset(args.visual_length, args.ucf_list, False, None, "Fighting")
    ucf_fighting_loader = DataLoader(ucf_fighting_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train(gta_normal_loader, gta_shooting_loader, gta_fighting_loader, ucf_normal_loader, ucf_shooting_loader, ucf_fighting_loader, args, device)

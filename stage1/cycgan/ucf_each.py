import time
import option
from models.models import *
from models.functions import *
import networks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

import numpy as np
from utils.dataset import UCFDataset, GTAUCFDataset

import wandb
import os
from tqdm.auto import tqdm
from utils.tools import setup_seed, save_best_record


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    setup_seed(args.seed)

    # DataLoader
    normal_ucf_dataset = UCFDataset(args.visual_length, args.train_list, False, "Normal")
    normal_ucf_loader = DataLoader(normal_ucf_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    shoot_ucf_dataset = UCFDataset(args.visual_length, args.train_list, False, "Shooting")
    shoot_ucf_loader = DataLoader(shoot_ucf_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fight_ucf_dataset = UCFDataset(args.visual_length, args.train_list, False, "Fighting")
    fight_ucf_loader = DataLoader(fight_ucf_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    normal_gta_dataset = GTAUCFDataset(args.visual_length, args.gta_list, False, "Normal")
    normal_gta_loader = DataLoader(normal_gta_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    shoot_gta_dataset = GTAUCFDataset(args.visual_length, args.gta_list, False, "Shooting")
    shoot_gta_loader = DataLoader(shoot_gta_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fight_gta_dataset = GTAUCFDataset(args.visual_length, args.gta_list, False, "Fighting")
    fight_gta_loader = DataLoader(shoot_gta_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # model
    netF_S = FeatureAdaptor(args.embed_dim).to(device)
    netF_T = FeatureAdaptor(args.embed_dim).to(device)
    netD_S = Discriminator(args.embed_dim).to(device)
    netD_T = Discriminator(args.embed_dim).to(device)
    
    # criterion
    criterionGAN = networks.GANLoss().to(device)
    criterionCycle = nn.L1Loss().to(device)
    criterionIdt = nn.L1Loss().to(device)
    
    # optimizer
    optimizerF = optim.Adam(itertools.chain(netF_S.parameters(), netF_T.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(itertools.chain(netD_S.parameters(), netD_T.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    
    # best loss
    best_loss = 100
    
    # training
    total_iters = 0
    for epoch in tqdm(range(args.epoch_count, args.n_epochs + 1)):

        epoch_iters = 0
        
        # setting before epoch
        normal_ucf_iter = iter(normal_ucf_loader)
        shoot_ucf_iter = iter(shoot_ucf_loader)
        fight_ucf_iter = iter(fight_ucf_loader)
        normal_gta_iter = iter(normal_gta_loader)
        shoot_gta_iter = iter(shoot_gta_loader)
        fight_gta_iter = iter(fight_gta_loader)
        
        # epoch 
        epoch_length = min(len(normal_ucf_loader), len(shoot_ucf_loader), len(fight_ucf_loader), len(normal_gta_loader), len(shoot_gta_loader), len(shoot_gta_loader))
        for i in tqdm(range(epoch_length)):
            
            normal_ucf_features, _, _ = next(normal_ucf_iter)           
            shoot_ucf_features, _, _ = next(shoot_ucf_iter)   
            fight_ucf_features, _, _ = next(fight_ucf_iter)  
            normal_gta_features, _, _ = next(normal_gta_iter)           
            shoot_gta_features, _, _ = next(shoot_gta_iter)   
            fight_gta_features, _, _ = next(fight_gta_iter)   
            
            # source: gta(synthetic) | target: ucf(real)    #()
            source_normal_real = normal_ucf_features.to(device)
            target_nomral_real = normal_gta_features.to(device)
            source_shoot_real = shoot_ucf_features.to(device)
            target_shoot_real = shoot_gta_features.to(device)
            source_fight_real = fight_ucf_features.to(device)
            target_fight_real = fight_gta_features.to(device)
            
            source_target_reals = [(source_normal_real, target_nomral_real), (source_shoot_real, target_shoot_real), (source_fight_real, target_fight_real)]
            
            total_iters += args.batch_size
            epoch_iters += args.batch_size
            
            total_loss = 0
            total_lossF = 0
            total_lossF_S = 0
            total_lossF_T = 0
            total_lossCyc_S = 0
            total_lossCyc_T = 0
            total_lossIdt_S = 0
            total_lossIdt_T = 0
            total_lossD_S = 0
            total_lossD_T = 0
            for source_real, target_real in source_target_reals:     
                 
                # forward
                target_fake = netF_S(source_real)
                source_rec = netF_T(target_fake)    # source cycle
                source_fake = netF_T(target_real)
                target_rec = netF_S(source_fake)    # target cycle
                    
                # set netD require_grad=false when optimizing netF
                set_requires_grad([netD_S, netD_T], False)
                    
                # optimizing netF
                optimizerF.zero_grad()
                lossF, lossF_S, lossF_T, lossCyc_S, lossCyc_T, lossIdt_S, lossIdt_T = backwardF(args, source_real, target_fake, source_rec, target_real, source_fake, target_rec, netF_S, netD_S, netF_T, netD_T, criterionGAN, criterionCycle, criterionIdt)
                lossF.backward()
                optimizerF.step()
                    
                # optimizing netD
                set_requires_grad([netD_S, netD_T], True)
                optimizerD.zero_grad()
                lossD_S, lossD_T = backwardD(args, source_real, target_fake, source_rec, target_real, source_fake, target_rec, netF_S, netD_S, netF_T, netD_T, criterionGAN, criterionCycle, criterionIdt)
                lossD_S.backward()
                lossD_T.backward()
                optimizerD.step()
                    
                print(f"Epoch: {epoch}/{args.n_epochs} Iter: {i+1}/{epoch_length} LossG: {lossF} LossD Source: {lossD_S} LossD Target: {lossD_T}")
                total_lossF += lossF / 3
                total_lossF_S += lossF_S / 3
                total_lossF_T += lossF_T / 3
                total_lossCyc_S += lossCyc_S / 3
                total_lossCyc_T += lossCyc_T / 3
                total_lossIdt_S += lossIdt_S / 3
                total_lossIdt_T += lossIdt_T / 3
                total_lossD_S += lossD_S / 3
                total_lossD_T += lossD_T / 3
                
                total_loss += (lossF + lossD_S + lossD_T) / 3
            
            if total_loss <= best_loss:
                best_loss = total_loss
                save_best_record({"total iters": total_iters}, "ucf_each_best.txt")
                torch.save(netF_S.state_dict(), 'ucf_each_best.pth')
                
            wandb.log({
                "lossF": total_lossF,
                "lossF_S": total_lossF_S,
                "lossF_T": total_lossF_T,
                "lossCyc_S": total_lossCyc_S,
                "lossCyc_T": total_lossCyc_T,
                "lossIdt_S": total_lossIdt_S,
                "lossIdt_T": total_lossIdt_T,
                "lossD_S": total_lossD_S,
                "lossD_T": total_lossD_T,
                "epoch": total_iters,
            })
                
            # save
            if total_iters % args.save_freq == 0:
                torch.save(netF_S.state_dict(), os.path.join(args.model_path, os.path.join('ucf_each', 'ucf_each_' + str(total_iters) + '.pth')))
            
    print("Training End")
    
    torch.save(netF_S.state_dict(), os.path.join(args.model_path, os.path.join('ucf_each', 'ucf_each_' + str(total_iters) + '.pth')))
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
            nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
                
def backwardF(args, source_real, target_fake, source_rec, target_real, source_fake, target_rec, netF_S, netD_S, netF_T, netD_T, criterionGAN, criterionCycle, criterionIdt):
    # GAN loss netD_S(netF_S)
    lossF_S = criterionGAN(netD_S(target_fake), True)
    # GAN loss netD_T(netF_T)
    lossF_T = criterionGAN(netD_T(source_fake), True)
    # forward cycle loss
    lossCyc_S = criterionCycle(source_rec, source_real) * args.lambda_S
    # backward cycle loss
    lossCyc_T = criterionCycle(target_rec, target_real) * args.lambda_T
    
    if args.identity_flag == True:
        source_idt = netF_S(target_real)
        lossIdt_S = criterionIdt(source_idt, target_real) * args.lambda_S * args.lambda_idt
        target_idt = netF_T(source_real)
        lossIdt_T = criterionIdt(target_idt, source_real) * args.lambda_T * args.lambda_idt
    else:
        lossIdt_S = 0.0
        lossIdt_T = 0.0
        
    lossF = lossF_S + lossF_T + lossCyc_S + lossCyc_T + lossIdt_S + lossIdt_T
    
    return lossF, lossF_S, lossF_T, lossCyc_S, lossCyc_T, lossIdt_S, lossIdt_T


def backwardD(args, source_real, target_fake, source_rec, target_real, source_fake, target_rec, netF_S, netD_S, netF_T, netD_T, criterionGAN, criterionCycle, criterionIdt):
    # source true
    target_pred_real = netD_S(target_real)
    lossD_S_real = criterionGAN(target_pred_real, True)
    # source false
    target_pred_fake = netD_S(target_fake.detach())
    lossD_S_fake = criterionGAN(target_pred_fake, False)
    # source combine
    lossD_S = (lossD_S_real + lossD_S_fake) * 0.5
    
    # target true
    source_pred_real = netD_T(source_real)
    lossD_T_real = criterionGAN(source_pred_real, True)
    # target false
    source_pred_fake = netD_T(source_fake.detach())
    lossD_T_fake = criterionGAN(source_pred_fake, False)
    # target combine
    lossD_T = (lossD_T_real + lossD_T_fake) * 0.5
    
    return lossD_S, lossD_T
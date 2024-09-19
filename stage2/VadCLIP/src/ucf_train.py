import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import wandb
from VadCLIP.src.adaptor import FeatureAdaptor
from model import CLIPVAD
from ucf_test import test
from utils.dataset import GTADataset, UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option
import os

def CLASM(logits, labels, lengths, device): 
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0) 
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0) 

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0) 
    return milloss

def CLAS2(logits, labels, lengths, device): 
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, ucf_normal_loader, ucf_anomaly_loader, gta_normal_loader, gta_anomaly_loader, testloader, args, label_map, device):
    
    wandb.login()
    wandb.init(project="VadCLIP_UCF3", 
               name="UCF+GTA_each_{}".format(args.seed), 
               config=args,
               settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
               save_code=True)   

    model.to(device)
    gt = np.load(args.gt_path)

    # load feature adaptor model
    netF = FeatureAdaptor(args.embed_dim).to(device)
    netF.load_state_dict(torch.load('model/ucf3_wgan_each.pth'))
    netF.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map) 
    ap_best = 0
    auc_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        ucf_normal_iter = iter(ucf_normal_loader)
        ucf_anomaly_iter = iter(ucf_anomaly_loader)
        gta_normal_iter = iter(gta_normal_loader)
        gta_anomaly_iter = iter(gta_anomaly_loader)
        for i in range(min(len(ucf_normal_loader), len(ucf_anomaly_loader), len(gta_normal_loader), len(gta_anomaly_loader))):
            step = 0
            ucf_normal_features, ucf_normal_label, ucf_normal_lengths = next(ucf_normal_iter)
            ucf_anomaly_features, ucf_anomaly_label, ucf_anomaly_lengths = next(ucf_anomaly_iter)
            gta_normal_features, gta_normal_label, gta_normal_lengths = next(gta_normal_iter) 
            gta_anomaly_features, gta_anomaly_label, gta_anomaly_lengths = next(gta_anomaly_iter) 

            # make adapted gta features
            with torch.no_grad():
                gta_normal_features = netF(gta_normal_features.to(device)).cpu()
                gta_anomaly_features = netF(gta_anomaly_features.to(device)).cpu()

            visual_features = torch.cat([ucf_normal_features, gta_normal_features, ucf_anomaly_features, gta_anomaly_features], dim=0).to(device) 
            text_labels = list(ucf_normal_label) + list(gta_normal_label) + list(ucf_anomaly_label) + list(gta_anomaly_label) 
            feat_lengths = torch.cat([ucf_normal_lengths, gta_normal_lengths, ucf_anomaly_lengths, gta_anomaly_lengths], dim=0).to(device) 
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths) 

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            loss3 = torch.zeros(1).to(device)

            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * 64 * 2 
            if step % 640 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC1, AP1, AUC2, AP2 = test(model, testloader, args.visual_length, prompt_text, gt, device)
                AUC = max(AUC1, AUC2)
                AP = max(AP1, AP2)    
                
                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': auc_best}
                    torch.save(checkpoint, args.checkpoint_path)

                if AP > ap_best:
                    ap_best = AP
                    
                wandb.log({
                    "epoch": e+1,
                    "AUC1": AUC1,
                    "AUC2": AUC2,
                    "AP1": AP1,
                    "AP2": AP2,
                    "best_AUC": auc_best,
                    "best_AP": ap_best,
                    "loss1": loss_total1 / (i+1),
                    "loss2": loss_total2 / (i+1),
                    "loss3": loss3.item(),
                    "total_loss": loss.item()
                })
                
        scheduler.step()
        
        torch.save(model.state_dict(), 'model/model_cur.pth') 
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path) 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Fighting': 'fighting', 'Shooting': 'shooting'})

    ucf_normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    ucf_normal_loader = DataLoader(ucf_normal_dataset, batch_size=32, shuffle=True, drop_last=True)
    ucf_anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    ucf_anomaly_loader = DataLoader(ucf_anomaly_dataset, batch_size=32, shuffle=True, drop_last=True)

    gta_normal_dataset = GTADataset(args.visual_length, args.gta_train_list, False, label_map, True)
    gta_normal_loader = DataLoader(gta_normal_dataset, batch_size=32, shuffle=True, drop_last=True)
    gta_anomaly_dataset = GTADataset(args.visual_length, args.gta_train_list, False, label_map, False)
    gta_anomaly_loader = DataLoader(gta_anomaly_dataset, batch_size=32, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, ucf_normal_loader, ucf_anomaly_loader, gta_normal_loader, gta_anomaly_loader, test_loader, args, label_map, device)
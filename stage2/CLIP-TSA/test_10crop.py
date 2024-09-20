import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import ipdb
from pathlib import Path
from model import Model
from dataset import Dataset
import argparse
from config import *
import random
from torch.utils.data import DataLoader

def test(dataloader, model, args, viz, device, evals="AUC", wandb_pack=None):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        log = 0

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

            log += input.shape[2]

        if args.dataset in ['sh', 'shanghai']:
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == "ucf":
            gt = np.load('list/gt-ucf.npy')
            normal_gt = np.load('list/gt_ucf_normal.npy')
        elif args.dataset == "xd":
            gt = np.load('list/gt-xd.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
    
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        

        if args.dataset == "xd":
            np.save(f"pr_auc.npy", pr_auc)

            print('AP: ' + str(pr_auc))
            return pr_auc
        else:
            print('AUC @ ROC: ' + str(rec_auc))
            return rec_auc


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--lr', type=str, default='[0.001]*4000', help='learning rates for steps(list form)')
    arg_parser.add_argument('--embed_dim', type=int, default=512, help='vit: 512, i3d: 1024 or 2048, c3d: ?')
    arg_parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 32)')
    arg_parser.add_argument('--k', type=float, default=0.7, help="0 <= k <= 1")
    arg_parser.add_argument('--num_samples', type=int, default=100)
    arg_parser.add_argument('--visual', default='vit', help='vit, i3d, c3d')
    arg_parser.add_argument('--dataset', default='ucf', type=str, help=",".join(["ucf", "sh", "xd"]))
    arg_parser.add_argument('--note', default='None', help='Note')
    arg_parser.add_argument('--seed', type=int, default=2, help='random seed')
    arg_parser.add_argument('--gpu', default="1", type=str)
    arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
    arg_parser.set_defaults(enable_wandb=True)
    arg_parser.add_argument("--disable_HA", dest="enable_HA", action="store_false")
    arg_parser.set_defaults(enable_HA=True)
    
    args = arg_parser.parse_args()
    config = Config(args)
    wandb_config = args.__dict__
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif os.environ["CUDA_VISIBLE_DEVICES"] in ["1", 1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif os.environ["CUDA_VISIBLE_DEVICES"] in ["2", 2]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        
    seed = wandb_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = Model(args.embed_dim, args.batch_size, args.k, args.num_samples, args.enable_HA, args)
    ckpt = torch.load('./ckpt/ucf/clip-tsa_best.pkl')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    auroc = test(test_loader, model, args, None, device)
    
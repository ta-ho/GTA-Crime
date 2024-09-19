import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import ipdb
from pathlib import Path

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
            fight_gt = np.load('list/gt_ucf_fighting.npy')
            shoot_gt = np.load('list/gt_ucf_shooting.npy')
        elif args.dataset == "xd":
            gt = np.load('list/gt-xd.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        
        fight_pred = pred[:len(fight_gt)]
        shoot_pred = pred[len(fight_gt) : len(fight_gt) + len(shoot_gt)]
        normal_pred = pred[len(fight_gt) + len(shoot_gt) : ]

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        
        fight_fpr, fight_tpr, _ = roc_curve(list(np.concatenate((fight_gt, normal_gt))), np.concatenate((fight_pred, normal_pred)))
        fight_auc = auc(fight_fpr, fight_tpr)
        
        shoot_fpr, shoot_tpr, _ = roc_curve(list(np.concatenate((shoot_gt, normal_gt))), np.concatenate((shoot_pred, normal_pred)))
        shoot_auc = auc(shoot_fpr, shoot_tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        
        print(f'fight: {fight_auc}')
        print(f'shoot: {shoot_auc}')

        if args.dataset == "xd":
            np.save(f"pr_auc.npy", pr_auc)

            print('AP: ' + str(pr_auc))
            return pr_auc
        else:
            print('AUC @ ROC: ' + str(rec_auc))
            return rec_auc, fight_auc, shoot_auc

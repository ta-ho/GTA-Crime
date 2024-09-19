import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def train(netF, ucf_nloader, ucf_aloader, gta_nloader, gta_aloader, model, batch_size, optimizer, viz, device, args):
    parallel = 0.5 if "," in args.gpu else 1

    with torch.set_grad_enabled(True):
        model.train()
        
        ucf_ninput, ucf_nlabel = next(ucf_nloader)  # (4, 32, 512)
        ucf_ainput, ucf_alabel = next(ucf_aloader)  # (4, 32, 512)
        gta_ninput, gta_nlabel = next(gta_nloader)  # (12, 32, 512)
        gta_ainput, gta_alabel = next(gta_aloader)  # (12, 32, 512)
        
        # edited
        nlabel = torch.cat((ucf_nlabel, gta_nlabel), 0)
        alabel = torch.cat((ucf_alabel, gta_alabel), 0)
        
        # domain adaptation part
        with torch.no_grad():
            gta_ninput = netF(gta_ninput)
            gta_ainput = netF(gta_ainput)
            
        if parallel == 0.5:
            adjusted_batch_size = int(batch_size * parallel)

            first_half = torch.cat((ucf_ninput[:adjusted_batch_size], gta_ninput[:adjusted_batch_size], ucf_ainput[:adjusted_batch_size], gta_ainput[:adjusted_batch_size]), 0).to(device)  # (16, 32, 512)
            second_half = torch.cat((ucf_ninput[adjusted_batch_size:], gta_ninput[adjusted_batch_size:], ucf_ainput[adjusted_batch_size:], gta_ainput[adjusted_batch_size:]), 0).to(device) # (16, 32, 512)

            input = torch.cat((first_half, second_half), 0).to(device)  # (32, 32, 512)
        else:
            input = torch.cat((ucf_ninput, gta_ninput, ucf_ainput, gta_ainput), 0).to(device)   # (32, 32, 512)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32 x 512   (32 * 32, 512)

        # 여기 32 -> 16
        scores = scores.view(batch_size * 24 * 2, -1)

        scores = scores.squeeze()
        # 여기 32 -> 16
        abn_scores = scores[batch_size * 24:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        return {
            "avg_score_abnormal": torch.mean(score_abnormal),
            "avg_score_normal": torch.mean(score_normal),
            "avg_scores": torch.mean(scores),
            "loss": cost.item(),
            "smooth_loss": loss_smooth.item(),
            "sparsity_loss": loss_sparse.item()
        }


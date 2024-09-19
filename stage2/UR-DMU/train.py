import torch
import torch.nn as nn

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
        
    def forward(self, result, _label):
        loss = {}

        _label = _label.float()

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        b = _label.size(0)//2
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
        A_att = torch.topk(A_att, t//16 + 1, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        return cost, loss



def train(net, netF, normal_ucf_loader, abnormal_ucf_loader, normal_gta_loader, abnormal_gta_loader, optimizer, criterion, step):
    net.train()
    net.flag = "Train"

    ucf_ninput, ucf_nlabel = next(normal_ucf_loader)      
    ucf_ainput, ucf_alabel = next(abnormal_ucf_loader)     
    gta_ninput, gta_nlabel = next(normal_gta_loader)      
    gta_ainput, gta_alabel = next(abnormal_gta_loader)    

    # make adapted gta features
    with torch.no_grad():
        gta_da_ninput = netF(gta_ninput.cuda()).cpu()
        gta_da_ainput = netF(gta_ainput.cuda()).cpu()

    _data = torch.cat((ucf_ninput, gta_da_ninput, ucf_ainput, gta_da_ainput), 0)
    _label = torch.cat((ucf_nlabel, gta_nlabel, ucf_alabel, gta_alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost, loss = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    #for key in loss.keys():     
    #    wind.plot_lines(key, loss[key].item())

    return loss
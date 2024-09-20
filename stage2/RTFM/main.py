from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset, GTADataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import wandb
import random
from adaptor import FeatureAdaptor
import re

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
    args = option.parser.parse_args()
    config = Config(args)
    setup_seed(args.seed)
    itp_unif = re.findall(r'itp|unif', args.weight_path)[0]
    device = torch.device('cuda:'+str(args.gpus))
    each_all = re.findall(r'all|each', args.weight_path)[0]
    numbers = re.findall(r'\d+', args.weight_path)[0]
    device = torch.device('cuda:'+ str(args.gpus))
    wandb.init(entity="HITVision",
               project='RTFM_UCF',
               name=f'UCF+GTA_WGAN_{itp_unif}_{each_all}_snp{numbers}_seed({args.seed})_batch1:1', 
               config=option.parser.parse_args(),
               settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))), save_code=True)

    ucf_train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=16, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    ucf_train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=16, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    gta_train_nloader = DataLoader(GTADataset(args, test_mode=False, is_normal=True), 
                                batch_size=16, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
    gta_train_aloader = DataLoader(GTADataset(args, test_mode=False, is_normal=False),
                                batch_size=16, shuffle=True,
                                num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    netF = FeatureAdaptor(args.feature_size)
    netF.load_state_dict(torch.load(args.weight_path))
    netF = netF.to(device)
    netF.eval()

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = 'output'   # put your own path here
    auc = test(test_loader, model, args, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(ucf_train_nloader) == 0:
            ucf_loadern_iter = iter(ucf_train_nloader)

        if (step - 1) % len(ucf_train_aloader) == 0:
            ucf_loadera_iter = iter(ucf_train_aloader)

        if (step - 1) % len(gta_train_nloader) == 0:
            gta_loadern_iter = iter(gta_train_nloader)

        if (step - 1) % len(gta_train_aloader) == 0:
            gta_loadera_iter = iter(gta_train_aloader)

        loss = train(netF, ucf_loadern_iter, ucf_loadera_iter, gta_loadern_iter, gta_loadera_iter, model, args.batch_size, optimizer, device)
        if step % 5 == 0:

            auc = test(test_loader, model, args, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))

        wandb.log({"AUC": auc, "best_AUC": best_AUC, "loss": loss}, step=step)

        if step % 1000 == 0:
            torch.save(model.state_dict(), './ckpt/' + args.dataset + '{}-i3d.pkl'.format(step))
            model.load_state_dict(torch.load('./ckpt/' + args.dataset + '{}-i3d.pkl'.format(step)))

    # torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')


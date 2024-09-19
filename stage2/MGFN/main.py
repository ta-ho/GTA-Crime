from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils.utils import save_best_record, set_seed

from tqdm import tqdm
from torch.multiprocessing import set_start_method
import wandb
import option
args=option.parse_args()
from config import *
from models.mgfn import mgfn
from datasets.dataset import Dataset, GTADataset
from train import train
from test import test
import datetime
import re
from adaptor import FeatureAdaptor

# def save_config(save_path):
#     path = save_path+'/'
#     os.makedirs(path,exist_ok=True)
#     f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')
#     for key in vars(args).keys():
#         f.write('{}: {}'.format(key,vars(args)[key]))
#         f.write('\n')
# savepath = './ckpt/{}_{}_{}_{}_{}_{}'.format(args.datasetname, args.feat_extractor, args.lr, args.batch_size,args.mag_ratio,
#                                               args.comment)

# save_config(savepath)
try:
     set_start_method('spawn')
except RuntimeError:
    pass


if __name__ == '__main__':
    args=option.parse_args()
    config = Config(args)
    itp_unif = re.findall(r'itp|unif', args.weight_path)[0]
    device = torch.device('cuda:'+str(args.gpus[0]))
    each_all = re.findall(r'all|each', args.weight_path)[0]
    numbers = re.findall(r'\d+', args.weight_path)[0]
    wandb.init(
        entity = 'HITVision',
        project= 'MGFN',  # Replace with your project name   
        name = f'UCF+GTA_WGAN_{itp_unif}_{each_all}_snp{numbers}_seed({args.seed})_batch1:1',
        config=args,
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))), 
        save_code=True
    )

    if args.seed >= 0:
        set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)

    ucf_train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size//2, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True, generator= torch.Generator(device = device))
    ucf_train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size//2, shuffle=False,
                               num_workers=args.workers, pin_memory=False, drop_last=True, generator= torch.Generator(device = device))
    gta_train_nloader = DataLoader(GTADataset(args, test_mode=False, is_normal=True), 
                                batch_size=args.batch_size//2, shuffle=True,
                                num_workers=args.workers, pin_memory=False, drop_last=True, generator= torch.Generator(device = device))
    gta_train_aloader = DataLoader(GTADataset(args, test_mode=False, is_normal=False),
                                batch_size=args.batch_size//2, shuffle=True,
                                num_workers=args.workers, pin_memory=False, drop_last=True, generator= torch.Generator(device = device))    
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)


    model = mgfn()
    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    
    model = model.to(device)

    netF = FeatureAdaptor(args.feature_size)
    netF.load_state_dict(torch.load(args.weight_path))
    netF = netF.to(device)
    netF.eval()

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.0005)
    test_info = {"epoch": [], "test_AUC": [], "test_PR":[]}

    best_AUC = -1
    best_PR = -1 # put your own path here

    # for name, value in model.named_parameters():
    #     print(name)
    iterator = 0
    # test(test_loader, model, args, device)
    
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        # for step in range(1, args.max_epoch + 1):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        cost, loss_smooth, loss_sparse = train(netF.to(device), ucf_train_nloader, ucf_train_aloader, gta_train_nloader, gta_train_aloader, model, args.batch_size, optimizer,
                                                   device, iterator)
        wandb.log({'loss_contrastive': cost}, step=step)

        if step % 5 == 0 and step > 0:
            auc, pr_auc = test(test_loader, model, args, device)
            # wandb.log({'AUC': auc, 'AP': pr_auc}, step=step)

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR"].append(pr_auc)
            if args.datasetname == 'XD':
                if test_info["test_PR"][-1] > best_PR:
                    best_PR = test_info["test_PR"][-1]
                    # torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-i3d.pkl'.format(step))
                    # save_best_record(test_info, os.path.join(savepath + "/", '{}-step-AUC.txt'.format(step)))
                    # Log best PR AUC
                    # wandb.log({'best_pr_auc': best_PR}, step=step)

            else:
                if test_info["test_AUC"][-1] > best_AUC :
                    best_AUC = test_info["test_AUC"][-1]
                    # torch.save(model.state_dict(), savepath + '/' + args.model_name + '{}-i3d.pkl'.format(step))
                    # save_best_record(test_info, os.path.join(savepath + "/", '{}-step-AUC.txt'.format(step)))
                    # Log best AUC
                    wandb.log({'best_auc-roc': best_AUC}, step=step)

    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

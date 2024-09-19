import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
from ucf_test import test
from model import *
from adaptor import FeatureAdaptor

from utils import Visualizer
import os
from dataset_loader import *
from tqdm import tqdm
import wandb

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    wandb.init(
        project="UR-DMU_UCF3",
        name="UCF+GTA_wgan_each_{}".format(args.seed), 
        config=args,
        settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
        save_code=True,
    )

    config = Config(args)
    worker_init_fn = None
    gpus = [0]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    weight_path = args.weight_path
    netF = FeatureAdaptor(config.len_feature)
    netF.load_state_dict(torch.load(weight_path))
    netF = netF.cuda()
    netF.eval()

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag = "Train", a_nums = 60, n_nums = 60)
    net = net.cuda()

    normal_ucf_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 32,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_ucf_train_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 32,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    normal_gta_train_loader = data.DataLoader(
        GTA(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 32 ,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_gta_train_loader = data.DataLoader(
        GTA(root_dir = config.root_dir, mode = 'Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 32 ,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir = config.root_dir, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 1,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}

    best_scores = {
        'best auc': -1,
        'best ap': -1,
        'best ac': -1,
    }   
    best_auc = 0

    criterion = AD_Loss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr = config.lr[0],
        betas = (0.9, 0.999), weight_decay = 0.00005)

    #wind = Visualizer(env = 'UCF_URDMU', port = "2022", use_incoming_socket = False)
    metric = test(net, config, test_loader, test_info, 0)
    #test(net, config, test_loader, test_info, 0)

    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(normal_gta_train_loader) ==0:
            normal_gta_loader_iter = iter(normal_gta_train_loader)

        if (step - 1) % len(abnormal_gta_train_loader) ==0:
            abnormal_gta_loader_iter = iter(abnormal_gta_train_loader)

        if (step - 1) % len(normal_ucf_train_loader) == 0:
            normal_ucf_loader_iter = iter(normal_ucf_train_loader)

        if (step - 1) % len(abnormal_ucf_train_loader) == 0:
            abnormal_ucf_loader_iter = iter(abnormal_ucf_train_loader)
        

        loss_train = train(net, netF, normal_ucf_loader_iter, abnormal_ucf_loader_iter, normal_gta_loader_iter, abnormal_gta_loader_iter, optimizer, criterion, step)
        wandb.log(loss_train, step=step)
        
        if step % 10 == 0 and step > 10:
            metric = test(net, config, test_loader, test_info, step)
            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "ucf_gta_wgan_each_best_record_{}.txt".format(config.seed)))

                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(config.seed)))
            if step == config.num_iters:
                torch.save(net.state_dict(), os.path.join(args.model_path, \
                    "ucf_trans_{}.pkl".format(step)))
            for n,v in metric.items():
                best_name = "best "+ n
                if n!="far":
                    best_scores[best_name] = v if v > best_scores[best_name] else best_scores[best_name]
                else:
                    best_scores[best_name] = v if v < best_scores[best_name] else best_scores[best_name]
        wandb.log(metric, step=step)
        wandb.log(best_scores, step=step)

        if step % 300 == 0 and step > 100:
            net.load_state_dict(torch.load(os.path.join(args.model_path, "ucf_trans_{}.pkl".format(config.seed))))
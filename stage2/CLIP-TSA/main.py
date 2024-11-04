import os
import wandb
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from time import gmtime, strftime
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils.utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
from config import *
from adaptor import FeatureAdaptor

UNIQ_ID = strftime("%Y%m%d_%H%M%S", gmtime())

arg_parser = argparse.ArgumentParser()

np.int = int

# hyperparam
arg_parser.add_argument('--lr', type=str, default='[0.001]*4000', help='learning rates for steps(list form)')
arg_parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 32)')
arg_parser.add_argument('--max_epoch', type=int, default=4000, help='maximum iteration to train (default: 100)')
arg_parser.add_argument('--embed_dim', type=int, default=512, help='vit: 512, i3d: 1024 or 2048, c3d: ?')

# param
arg_parser.add_argument('--k', type=float, default=0.7, help="0 <= k <= 1")
arg_parser.add_argument('--num_samples', type=int, default=100)
arg_parser.add_argument('--visual', default='vit', help='vit, i3d, c3d')

# files/paths
arg_parser.add_argument('--model_name', default='clip-tsa', help='name to save model as')
arg_parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
arg_parser.add_argument(
    '--dataset', 
    default='ucf',
    type=str,
    help=",".join([
        "ucf",
        "sh",
        "xd"
    ])
)
arg_parser.add_argument('--weight_path', default='weights/ucf_unif_each_best.pth')

# miscellaneous
arg_parser.add_argument('--note', default='None', help='Note')
arg_parser.add_argument('--seed', type=int, default=2, help='random seed')
arg_parser.add_argument('--gpu', default="1", type=str)
arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
arg_parser.set_defaults(enable_wandb=True)
arg_parser.add_argument("--disable_HA", dest="enable_HA", action="store_false")
arg_parser.set_defaults(enable_HA=True)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    config = Config(args)
    wandb_config = args.__dict__
    args2 = arg_parser.parse_args()
    args2.dataset = "gta"
    print(f"args1 dataset: {args.dataset}")
    print(f"args2 dataset: {args2.dataset}")

    """
    For manual cuda assignment (e.g., CUDA_VISIBLE_DEVICES=1 python main.py ...)
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif os.environ["CUDA_VISIBLE_DEVICES"] in ["1", 1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    elif os.environ["CUDA_VISIBLE_DEVICES"] in ["2", 2]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    ucf_train_nloader = DataLoader(Dataset(args, is_normal=True, test_mode=False),
                               batch_size=8, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    ucf_train_aloader = DataLoader(Dataset(args, is_normal=False, test_mode=False),
                               batch_size=8, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    gta_train_nloader = DataLoader(Dataset(args2, is_normal=True, test_mode=False),
                               batch_size=8, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    gta_train_aloader = DataLoader(Dataset(args2, is_normal=False, test_mode=False),
                               batch_size=8, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    wandb_config["ucf_train_normal_dataset_size"] = len(ucf_train_nloader.dataset)
    wandb_config["ucf_train_normal_minibatch_size"] = len(ucf_train_nloader)
    wandb_config["ucf_train_abnormal_dataset_size"] = len(ucf_train_aloader.dataset)
    wandb_config["ucf_train_abnormal_minibatch_size"] = len(ucf_train_aloader)
    wandb_config["gta_train_normal_dataset_size"] = len(gta_train_nloader.dataset)
    wandb_config["gta_train_normal_minibatch_size"] = len(gta_train_nloader)
    wandb_config["gta_train_abnormal_dataset_size"] = len(gta_train_aloader.dataset)
    wandb_config["gta_train_abnormal_minibatch_size"] = len(gta_train_aloader)
    wandb_config["test_dataset_size"] = len(test_loader.dataset)
    wandb_config["test_minibatch_size"] = len(test_loader)

    seed = wandb_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = wandb_config["dataset"].lower()
    assert dataset in ["ucf", "gta", "sh", "xd", "shanghai"]

    if dataset in ["sh", "shanghai"] and args.pretrained_ckpt is None:
        assert wandb_config["max_epoch"] >= 15000

    UNIQ_ID = dataset.upper() + UNIQ_ID

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    feature_size = ucf_train_aloader.dataset[0][0].shape[-1]    # 512

    if args.pretrained_ckpt is None:
        model = Model(feature_size, args.batch_size, args.k, args.num_samples, args.enable_HA, args)

        if "," in args.gpu:
            model = torch.nn.DataParallel(model, device_ids=[int(f) for f in args.gpu.split(",")])

        model = model.to(device)

        #ckpt = Path(f"./ckpt/{dataset}/{UNIQ_ID}")
        #auc_record = Path(f"./auc_record/{dataset}/{UNIQ_ID}")
        ckpt = Path(f"./ckpt/{dataset}")
        auc_record = Path(f"./auc_record/{dataset}")

        ckpt.mkdir(exist_ok=True, parents=True)
        auc_record.mkdir(exist_ok=True, parents=True)

        optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

        test_info = { "epoch": [], "test_AUC": [], "best_test_AUC": [], "test_fight_AUC":[], "test_shoot_AUC":[]}
        best_AUC = -1
    else:
        print("[NOTE] For inference, you must specify: --dataset, --gpu, --k, --seed")
        model = Model(feature_size, args.batch_size, args.k, args.num_samples, args.enable_HA)

        if len(args.gpu.split(",")) == 2:
            model = torch.nn.DataParallel(model, device_ids=[0,1])

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

        ckpt = torch.load(str(args.pretrained_ckpt))

        if type(ckpt) == dict and "optimizer_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        auc = test(test_loader, model, args, None, device) # AUC
        quit()

    auc = test(test_loader, model, args, None, device)
    
    ## define a feature adaptor
    netF = FeatureAdaptor(args.embed_dim)
    netF.load_state_dict(torch.load(args.weight_path))
    netF = netF.to(device)
    netF.eval()

    # For wandb users...
    """
    if wandb_config["enable_wandb"]:
        run = wandb.init(
            entity=,
            project=,
            config=wandb_config,
            name=,
            settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__))),
            save_code=True,
        )
    """
        
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

        log = train(netF, ucf_loadern_iter, ucf_loadera_iter, gta_loadern_iter, gta_loadera_iter, model, args.batch_size, optimizer, None, device, args)

        record = {}

        if step % 5 == 0 and step > 150:
            auc = test(test_loader, model, args, None, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, str(ckpt / f"{args.model_name}_best.pkl"))
                save_best_record(test_info, str(auc_record /  f"best-AUC.txt"))
                print("*" * 30 + "RECORD" + "*" * 30)
            
            test_info["best_test_AUC"].append(best_AUC)

            record["test_AUC"] = {
                "epoch": test_info["epoch"][-1],
                "AUC": test_info["test_AUC"][-1],
                "best_test_AUC": test_info["best_test_AUC"][-1]
            }

        if not wandb_config["enable_wandb"]:
            continue
        
        record["epoch"] = step
        record["metrics"] = log

        #wandb.log(record)

    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, str(ckpt / f"{args.model_name}_final.pkl"))

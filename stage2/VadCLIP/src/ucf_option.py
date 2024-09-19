import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)  # 234, 235, 236

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=24, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=15, type=int) 
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--gta-train-list', default='list/newGTA.csv')
parser.add_argument('--train-list', default='list/ucf_SFN_CLIP_rgb.csv') 
parser.add_argument('--test-list', default='list/ucf_SFN_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf_SFN.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])
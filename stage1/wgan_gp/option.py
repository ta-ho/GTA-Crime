import argparse

parser = argparse.ArgumentParser(description='WGAN')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)

parser.add_argument('--critic-iter', default=5, type=int)
parser.add_argument('--total-epoch', default=100, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--ucf-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--gta-list', default='list/newGTA.csv')

parser.add_argument('--lr', default=1e-4)
parser.add_argument('--gp-lambda', default=10.0)
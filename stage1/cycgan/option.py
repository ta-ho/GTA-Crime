import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)

parser.add_argument('--epoch-count', default=1, type=int)
parser.add_argument('--n-epochs', default=100, type=int)
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--gta-list', default='list/newGTA.csv')
parser.add_argument('--save-freq', type=int, default=100, help='frequency of save generator & discriminator')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--identity-flag', type=bool, default=False)
parser.add_argument('--lambda-idt', default=0.5)
parser.add_argument('--lambda-S', default=10.0)
parser.add_argument('--lambda-T', default=10.0)
parser.add_argument('--model-path', default='ckpt')
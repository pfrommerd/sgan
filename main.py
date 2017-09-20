import tensorflow as tf
import argparse

import dataio.cifar
import model

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/sgan_joint')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-bin')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
parser.add_argument('--labloss_weight', type=float, default=1.) # weight for entropy loss
parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

x, y = dataio.cifar.build_data_pipeline(args.data_dir, args.batch_size)

enc0 = model.build_enc0(x)

gen_h1 = model.build_gen1(y, tf.random_uniform(shape=(args.batch_size, 50))) 
gen_x = model.build_gen0(gen_h1, tf.random_uniform(shape=(args.batch_size, 16))) 

disc1_adv, disc1_z_recon = model.build_disc1(gen_h1)

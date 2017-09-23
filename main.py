import numpy as np

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

# -------------- Load stuff -----------------

# Load the input data
real_x, real_y = dataio.cifar.build_data_pipeline(args.data_dir, args.batch_size)

# Load the encoder weights
pre_enc_weights = np.load('pretrained/encoder.npz')
pre_enc_weights = [pre_enc_weights['arr_{}'.format(k)] for k in range(len(pre_enc_weights.files))]
# Transpose the conv filters so that they are tensorflow-compatible
pre_enc_weights[0] = np.transpose(pre_enc_weights[0], (2, 3, 1, 0))
pre_enc_weights[2] = np.transpose(pre_enc_weights[2], (2, 3, 1, 0))

# ------------- Make the model ---------------

enc0 = model.build_enc0(real_x, pre_enc_weights) # x --> fc3 layer
enc1 = model.build_enc1(enc0, pre_enc_weights) # fc3 --> y

# Try generating fc3 --> x
gen_x = model.build_gen0(enc0, tf.random_uniform(shape=(args.batch_size, 16))) 

disc1_adv, disc1_z_recon = model.build_disc0(gen_x, 0.3)



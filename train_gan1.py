import numpy as np

import tensorflow as tf
import argparse

import utils

import dataio.cifar
import model
import os

import scipy.ndimage

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/sgan_gen1')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-bin')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 10000)
parser.add_argument('--steps_per_epoch', type = int, default = 500)
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

real_x, real_y = dataio.cifar.build_data_pipeline(args.data_dir, args.batch_size)

# Load the meanimg data
meanimg = np.load('data/meanimg.npy').transpose([1, 2, 0])
real_x = real_x - meanimg

meanimg = scipy.ndimage.interpolation.zoom(meanimg, (0.5, 0.5, 1), mode='nearest')


# ------------- Make the model ---------------

with tf.variable_scope('enc0'):
    enc0 = model.build_enc0(real_x) # x --> fc3 layer

with tf.variable_scope('enc1'):
    enc1 = model.build_enc1(enc0) # fc3 --> y

with tf.variable_scope('gen1') as scope:
    z1 = tf.random_uniform(shape=(args.batch_size, 50))
    gen_x = model.build_gen1(enc1, z1) - meanimg

with tf.variable_scope('disc1') as scope:
    disc1_gen_adv, disc1_gen_z_recon = model.build_disc1(gen_x, True)
    scope.reuse_variables()
    disc1_real_adv, disc1_real_z_recon = model.build_disc1(enc0, True, reuse=True)

with tf.variable_scope('enc1_recon'):
    gen_enc1_recon = model.build_enc1(gen_x)

# Loss for disc0 and Q0
l_lab1 = tf.boolean_mask(disc1_real_adv, tf.cast(real_y, tf.bool))
l_unl1 = utils.log_sum_exp(disc1_real_adv)
l_gen1 = utils.log_sum_exp(disc1_gen_adv)

loss_disc1_class = -tf.reduce_mean(l_lab1) + tf.reduce_mean(l_unl1)
loss_real1 = -tf.reduce_mean(l_unl1) + tf.reduce_mean(tf.nn.softplus(l_unl1))
loss_fake1 = tf.reduce_mean(tf.nn.softplus(l_gen1))
loss_disc1_adv = 0.5*loss_real1 + 0.5*loss_fake1
loss_gen1_ent = tf.reduce_mean((disc1_gen_z_recon - z1)**2)
loss_disc1 = args.labloss_weight * loss_disc1_class + args.advloss_weight * loss_disc1_adv + args.entloss_weight * loss_gen1_ent

# Loss for generator
loss_gen1_adv = -tf.reduce_mean(tf.nn.softplus(l_gen1))
loss_gen1_cond = tf.reduce_mean((gen_enc1_recon - enc1)**2)
loss_gen1 = args.advloss_weight * loss_gen1_adv + args.condloss_weight * loss_gen1_cond + args.entloss_weight * loss_gen1_ent

# Make the optimizers
disc1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc1')
gen1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen1')

disc0_optimizer = tf.train.AdamOptimizer(learning_rate=args.disc_lr,
            beta1=0.5).minimize(loss_disc1, var_list=disc1_params)

gen0_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr,
            beta1=0.5).minimize(loss_gen1, var_list=gen1_params)

# Tensorboard output summaries
summary_loss_disc0 = tf.summary.scalar('loss_disc1', loss_disc1)
summary_loss_gen0 = tf.summary.scalar('loss_gen1', loss_gen1)

summary_real_img = tf.summary.image('enc_0', enc0 + meanimg, 32)
summary_gen_img = tf.summary.image('gen_x', gen_x + meanimg, 32)

summary_train = tf.summary.merge([summary_loss_disc0, summary_loss_gen0, summary_real_img, summary_gen_img])

writer = tf.summary.FileWriter(args.out_dir, graph=tf.get_default_graph())
writer.flush()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(args.num_epoch):
        print('Epoch %d/%d' % (epoch + 1, args.num_epoch)) 
        for i in range(args.steps_per_epoch):
            print('Batch %d/%d' % (i + 1, args.steps_per_epoch), end='\r')
            if i % 50 == 0:
                _, _, summary= sess.run([disc0_optimizer, gen0_optimizer, summary_train])
                iteration = epoch*args.steps_per_epoch + i
                writer.add_summary(summary, iteration)
                writer.flush()
            else:
                _, _ = sess.run([disc0_optimizer, gen0_optimizer])
        print()


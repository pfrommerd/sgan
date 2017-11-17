import numpy as np

import tensorflow as tf
import argparse

import utils

import dataio.cifar
import model
import os

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/sgan_gen0')
parser.add_argument('--ckpt_dir', type=str, default='checkpoint/sgan_gen0')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-bin')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 1000)
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

# ------------- Make the model ---------------

with tf.variable_scope('enc0'):
    enc0 = model.build_enc0(real_x) # x --> fc3 layer

with tf.variable_scope('enc1'):
    enc1 = model.build_enc1(enc0) # fc3 --> y

with tf.variable_scope('gen0') as scope:
    z0 = tf.random_uniform(shape=(args.batch_size, 16))
    gen_x = model.build_gen0(enc0, z0) - meanimg

with tf.variable_scope('disc0') as scope:
    disc0_gen_adv, disc0_gen_z_recon = model.build_disc0(gen_x, True)
    scope.reuse_variables()
    disc0_real_adv, disc0_real_z_recon = model.build_disc0(real_x, True, reuse=True)

with tf.variable_scope('enc0_recon'):
    gen_enc0_recon = model.build_enc0(gen_x)

# Loss for disc0 and Q0
l_lab0 = tf.boolean_mask(disc0_real_adv, tf.cast(real_y, tf.bool))
l_unl0 = utils.log_sum_exp(disc0_real_adv)
l_gen0 = utils.log_sum_exp(disc0_gen_adv)
loss_disc0_class = -tf.reduce_mean(l_lab0) + tf.reduce_mean(l_unl0)
loss_real0 = -tf.reduce_mean(l_unl0) + tf.reduce_mean(tf.nn.softplus(l_unl0))
loss_fake0 = tf.reduce_mean(tf.nn.softplus(l_gen0))
loss_disc0_adv = 0.5*loss_real0 + 0.5*loss_fake0
loss_gen0_ent = tf.reduce_mean((disc0_gen_z_recon - z0)**2)
loss_disc0 = args.labloss_weight * loss_disc0_class + args.advloss_weight * loss_disc0_adv + args.entloss_weight * loss_gen0_ent

# Loss for generator
loss_gen0_adv = -tf.reduce_mean(tf.nn.softplus(l_gen0))
loss_gen0_cond = tf.reduce_mean((gen_enc0_recon - enc0)**2)
loss_gen0 = args.advloss_weight * loss_gen0_adv + args.condloss_weight * loss_gen0_cond + args.entloss_weight * loss_gen0_ent

# Make the optimizers
disc0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc0')
gen0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen0')

disc0_optimizer = tf.train.AdamOptimizer(learning_rate=args.disc_lr,
            beta1=0.5).minimize(loss_disc0, var_list=disc0_params)

gen0_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr,
            beta1=0.5).minimize(loss_gen0, var_list=gen0_params)

# Tensorboard output summaries
summary_loss_disc0 = tf.summary.scalar('loss_disc0', loss_disc0)
summary_loss_gen0 = tf.summary.scalar('loss_gen0', loss_gen0)

#real_x = tf.minimum(tf.maximum(real_x, 0.0001), 0.999999)
#gen_x = tf.minimum(tf.maximum(gen_x, 0.0001), 0.999999)

summary_real_img = tf.summary.image('real_x', real_x + meanimg, 32)
summary_gen_img = tf.summary.image('gen_x', gen_x + meanimg, 32)
#summary_gen_img = tf.summary.image('gen_x', tf.cast(255 * gen_x, tf.uint8), max_outputs=32)
#summary_real_img = tf.summary.image('real_x', tf.cast(255 * real_x, tf.uint8), max_outputs=32)

summary_train = tf.summary.merge([summary_loss_disc0, summary_loss_gen0, summary_real_img, summary_gen_img])

writer = tf.summary.FileWriter(args.out_dir, graph=tf.get_default_graph())
writer.flush()

saver = tf.train.Saver()

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
        if epoch % 10 == 0:
            saver.save(sess, '%s/ckpt_%d.ckpt' % (args.ckpt_dir, epoch))
        print()

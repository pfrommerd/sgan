import numpy as np

from matplotlib import pyplot as plt
import math
import tensorflow as tf
import argparse

import utils

import dataio.cifar
import model
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/sgan_gen0_sampling')
# The checkpoint to load from
parser.add_argument('--checkpoint', type=str, default='checkpoint/sgan_gen0/ckpt_990.ckpt')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-bin')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--batch_samples', type=int, default=100)

args = parser.parse_args()

# -------------- Load stuff -----------------

real_x, real_y = dataio.cifar.build_data_pipeline(args.data_dir, args.batch_size)

# Load the meanimg data
meanimg = np.load('data/meanimg.npy').transpose([1, 2, 0])
meanimg_enc0 = model.build_enc0(meanimg)

real_x = real_x - meanimg

# ------------- Make the model ---------------

with tf.variable_scope('enc0'):
    enc0 = model.build_enc0(real_x) # x --> fc3 layer

# Separate the input so we can
# sample multiple times for a single image
enc0_input = tf.placeholder(tf.float32, shape=(args.batch_size, 16, 16, 3), name='enc0_input')

with tf.variable_scope('gen0') as scope:
    z0 = tf.random_uniform(shape=(args.batch_size, 16))
    gen_x = model.build_gen0(enc0_input, z0) - meanimg

generated_image = gen_x + meanimg

summary_real_img = tf.summary.image('real_x', enc0_input + meanimg_enc0, 32)
summary_gen_img = tf.summary.image('gen_x', gen_x + meanimg, 32)

summary_sample = tf.summary.merge([summary_real_img, summary_gen_img])

writer = tf.summary.FileWriter(args.out_dir, graph=tf.get_default_graph())
writer.flush()

saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the saved variables
    saver.restore(sess, args.checkpoint)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Get a sample of enc0 level inputs
    enc0_samples = sess.run(enc0)
    # Take the first one and repeat it so we just
    # feed in that input
    enc0_replicated = np.tile(enc0_samples[0], (args.batch_size, 1, 1, 1))
    
    num_samples = args.batch_samples*args.batch_size
    gen_samples = None
    for i in range(args.batch_samples):
        print('Sample batch %d/%d' % (i + 1, args.batch_samples), end='\r')
        gen_sample_batch, summary = sess.run([gen_x, summary_sample], feed_dict={enc0_input: enc0_replicated})
        # Add the sample batch
        if gen_samples is None:
            gen_samples = gen_sample_batch
        else:
            gen_samples = np.append(gen_samples, gen_sample_batch)
        writer.add_summary(summary, i)
        writer.flush()

    # Flatten the samples into vectors
    samples_flattened = np.reshape(gen_samples, (args.batch_samples*args.batch_size, 32*32*3))

    def row_distance(i): # Distance of the ith image
        i = int(i)
        diff_vec = np.sum((samples_flattened - samples_flattened[i])**2, axis=-1)
        min_dist = np.min(diff_vec[np.nonzero(diff_vec)])
        return min_dist
        
    # Set disagonals to infinity so when we take the min we don't get zero
    pairwise_distances = np.fromfunction(np.vectorize(row_distance), (num_samples,), dtype=float)
    # Now that we have all the distances, calculate the entropy
    # using the formula H = 1/n sum(ln(n*dist) + ln(2) + euler's constant
    entropy = 1 / num_samples * np.sum(np.log(num_samples*pairwise_distances)) + np.log(2) + 0.5772156649
    print(entropy)

import tensorflow as tf

import utils

def build_enc0(x):
    enc_conv1 = tf.nn.relu( utils.conv2d((5, 5, 3, 64), name='enc_conv1')(x) )
    enc_pool1 = utils.maxpool2d(pool_size=[1, 2, 2, 1], name='enc_pool1')(enc_conv1)
    enc_conv2 = tf.nn.relu( utils.conv2d((5, 5, 64, 128), name='enc_conv2')(enc_pool1) )
    enc_pool2 = utils.maxpool2d(pool_size=[1, 2, 2, 1], name='enc_pool2')(enc_conv2)
    # Output size of pool2 is (100, 32, 32, 128)
    enc_pool2_flatten = tf.reshape(enc_pool2, [-1, 32*32*128])

    enc_fc3 = tf.nn.relu( utils.dense(num_inputs=32*32*128, num_units=256, bias=True, name='enc_fc3')(enc_pool2_flatten) )

    return enc_fc3

def build_enc1(h1):
    enc_fc4 = tf.nn.softmax( utils.dense(num_inputs=256, num_units=10, bias=True, name='enc_fc4') )

    return enc_fc4

def build_gen1(y, z1):
    gen1_z_embed = tf.layers.batch_normalization(
                    tf.nn.relu(utils.dense(num_inputs=50, num_units=256, bias=True,
                        name='gen1_z_embed')(z1)))
    gen1_y_embed = tf.layers.batch_normalization(
                    tf.nn.relu(utils.dense(num_inputs=10, num_units=512, bias=True,
                        name='gen1_y_embed')(y)))

    gen1_in = tf.concat([gen1_z_embed, gen1_y_embed], axis=1)

    gen1_l1 = tf.layers.batch_normalization(
                    tf.nn.relu(utils.dense(num_inputs=768, num_units=512, bias=True,
                        name='gen1_l1')(gen1_in)))
    gen1_l2 = tf.layers.batch_normalization(
                    tf.nn.relu(utils.dense(num_inputs=512, num_units=512, bias=True,
                        name='gen1_l2')(gen1_l1)))

    gen1_l3 = tf.nn.relu(utils.dense(num_inputs=512, num_units=256, bias=True,
                        name='gen1_l3')(gen1_l2))

    return gen1_l3

def build_gen0(h1, z0):
    gen0_z_embed1 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.dense(num_inputs=16, num_units=128, bias=True,
                            name='gen0_z_embed1')(z0)))
    gen0_z_embed2 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.dense(num_inputs=128, num_units=128, bias=True,
                            name='gen0_z_embed2')(gen0_z_embed1)))

    gen0_in = tf.concat([h1, gen0_z_embed2], axis=1)

    gen0_in_reshaped = tf.reshape(
                        tf.layers.batch_normalization(
                                tf.nn.relu(utils.dense(num_inputs=384, num_units=256*5*5,
                                    name='gen0_embed')(gen0_in))), [-1, 5, 5, 256])

    gen0_deconv1 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.conv2d_transpose((5, 5, 256, 256), (-1, 10, 10, 256),
                            stride=(1, 2, 2, 1), padding='SAME', name='gen0_deconv1')(gen0_in_reshaped)))
    gen0_deconv2 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.conv2d_transpose((5, 5, 128, 256), (-1, 14, 14, 128),
                            padding='VALID', name='gen0_deconv2')(gen0_deconv1)))

    gen0_deconv3 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.conv2d_transpose((5, 5, 128, 128), (-1, 28, 28, 128),
                            stride=(1, 2, 2, 1), padding='SAME', name='gen0_deconv3')(gen0_deconv2)))
    gen0_deconv4 = tf.layers.batch_normalization(
                        tf.nn.relu(utils.conv2d_transpose((5, 5, 3, 128), (-1, 32, 32, 3),
                            padding='VALID', name='gen0_deconv4')(gen0_deconv3)))
    return gen0_deconv4

def build_disc1(h1):
    disc1_l1 = h1 + tf.random_normal(shape=tf.shape(h1), stddev=0.2)
    disc1_l1 = utils.lrelu(utils.dense(num_inputs=256, num_units=512, name='disc1_l1')(disc1_l1))
    disc1_l1 = disc1_l1 + tf.random_normal(shape=tf.shape(disc1_l1), stddev=0.2)

    disc1_l2 = tf.layers.batch_normalization(
                    utils.lrelu(utils.dense(num_inputs=512, num_units=512, name='disc1_l2')(disc1_l1)))
    disc1_l2 = disc1_l2 + tf.random_normal(shape=tf.shape(disc1_l2), stddev=0.2)

    # Now reconstruct the noise
    disc1_z_recon = utils.dense(num_inputs=512, num_units=50, name='disc1_z_recon')(disc1_l2)

    disc1_adv = utils.dense(num_inputs=512, num_units=10, name='disc1_z_adv')(disc1_l2)

    return disc1_adv, disc1_z_recon



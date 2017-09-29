import tensorflow as tf

import utils

def build_enc0(x, preload_weights=[None]*8):
    enc_conv1 = tf.nn.relu( utils.conv2d(x, (5, 5, 3, 64), padding='VALID',
                    weight_preset=preload_weights[0], bias_preset=preload_weights[1],
                        name='enc_conv1'))
    enc_pool1 = utils.maxpool2d(enc_conv1, pool_size=[1, 2, 2, 1], stride=[1, 2, 2, 1], name='enc_pool1')

    enc_conv2 = tf.nn.relu( utils.conv2d(enc_pool1, (5, 5, 64, 128), padding='VALID',
                    weight_preset=preload_weights[2], bias_preset=preload_weights[3],
                        name='enc_conv2'))
    enc_pool2 = utils.maxpool2d(enc_conv2, pool_size=[1, 2, 2, 1], stride=[1, 2, 2, 1], name='enc_pool2')

    # Output size of pool2 is (100, 5, 5, 128)
    enc_pool2_flatten = tf.reshape(enc_pool2, [-1, 5*5*128])

    enc_fc3 = tf.nn.relu( utils.dense(enc_pool2_flatten, num_inputs=32*32*128, num_units=256, bias=True,
            weight_preset=preload_weights[4], bias_preset=preload_weights[5],
                name='enc_fc3'))

    return enc_fc3

def build_enc1(h1, preload_weights=[None]*8):
    enc_fc4 = tf.nn.softmax( utils.dense(h1, num_inputs=256, num_units=10, bias=True,
                    weight_preset=preload_weights[6], bias_preset=preload_weights[7],
                        name='enc_fc4'))
    return enc_fc4

def build_gen1(y, z1):
    gen1_z_embed = utils.batch_normalization(
                    tf.nn.relu(utils.dense(z1, num_inputs=50, num_units=256, bias=True,
                        name='gen1_z_embed')))
    gen1_y_embed = utils.batch_normalization(
                    tf.nn.relu(utils.dense(y, num_inputs=10, num_units=512, bias=True,
                        name='gen1_y_embed')))

    gen1_in = tf.concat([gen1_z_embed, gen1_y_embed], axis=1)

    gen1_l1 = utils.batch_normalization(
                    tf.nn.relu(utils.dense(gen1_in, num_inputs=768, num_units=512, bias=True,
                        name='gen1_l1')))
    gen1_l2 = utils.batch_normalization(
                    tf.nn.relu(utils.dense(gen1_l1, num_inputs=512, num_units=512, bias=True,
                        name='gen1_l2')))

    gen1_l3 = tf.nn.relu(utils.dense(gen1_l2, num_inputs=512, num_units=256, bias=True,
                        name='gen1_l3'))

    return gen1_l3

def build_gen0(h1, z0, preload_weights=32*[None]):
    gen0_z_embed1 = tf.nn.relu(utils.bias(
                        utils.batch_normalization(
                            utils.dense(z0, num_inputs=16, num_units=128, bias=False,
                                weight_preset=preload_weights[0],
                                name='gen0_z_embed1'),
                        scale_preset=preload_weights[2], mean_preset=preload_weights[3],
                        variance_preset=preload_weights[4]),
                        # Bias:
                        (128,), bias_preset=preload_weights[1], name='gen0_z_embed1_bias'))
                            
    gen0_z_embed2 = tf.nn.relu(utils.bias(
                        utils.batch_normalization(
                        utils.dense(gen0_z_embed1, num_inputs=128, num_units=128, bias=False,
                            weight_preset=preload_weights[5],
                            name='gen0_z_embed2'),
                        scale_preset=preload_weights[7], mean_preset=preload_weights[8],
                        variance_preset=preload_weights[9]),
                        # Bias:
                        (128,), bias_preset=preload_weights[6], name='gen0_z_embed2_bias'))

    gen0_in = tf.concat([h1, gen0_z_embed2], axis=1)

    gen0_in_reshaped = tf.reshape(
                    tf.nn.relu(utils.bias(
                        utils.batch_normalization(
                                utils.dense(gen0_in, num_inputs=384, num_units=256*5*5, bias=False,
                                    weight_preset=preload_weights[10],
                                    name='gen0_embed'),
                                scale_preset=preload_weights[12], mean_preset=preload_weights[13],
                                variance_preset=preload_weights[14]),
                        # Bias:
                        (256*5*5,), bias_preset=preload_weights[11], name='gen0_embed_bias')), [-1, 5, 5, 256])

    gen0_deconv1 = tf.nn.relu(utils.bias(
                    utils.batch_normalization(
                        utils.conv2d_transpose(gen0_in_reshaped, (5, 5, 256, 256), (100, 10, 10, 256), bias=False,
                            weight_preset=preload_weights[15],
                            stride=(1, 2, 2, 1), padding='SAME', name='gen0_deconv1'),
                        scale_preset=preload_weights[17], mean_preset=preload_weights[18],
                        variance_preset=preload_weights[19]), 
                    # Bias
                    (256,), bias_preset=preload_weights[16], name='gen0_deconv1_bias'))

    gen0_deconv2 = tf.nn.relu(utils.bias(
                    utils.batch_normalization(
                        utils.conv2d_transpose(gen0_deconv1, (5, 5, 128, 256), (100, 14, 14, 128), bias=False,
                            weight_preset=preload_weights[20], 
                            padding='VALID', name='gen0_deconv2'),
                        scale_preset=preload_weights[22], mean_preset=preload_weights[23],
                        variance_preset=preload_weights[24]),
                    # Bias:
                    (128,), bias_preset=preload_weights[21], name='gen0_deconv2_bias'))

    gen0_deconv3 = tf.nn.relu(utils.bias(
                    utils.batch_normalization(
                        utils.conv2d_transpose(gen0_deconv2, (5, 5, 128, 128), (100, 28, 28, 128), bias=False,
                            weight_preset=preload_weights[25],
                            stride=(1, 2, 2, 1), padding='SAME', name='gen0_deconv3'),
                        scale_preset=preload_weights[27], mean_preset=preload_weights[28],
                        variance_preset=preload_weights[29]),
                    # Bias:
                    (128,), bias_preset=preload_weights[26], name='gen0_deconv3_bias'))

    gen0_deconv4 = utils.conv2d_transpose(gen0_deconv3, (5, 5, 3, 128), (100, 32, 32, 3),
                            weight_preset=preload_weights[30], bias_preset=preload_weights[31],
                            padding='VALID', name='gen0_deconv4')
    return gen0_deconv4

def build_disc1(h1, testing=False, reuse=False):
    disc1_l1 = h1 + tf.random_normal(shape=tf.shape(h1), stddev=0.2)
    disc1_l1 = utils.lrelu(utils.dense(num_inputs=256, num_units=512, name='disc1_l1')(disc1_l1))
    disc1_l1 = disc1_l1 + tf.random_normal(shape=tf.shape(disc1_l1), stddev=0.2)

    disc1_l2 = utils.batch_normalization(
                    utils.lrelu(utils.dense(num_inputs=512, num_units=512, name='disc1_l2')(disc1_l1)), name='bn1', reuse=True)
    disc1_l2 = disc1_l2 + tf.random_normal(shape=tf.shape(disc1_l2), stddev=0.2)

    # Now reconstruct the noise
    disc1_z_recon = utils.dense(num_inputs=512, num_units=50, name='disc1_z_recon')(disc1_l2)

    disc1_adv = utils.dense(num_inputs=512, num_units=10, name='disc1_z_adv')(disc1_l2)

    return disc1_adv, disc1_z_recon

def build_disc0(x, testing=False, reuse=False):
    disc0_l1 = x + tf.random_normal(shape=tf.shape(x), stddev=0.05)
    disc0_l1 = utils.lrelu(utils.conv2d(disc0_l1, (3, 3, 3, 96), name='disc0_conv1'))

    # 32 x 32 --> 16 x 16
    disc0_l2 = utils.batch_normalization(
                    utils.lrelu(utils.conv2d(disc0_l1, (3, 3, 96, 96), stride=[1,2,2,1],
                        name='disc0_conv2')), name='bn1', reuse=reuse)

    disc0_l2 = tf.nn.dropout(disc0_l2, 0.1 if testing else 1)

    # 16 x 16 --> 8x8
    disc0_l3 = utils.batch_normalization(
                    utils.lrelu(utils.conv2d(disc0_l2, (3, 3, 96, 192), stride=[1,2,2,1],
                        name='disc0_conv3')), name='bn2', reuse=reuse)

    # 8x8 --> 8x8
    disc0_l4 = utils.batch_normalization(
                    utils.lrelu(utils.conv2d(disc0_l3, (3, 3, 192, 192),
                        name='disc0_conv4')), name='bn3', reuse=reuse)

    disc0_l4 = tf.nn.dropout(disc0_l4, 0.1 if testing else 1)

    # 8x8 --> 6x6
    disc0_l5 = tf.layers.batch_normalization(
                    utils.lrelu(utils.conv2d(disc0_l4, (3, 3, 192, 192), padding='VALID',
                        name='disc0_conv5')), name='bn4', reuse=reuse)

    disc0_l5 = tf.reshape(disc0_l5, [100, 6, 6, 192])
    disc0_shared = utils.lrelu(utils.network_in_network(disc0_l5, 192, num_units=192, name='disc0_shared'))
    disc0_shared_flat = tf.reshape(disc0_shared, [-1, 192*6*6])
    disc0_z_recon = utils.dense(disc0_shared_flat, num_inputs=192*6*6, num_units=16, name='disc0_z_recon')
    
    disc0_shared_pool = tf.reduce_mean(disc0_shared, [1, 2])
    disc0_adv = utils.dense(disc0_shared_pool, num_inputs=192, num_units=10, name='disc1_z_adv')
    # disc0_adv is the pre-softmax classification output for the discriminator

    return disc0_adv, disc0_z_recon

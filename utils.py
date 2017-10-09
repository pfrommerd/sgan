import tensorflow as tf
import os, sys, tarfile, shutil
import tempfile, urllib

#from tensorflow.contrib.layers import xavier_initializer
default_initializer = tf.random_normal_initializer(0, 0.02)

def conv2d(x, filter_size, stride=[1, 1, 1, 1], padding='SAME', bias=True,
            weight_init=default_initializer, bias_init=tf.zeros_initializer,
            weight_preset=None, bias_preset=None, name=None):
    with tf.variable_scope(name):
        filter_weights = weight_preset if weight_preset is not None else \
                  tf.get_variable('weights',filter_size, initializer=weight_init)
        if bias:
            filter_bias = bias_preset if bias_preset is not None else \
                            tf.get_variable('bias', [filter_size[-1]], initializer=bias_init)
            return tf.nn.conv2d(x, filter_weights, stride, padding) + filter_bias
        else:
            return tf.nn.conv2d(x, filter_weights, stride, padding) + filter_bias

def conv2d_transpose(x, filter_size, output_shape,
        stride=[1, 1, 1, 1], padding='SAME', bias=True,
        weight_preset=None, bias_preset=None,
        weight_init=default_initializer, bias_init=tf.zeros_initializer, name=None):
    with tf.variable_scope(name):
        filter_weights = weight_preset if weight_preset is not None else \
                tf.get_variable('weights', filter_size, initializer=weight_init)
        if bias:
            filter_bias = bias_preset if bias_preset is not None else \
                    tf.get_variable('bias', [filter_size[-2]], initializer=bias_init)
            return tf.nn.conv2d_transpose(x, filter_weights, output_shape, stride, padding, name=name) + filter_bias
        else:
            return tf.nn.conv2d_transpose(x, filter_weights, output_shape, stride, padding, name=name)

def maxpool2d(x, pool_size=[1, 1, 1, 1], stride=[1, 1, 1, 1], padding='SAME', name=None):
    return tf.nn.max_pool(x, pool_size, stride, padding, name=name) 

def batch_normalization(x, reuse=None,
        mean_preset=None, variance_preset=None, offset_preset=None, scale_preset=None,
        name=None):
    if mean_preset is not None or \
        variance_preset is not None or \
        offset_preset is not None or \
        scale_preset is not None:
        return (x - mean_preset) / tf.sqrt(1e-6 + variance_preset) * scale_preset
    else:
        return tf.layers.batch_normalization(x, reuse=reuse, name=name)

#def batch_normalization(x, reuse=None,
#        mean_preset=None, variance_preset=None, offset_preset=None, scale_preset=None,
#        name=None):
#    if mean_preset is not None or \
#        variance_preset is not None or \
#        offset_preset is not None or \
#        scale_preset is not None:
#        return tf.nn.batch_normalization(x, mean_preset, variance_preset,
#                    offset_preset, scale_preset, 0.000001, name=name)
#    else:
#        return tf.layers.batch_normalization(x, reuse=reuse, name=name)

def dense(x, num_inputs, num_units, bias=True,
        weight_init=default_initializer, bias_init=tf.zeros_initializer,
        weight_preset=None, bias_preset=None,
        name=None):

    with tf.variable_scope(name):
        weights = weight_preset if weight_preset is not None else \
                    tf.get_variable('weights',
                        [num_inputs, num_units], initializer=weight_init)
        if bias:
            bias_weights = bias_preset if bias_preset is not None else \
                    tf.get_variable('bias', [num_units], initializer=weight_init)

            return tf.matmul(x, weights) + bias_weights
        else:
            return tf.matmul(x, weights)

def network_in_network(x, num_input_channels, num_units, bias=True,
        weight_init=default_initializer, bias_init=tf.zeros_initializer, name=None):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [num_input_channels, num_units], initializer=weight_init)
        # Broadcast the network in network across the 2nd and third dimensions
        # allows mixing across the different channels
        if bias:
            bias_weights = tf.get_variable('bias', [num_units], initializer=bias_init)
            # Broadcast over everything but the channel dimension
            # (i.e one set of weights that gets applied to each image and pixel and dotted along the filter dimension)
            return tf.transpose(tf.tensordot(weights, x, [[0], [3]]), [1, 2, 3, 0]) + bias_weights
        else:
            return tf.transpose(tf.tensordot(weights, x, [[0], [3]]), [1, 2, 3, 0])

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def bias(x, shape, bias_preset = None, bias_init=tf.zeros_initializer, name=None):
    bias = tf.get_variable(name, shape, initializer=bias_init) if bias_preset is None else \
            bias_preset

    return x + bias

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, axis=axis)
    return m + tf.log(tf.reduce_sum(tf.exp(x - tf.expand_dims(m, axis=axis)), axis=axis))

def logits_sigmoid_cross_entropy(logits, target):
    values = tf.cast(tf.fill(tf.stack([tf.shape(logits)[0], 1]), target), tf.float32)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=values)

def diff_summary(name, diff, max_outputs=3):
   return tf.summary.image(name, tf.cast(255 * (diff + 1.0) / 2.0, tf.uint8), max_outputs=max_outputs)

def filter_inputs(inputs, shapes, filter_test):
    match = filter_test(inputs)
    def real():
        return [tf.expand_dims(i,0) for i in inputs]
    def fake():
        return [tf.zeros((0,) + s, dtype=i.dtype) for i, s in zip(inputs, shapes)]

    return tf.cond(match, real, fake)


def cond_wget_untar(dest_dir, conditional_files, wget_url, moveFiles=()):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Determine if we need to download
    if not files_exist(conditional_files):
        filename = wget_url.split('/')[-1]
        filepath = os.path.join(tempfile.gettempdir(), filename)
        # Download
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(wget_url, filepath,
                                                 reporthook=_progress)
        print()
        print('Downloaded %s, extracting...' % filename)
        tarfile.open(filepath, 'r:gz').extractall(tempfile.gettempdir())

        for src, tgt in moveFiles:
            shutil.move(os.path.join(tempfile.gettempdir(), src), tgt)

def join_files(dir, files):
    return [os.path.join(dir, f) for f in files]

def files_exist(files):
    return all([os.path.isfile(f) for f in files])

def file_exists(f):
    return os.path.isfile(f)

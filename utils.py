import tensorflow as tf
import os, sys, tarfile, shutil
import tempfile, urllib

#from tensorflow.contrib.layers import xavier_initializer
default_initializer = lambda : tf.random_normal_initializer(0, 0.02)

def conv2d(filter_size, stride=[1, 1, 1, 1], padding='SAME', bias=True, weight_init=default_initializer(), bias_init=tf.zeros_initializer, name=None):
    filter_weights = tf.get_variable(('%s_weights' % (name,)), filter_size, initializer=weight_init)
    if bias:
        filter_bias = tf.get_variable(('%s_bias' % (name,)), [filter_size[-1]], initializer=bias_init)
        return lambda x: tf.nn.conv2d(x, filter_weights, stride, padding, name=name) + filter_bias
    else:
        return lambda x: tf.nn.conv2d(x, filter_weights, stride, padding, name=name)

def conv2d_transpose(filter_size, output_shape, stride=[1, 1, 1, 1], padding='SAME', bias=True,
        weight_init=default_initializer(), bias_init=tf.zeros_initializer, name=None):
    filter_weights = tf.get_variable(('%s_weights' % (name,)), filter_size, initializer=weight_init)
    if bias:
        filter_bias = tf.get_variable(('%s_bias' % (name,)), [filter_size[-2]], initializer=bias_init)
        return lambda x: tf.nn.conv2d_transpose(x, filter_weights, output_shape, stride, padding, name=name) + filter_bias
    else:
        return lambda x: tf.nn.conv2d_transpose(x, filter_weights, output_shape, stride, padding, name=name)

def maxpool2d(pool_size, stride=[1, 1, 1, 1], padding='SAME', name=None):
    return lambda x: tf.nn.max_pool(x, pool_size, stride, padding, name=name) 

def dense(num_inputs, num_units, bias=False, weight_init=default_initializer(), name=None):
    print(name)
    weights = tf.get_variable(('%s_weights' % (name,)),
                    [num_inputs, num_units], initializer=weight_init)
    if bias:
        bias_weights = tf.get_variable(('%s_bias' % (name,)), [num_units], initializer=weight_init)
        return lambda x: tf.matmul(x, weights) + bias_weights
    else:
        return lambda x: tf.matmul(x, weights)

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

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

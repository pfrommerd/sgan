import tensorflow as tf

#import numpy as np

import itertools
import utils
import time

TRAIN_FILES = ['data_batch_1.bin', 'data_batch_2.bin',
               'data_batch_3.bin', 'data_batch_4.bin',
               'data_batch_5.bin']
TEST_FILES = ['test_batch.bin']

NUM_IMAGE_BYTES = 32 * 32 * 3
NUM_LABEL_BYTES = 1
NUM_RECORD_BYTES = NUM_IMAGE_BYTES + NUM_LABEL_BYTES

# Downloads and processes data to a subdirectory in directory
# returns the training data and testing data pyramids as two lists in a tuple
def build_data_pipeline(data_directory, batch_size, preload=True, test=False):
    train_files = utils.join_files(data_directory, TRAIN_FILES)
    test_files = utils.join_files(data_directory, TEST_FILES)

    utils.cond_wget_untar(data_directory,
                           train_files + test_files,
                           'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
                            moveFiles=zip(utils.join_files('cifar-10-batches-bin', TRAIN_FILES + TEST_FILES),
                                            utils.join_files(data_directory, TRAIN_FILES + TEST_FILES)))

    # Images are 32x32x3 bytes, with an extra byte at the start for the label
    if test:
        return _file_pipeline(batch_size, utils.join_files(data_directory, TEST_FILES), preload=preload, test=True)
    else:
        return _file_pipeline(batch_size, utils.join_files(data_directory, TRAIN_FILES), preload=preload)

def _file_pipeline(batch_size, files, preload=None, test=False):
#    record_pipeline = None
    # Load into a numpy array
#    buf = np.concatenate([np.fromfile(f, dtype=np.uint8) for f in files])
#    buf = np.reshape(buf, (-1, NUM_RECORD_BYTES))
    # Keep rows starting with a 0
    #buf = buf[buf[:,0] == 0, :]

    # Constantly load the buffer into a queue
#    io_queue = tf.FIFOQueue(batch_size, [tf.uint8])
#    io_enqueue = io_queue.enqueue_many(tf.constant(buf))

#    io_runner = tf.train.QueueRunner(io_queue, [io_enqueue])
#    tf.train.add_queue_runner(io_runner)

#    record_pipeline = io_queue.dequeue()

    reader = tf.FixedLengthRecordReader(record_bytes=NUM_RECORD_BYTES)
    _, value = reader.read(tf.train.string_input_producer(files))
    record_pipeline = tf.decode_raw(value, tf.uint8)
    return _process_pipeline(batch_size, record_pipeline)

def _process_pipeline(batch_size, record_pipeline): 
    label = tf.cast(tf.strided_slice(record_pipeline, [0], [NUM_LABEL_BYTES]), tf.int32)
    label = tf.reshape(tf.one_hot(label,10, dtype=tf.int32), [10])
    depth_major = tf.reshape(tf.strided_slice(record_pipeline, [NUM_LABEL_BYTES],
                                [NUM_LABEL_BYTES + NUM_IMAGE_BYTES]),
                                [3, 32, 32])

    img = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32) / 255.0

    batch_label, batch_img = \
            tf.train.shuffle_batch([label, img],
                            enqueue_many=False,
                            batch_size=batch_size,
                            num_threads=3,
                            capacity=100 * batch_size, min_after_dequeue=10 * batch_size)

    return batch_img, batch_label

# -*- coding: utf-8 -*-
# file: train_lstm.py
# author: JinTian
# time: 04/03/2017 7:59 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import tensorflow as tf
import os
import pickle
import math


tf.app.flags.DEFINE_string('tf_record_dir', os.path.abspath('./data/tf_records/'), 'tf records directory.')
tf.app.flags.DEFINE_string('map_file', './data/tf_records/novel.pkl', 'map pkl file.')

tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size of training.')
tf.app.flags.DEFINE_integer('embedding_size', 200, """the size of embedding. embed should be shape (num_chars_set,
                                                   embedding_size).""")
tf.app.flags.DEFINE_integer('num_epochs', 10, 'epochs of training.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate of training.')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'slim train log dir.')
tf.app.flags.DEFINE_boolean('restore', True, 'if restore or not.')

tf.app.flags.DEFINE_integer('max_steps', 1000, 'training max steps.')
tf.app.flags.DEFINE_integer('save_steps', 30, 'training save steps.')
tf.app.flags.DEFINE_integer('eval_steps', 1, 'training eval steps.')

tf.app.flags.DEFINE_string('checkpoints_prefix', 'novel', 'prefix for checkpoints file.')

FLAGS = tf.app.flags.FLAGS

with open(FLAGS.map_file, 'rb') as f:
    map_dict = pickle.load(f)
    num_chars_set = map_dict['num_chars_set']


def tf_records_walker(tf_records_dir):
    all_files = [os.path.abspath(os.path.join(tf_records_dir, i_)) for i_ in os.listdir(tf_records_dir)]
    if all_files:
        print("[INFO] %s files were found under current folder. " % len(all_files))
        print("[INFO] Please be noted that only files end with '*.tf record' will be load!")
        tf_records_files = [i for i in all_files if os.path.basename(i).endswith('tfrecord')]
        if tf_records_files:
            for i_ in tf_records_files:
                print('[INFO] loaded train tf_records file at: {}'.format(i_))
            return tf_records_files
        else:
            raise FileNotFoundError("Can not find any records file.")
    else:
        raise Exception("Cannot find any file under this path.")


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'text/seq_x': tf.FixedLenFeature([100], tf.int64),
            'text/seq_y': tf.FixedLenFeature([1], tf.int64),
        })
    seq_x = tf.cast(features['text/seq_x'], dtype=tf.int32)
    seq_y = tf.cast(features['text/seq_y'], dtype=tf.int32)

    return seq_x, seq_y


def inputs(batch_size):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            tf_records_walker(tf_records_dir=FLAGS.tf_record_dir),
            num_epochs=None,
            shuffle=True)
        seq_x, seq_y = read_and_decode(filename_queue)

        seq_x, seq_y = tf.train.shuffle_batch(
            [seq_x, seq_y],
            batch_size=batch_size,
            num_threads=2,
            capacity=10 + 3 * batch_size,
            min_after_dequeue=10)

        return seq_x, seq_y


def run_training():
    seq_x, seq_y = inputs(FLAGS.batch_size)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        try:
            while not coord.should_stop():
                seq_x_, seq_y_ = sess.run([seq_x, seq_y])
                print('seq_x: ', seq_x_)
                print('seq_y: ', seq_y_)

                train_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])

                embedding_dict = tf.Variable(tf.random_uniform([num_chars_set, FLAGS.embedding_size], -1.0, 1.0))

                nce_weights = tf.Variable(tf.truncated_normal([num_chars_set, FLAGS.embedding_size],
                                                              stddev=1.0/math.sqrt(FLAGS.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([num_chars_set]))

                embed = tf.nn.embedding_lookup(embedding_dict, train_inputs)

                loss = tf.reduce_mean(tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=
                ))

        except tf.errors.OutOfRangeError:
            print('[INFO] train finished.')
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
        finally:
            coord.request_stop()
            coord.join(threads)

        coord.request_stop()
        coord.join(threads)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()

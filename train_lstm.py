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


tf.app.flags.DEFINE_string('record_file', os.path.abspath('./tf_records/novel.tfrecord'), 'tfrecord file path.')
tf.app.flags.DEFINE_string('map_file', './tf_records/novel.pkl', 'map pkl file.')

tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size of training.')
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


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'text/seq_x': tf.FixedLenFeature([], tf.int64),
            'text/seq_y': tf.FixedLenFeature([], tf.int64),
        })
    seq_x = tf.cast(features['text/seq_x'], dtype=tf.int32)
    seq_y = tf.cast(features['text/seq_y'], dtype=tf.int32)
    seq_x = seq_x/num_chars_set
    seq_y = seq_y/num_chars_set

    return seq_x, seq_y


def inputs(is_train, batch_size, one_hot_labels):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [FLAGS.record_file],
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
    with tf.Session() as sess:
        sess.run()


def main(_):
    print('[INFO] Reading from %s' % FLAGS.record_file)
    run_training()


if __name__ == '__main__':
    tf.app.run()

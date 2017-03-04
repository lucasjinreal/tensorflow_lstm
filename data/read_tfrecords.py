# -*- coding: utf-8 -*-
# file: read_tfrecords.py
# author: JinTian
# time: 04/03/2017 9:25 PM
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

FLAGS = tf.app.flags.FLAGS


def read_records():
    record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.record_file)

    with open('./tf_records/novel.pkl', 'rb') as f:
        map_dict = pickle.load(f)
        print(map_dict['num_chars_set'])

    with tf.Session() as sess:
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            sentence_in_bytes = example.features.feature['text/sentence_in'].bytes_list.value[0]

            seq_x = example.features.feature['text/seq_x'].int64_list.value
            seq_y = example.features.feature['text/seq_y'].int64_list.value

            sentence_in = sentence_in_bytes.decode('utf-8')
            print(sentence_in)
            print(seq_x)
            print(seq_y)


def main(_):
    print('[INFO] Reading from %s' % FLAGS.record_file)
    read_records()


if __name__ == '__main__':
    tf.app.run()
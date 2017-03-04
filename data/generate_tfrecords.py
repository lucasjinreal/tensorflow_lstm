# -*- coding: utf-8 -*-
# file: generate_tfrecords.py
# author: JinTian
# time: 04/03/2017 8:46 PM
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
"""
this file convert a txt file into tf record
for read tf_record, you just need get example
and:
text/sentence_in: original sentence text tf.bytes
text/sentence_out: original sentence text tf.bytes
text/seq_x: the input array tf.int64 list
text/seq_y: the output array tf.int64 list
"""
import tensorflow as tf
import numpy as np
import jieba
import os
import pickle


tf.app.flags.DEFINE_string('txt_file', './原来你还在这里.txt', 'path of input text file.')
tf.app.flags.DEFINE_integer('seq_length', 100, 'length for per simple input.')
tf.app.flags.DEFINE_integer('steps', 1, 'text serialize step.')

tf.app.flags.DEFINE_string('output_dir', './tf_records', 'Output data directory')
tf.app.flags.DEFINE_string('tf_record_prefix', 'novel', 'prefix for saving text to tf records.')
tf.app.flags.DEFINE_string('map_file_dir', './tf_records', 'map file for int back to words.')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(sentence_in, sentence_out, seq_x, seq_y):
    example = tf.train.Example(features=tf.train.Features(feature={
        'text/sentence_in': _bytes_feature(tf.compat.as_bytes(sentence_in)),
        'text/sentence_out': _bytes_feature(tf.compat.as_bytes(sentence_out)),
        'text/seq_x': _int64_feature(seq_x),
        'text/seq_y': _int64_feature(seq_y),
    }))
    return example


def segment_text(txt):
    seg_all = jieba.cut(txt, cut_all=False)
    seg_list = jieba.lcut(txt, cut_all=False)
    chars = list(set(seg_all))
    return seg_list, chars


def _process_text():
    raw_text = []
    with open(FLAGS.txt_file, 'r') as f:
        for l in f.readlines():
            raw_text.append(l.strip())
    raw_text = ''.join(list(filter(None, raw_text)))
    segment_list, all_chars_set = segment_text(raw_text)

    all_length = len(segment_list)
    num_chars_set = len(all_chars_set)
    characters_to_int = dict((c, i) for i, c in enumerate(all_chars_set))
    characters_to_int['num_chars_set'] = num_chars_set

    chars_dict_file = os.path.join(FLAGS.map_file_dir, FLAGS.tf_record_prefix + '.pkl')
    with open(chars_dict_file, 'wb') as f:
        pickle.dump(characters_to_int, f)
    print('[INFO] map dictionary has been saved into %s.' % chars_dict_file)

    output_filename = '%s.tfrecord' % FLAGS.tf_record_prefix
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    j = 0
    for i in range(0, all_length - FLAGS.seq_length, FLAGS.steps):
        seq_in = segment_list[i: i + FLAGS.seq_length]
        seq_out = segment_list[i + FLAGS.seq_length]

        seq_in_serial = [characters_to_int[c] for c in seq_in]
        seq_out_serial = characters_to_int[seq_out]

        example = _convert_to_example(''.join(seq_in), ''.join(seq_out), seq_in_serial, seq_out_serial)
        writer.write(example.SerializeToString())
        print('[INFO] Finish write %d sequence.' % i)
        j = (j+1)
    print('[INFO] Finished. tf record file contains %d sequences.' % j)


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    _process_text()


if __name__ == '__main__':
    tf.app.run()
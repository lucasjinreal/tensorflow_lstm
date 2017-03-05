# -*- coding: utf-8 -*-
# file: test.py
# author: JinTian
# time: 05/03/2017 6:57 PM
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
import numpy as np
import random
from random import shuffle

NUM_EXAMPLES = 10000

def embed_test():
    with tf.Session() as sess:
        embedding_dict = tf.random_uniform([388, 30], -1.0, 1.0)
        print(embedding_dict.eval())
        train_inputs = tf.constant(
            [
                [1, 3, 56, 67, 4],
                [4, 6, 4, 56, 56],
                [4, 9, 34, 12, 4]
            ]
        )
        embed = tf.nn.embedding_lookup(params=embedding_dict, ids=train_inputs)
        print('embed shape: ', embed.get_shape())


def test():
    train_input = ['{0:020b}'.format(i) for i in range(2 ** 10)]
    print(train_input)
    print(len(train_input))


def zeroes_ones():
    train_input = ['{0:020b}'.format(i) for i in range(2 ** 10)]
    shuffle(train_input)
    train_input = [map(int, i) for i in train_input]
    ti = []
    for i in train_input:
        temp_list = []
        for j in i:
            temp_list.append([j])
        ti.append(np.array(temp_list))
    train_input = ti

    train_output = []
    for i in train_input:
        count = 0
        for j in i:
            if j[0] == 1:
                count += 1
        temp_list = ([0] * 21)
        temp_list[count] = 1
        train_output.append(temp_list)

    test_input = train_input[NUM_EXAMPLES:]
    test_output = train_output[NUM_EXAMPLES:]
    train_input = train_input[:NUM_EXAMPLES]
    train_output = train_output[:NUM_EXAMPLES]

    print(test_input)
    print(test_output)

    print('[INFO] test and training data loaded')

    # data = tf.placeholder(tf.float32, [None, 20, 1])
    # target = tf.placeholder(tf.float32, [None, 21])
    #
    # num_hidden = 24
    # cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    # val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    # val = tf.transpose(val, [1, 0, 2])
    # last = tf.gather(val, int(val.get_shape()[0]) - 1)
    #
    # weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    # bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    # prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    #
    # cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    # optimizer = tf.train.AdamOptimizer()
    # minimize = optimizer.minimize(cross_entropy)
    # mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    # error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    #
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     batch_size = 1000
    #     no_of_batches = int(len(train_input)) / batch_size
    #
    #     epoch = 5000
    #     for i in range(epoch):
    #         ptr = 0
    #         for j in range(no_of_batches):
    #             inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
    #             ptr += batch_size
    #             sess.run(minimize, {data: inp, target: out})
    #         print("Epoch ", str(i))
    #     incorrect = sess.run(error, {data: test_input, target: test_output})
    #     print(sess.run(prediction, {
    #         data: [
    #             [[1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0]]]}))
    #     print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))




if __name__ == '__main__':
    zeroes_ones()
    # test()

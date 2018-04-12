# -*- coding: utf-8 -*-
from common import BaseModel

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import learn as tflearn_old
import tflearn



"""
****************************************************************************************************************************
"""
class RNNSampleModel(BaseModel):
    def __init__(self,n_hidden=128,n_class=10):
        self.n_hidden = n_hidden
        self.n_class = n_class

    def build_graph(self, input, expected, reuse):



        with tf.variable_scope("variables", reuse=reuse):
            W = tf.get_variable("weight",[self.n_hidden,self.n_class],
                                initializer=tf.random_normal_initializer())
            b = tf.get_variable("bias", [self.n_class],
                                initializer=tf.random_normal_initializer())


        with tf.variable_scope('mainRNN', reuse=reuse):

            ######bidirectional_rnn######
            cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden/2, state_is_tuple=True, activation=tf.tanh,
                                                  reuse=reuse);
            cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden/2, state_is_tuple=True, activation=tf.tanh,
                                                  reuse=reuse);
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, input, dtype=tf.float32);

            outputs = tf.concat([outputs[0], outputs[1]], axis=2)



            '''
            ######dynamic_rnn######

            #GRUCell
            #cell = tf.nn.rnn_cell.GRUCell(num_units=self.n_hidden, activation=tf.tanh,
            #                                    reuse=reuse);

            #LSTMCell
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden, state_is_tuple=True, activation=tf.tanh,
                                                       reuse=reuse);
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            '''

            #using only last step
            outputs = outputs[:,-1]
            outputs = tf.matmul(outputs, W) + b

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=expected))




        return dict(prediction=outputs, loss=cost)



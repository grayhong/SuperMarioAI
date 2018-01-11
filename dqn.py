import tensorflow as tf
import numpy as np
import os

from ops import *


class DQN(object):

    def __init__(self, sess, name):
        self.n_input = 84 * 84
        self.n_size = 84
        self.n_channel = 1
        self.n_hidden = 256
        self.n_actions = 14
        self.sess = sess
        self.name = name
        self.learning_rate = 1e-4


        self.d1 = 11

        self.weights ={
            'w1': tf.Variable(tf.random_normal([self.d1 * self.d1 * 64, self.n_hidden], stddev=0.01)),
            'w2': tf.Variable(tf.random_normal([self.n_hidden, self.n_actions], stddev=0.01)),
        }

        self.biases ={
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'b2': tf.Variable(tf.zeros([self.n_actions])),
        }

        self.build()

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        


    def build(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, self.n_size, self.n_size, 4], name='state')
            self.Y = tf.placeholder(tf.float32, [None, self.n_actions], name='reward')

            conv_1 = tf.nn.relu(batch_normal(conv2d(self.X, output_dim = 32, k_h=8, k_w=8, d_h=4, d_w=4, name = 'conv_1'), scope='bn_1'))
            conv_2 = tf.nn.relu(batch_normal(conv2d(conv_1, output_dim = 64, k_h=4, k_w=4, d_h=2, d_w=2, name = 'conv_2'), scope='bn_2'))
            conv_3 = tf.nn.relu(batch_normal(conv2d(conv_2, output_dim = 64, k_h=3, k_w=3, d_h=1, d_w=1, name = 'conv_3'), scope='bn_3'))

            conv_3 = tf.reshape(conv_3 , [-1, self.d1 * self.d1 * 64])
            fc_1 = tf.nn.relu(tf.matmul(conv_3, self.weights['w1']) + self.biases['b1'])
            self.output = tf.matmul(fc_1, self.weights['w2']) + self.biases['b2']

            self.loss = tf.losses.mean_squared_error(self.Y, self.output)
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



    def update(self, X, Y):
        _, loss = self.sess.run([self.train, self.loss], feed_dict={self.X: X, self.Y: Y})
        return loss

    def predict(self, X):
        output = self.sess.run(self.output, feed_dict={self.X: X})
        return output

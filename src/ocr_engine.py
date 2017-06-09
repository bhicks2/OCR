from splitter import LineSplitter, WordSplitter, LetterSplitter
import numpy as np
import tensorflow as tf
import os

HISTOGRAM_OF_ORIENTED_GRADIENTS = "hog"
SHAPE_FEATURES = "shape"

DEFAULT_TRAINING_DIR = "../resources/training_examples"

class OCR_Engine(object):

    def __init__(self, technique = "hog", training = None, args = None):
        if technique == HISTOGRAM_OF_ORIENTED_GRADIENTS:
            if training is None:
                training = DEFAULT_TRAINING_DIR
            if args is None:
                self.engine = HoG_OCR_Engine(training)

        elif technique == SHAPE_FEATURES:
            pass
        else:
            raise ValueError()

class HoG_OCR_Engine(object):

    def __init__(self, training, num_cells = 9, num_samples = 25, num_hidden_layers=0, hidden_state_size = 500, name = None):

        self.num_cells = num_cells
        self.num_samples = num_samples
        self.num_hidden_layers = num_hidden_layers
        self.hidden_state_size = hidden_state_size

        self.__construct_tf_graph__()

        self.sess = tf.Session()

        with self.sess as sess:
            sess.run(tf.global_variables_initializer())

            if not training.endswith(".ckpt"):
                if not os.path.isdir(training):
                    raise ValueError()

                    self.__train__(training, name)

            else:
                saver = tf.train.Saver()
                saver.restore(self.sess, training)

    def __construct_tf_graph__(self):

        self.__define_placeholders__()
        self.__define_graph__()
        self.__define_loss__()
        self.__define_training_op__()

    def __define_placeholders__(self):
        # this is the placeholder for the
        # features that we will pass into the
        # tensorflow graph. The "None" argument
        # as the first part of the shape
        # allows us to pass in arbitrarily sized
        # blocks
        feature_shape = [None, self.num_samples ** 2 * self.num_cells]
        self.features = tf.placeholder(name = "features", shape = feature_shape, dtype = tf.float32)

        labels_shape = [None]
        self.labels = tf.placeholder(name = "labels", shape = labels_shape, dtype = tf.int32)

    def __define_graph__(self):

        if self.num_hidden_layers == 0:
            W_shape = [self.num_samples ** 2 * self.num_cells, 56]
            b_shape = [56]
        else:
            W_shape = [self.num_samples ** 2 * self.num_cells, self.hidden_state_size]
            b_shape = [self.hidden_state_size]

        W = tf.get_variable(name = "W", shape = W_shape, initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = "b", shape = b_shape, initializer = tf.contrib.layers.xavier_initializer())

        xW = tf.matmul(self.features, W)
        xWb = tf.add(xW, b)

        if self.num_hidden_layers == 0:
            self.result = tf.nn.softmax(xWb)
            self.unnormed_result = xWb
        else:
            z = tf.tanh(xWb)

            V_shape = [self.hidden_state_size, self.hidden_state_size]
            c_shape = [self.hidden_state_size]

            for i in range(1, self.num_hidden_layers):
                with tf.variable_scope("hidden_layer_{}".format(i)):
                    V = tf.get_variable(name = "V", shape = V_shape, initializer = tf.contrib.layers.xavier_initializer())
                    c = tf.get_variable(name = "c", shape = V_shape, initializer = tf.contrib.layers.xavier_initializer())

                    zV = tf.matmul(z, V)
                    zVc = tf.add(zV, c)

                    z = tf.tanh(zVc)

            U_shape = [self.hidden_state_size, 56]
            d_shape = [56]

            U = tf.get_variable(name = "U", shape = U_shape, initializer = tf.contrib.layers.xavier_initializer())
            d = tf.get_variable(name = "d", shape = d_shape, initializer = tf.contrib.layers.xavier_initializer())

            zU = tf.matmul(z, U)
            zUd = tf.add(zU, d)

            self.result = tf.nn.softmax(zUd)
            self.unnormed_result = zUd

    def __define_loss__(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.unnormed_result, labels = self.labels)
        self.loss = tf.reduce_mean(loss)

    def __define_training_op__(self):
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

    def __train__(self, directory, name):
        


OCR_Engine()

from splitter import LineSplitter, WordSplitter, LetterSplitter
from image import Image
import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle


HISTOGRAM_OF_ORIENTED_GRADIENTS = "hog"
SHAPE_FEATURES = "shape"

DEFAULT_TRAINING_DIR = "../resources/training_examples"
DEFAULT_TESTING_DIR = "../resources/testing_data"
SAVED_DATA_DIR = "../resources/saved_data"
CHECKPOINT_DIR = "../resources/checkpoints"

class HoG_OCR_Engine(object):

    def __init__(self, training, name, num_cells = 9, sample_width = 4, num_hidden_layers=2, hidden_state_size = 500, load = False, verbose = False):

        self.num_cells = num_cells
        self.sample_width = sample_width
        self.num_hidden_layers = num_hidden_layers
        self.hidden_state_size = hidden_state_size

        self.verbose = verbose

        checkpoint_path = None
        if load:
            checkpoint_path = "model_hl{}_hs{}_nc{}_sw{}".format(self.num_hidden_layers, self.hidden_state_size, self.num_cells, self.sample_width)
            if not os.path.exists(CHECKPOINT_DIR + "/" + checkpoint_path + ".meta"):
                checkpoint_path = None

        self.__construct_tf_graph__()

        self.sess = tf.Session()

        if checkpoint_path is None:
            self.sess.run(tf.global_variables_initializer())
            self.__train__(self.sess, training, name)

        else:
            saver = tf.train.import_meta_graph(CHECKPOINT_DIR + "/" + checkpoint_path + ".meta")
            saver.restore(self.sess, tf.train.latest_checkpoint(CHECKPOINT_DIR + "/"))

    def close(self):
        self.sess.close()

    def predict(self, image):

        resized = image.resize(102, 102)

        features = []

        i_start = 1
        while i_start + self.sample_width < resized.height:
            i_end = i_start + self.sample_width
            j_start = 1

            while j_start + self.sample_width < resized.width:
                j_end = j_start + self.sample_width

                region = (i_start, j_start, i_end, j_end)
                hog = resized.calculate_hog(region, self.num_cells)

                features.append(hog)

                j_start += self.sample_width

            i_start += self.sample_width

        features = np.array(features).flatten()
        features_size = np.linalg.norm(features)
        features = features / features_size

        features = np.reshape(features, [1, features.shape[0]])

        feed_dict = {}
        feed_dict[self.features] = features

        probabilities = self.sess.run(self.result, feed_dict = feed_dict)
        index = np.argmax(probabilities)

        guess = self.mappings[index]
        if guess == "non-letter":
            return ''
        else:
            return guess

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
        feature_shape = [None, (100 / self.sample_width) ** 2 * self.num_cells]
        self.features = tf.placeholder(name = "features", shape = feature_shape, dtype = tf.float32)

        labels_shape = [None]
        self.labels = tf.placeholder(name = "labels", shape = labels_shape, dtype = tf.int32)

    def __define_graph__(self):

        if self.num_hidden_layers == 0:
            W_shape = [(100 / self.sample_width) ** 2 * self.num_cells, 43]
            b_shape = [43]
        else:
            W_shape = [(100 / self.sample_width) ** 2 * self.num_cells, self.hidden_state_size]
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
                    c = tf.get_variable(name = "c", shape = c_shape, initializer = tf.contrib.layers.xavier_initializer())

                    zV = tf.matmul(z, V)
                    zVc = tf.add(zV, c)

                    z = tf.tanh(zVc)

            U_shape = [self.hidden_state_size, 43]
            d_shape = [43]

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

    def __train__(self, sess, directory, name):

        data_file_name = SAVED_DATA_DIR + "/{}_data.npy".format(name)
        label_file_name = SAVED_DATA_DIR + "/{}_labels.npy".format(name)
        mapping_file_name = SAVED_DATA_DIR + "/{}_mappings".format(name)

        if os.path.exists(data_file_name):
            data = np.load(data_file_name)
            labels = np.load(label_file_name)
            self.mappings = self.__read_map__(mapping_file_name)
        else:
            data, labels, self.mappings = self.__read_data__(directory)

            np.save(data_file_name, data)
            np.save(label_file_name, labels)
            self.__write_map__(mapping_file_name)

        training_data = data
        training_labels = labels

        for i in range(0, 5):

            batches = self.__build_batches__(training_data, training_labels, 10)

            if self.verbose:
                print "Epoch {} of 5".format(i + 1)

            average_loss = 0.0
            num_batches = 0
            for batch, b_data, b_labels in batches:
                feed_dict = {}

                feed_dict[self.features] = b_data
                feed_dict[self.labels] = b_labels

                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                average_loss += loss
                num_batches += 1

            if self.verbose:
                print "\tTraining Loss: {}".format(average_loss / num_batches)

        saver = tf.train.Saver()
        saver.save(sess, CHECKPOINT_DIR + "/model_hl{}_hs{}_nc{}_sw{}".format(self.num_hidden_layers, self.hidden_state_size, self.num_cells, self.sample_width))

    def __build_batches__(self, data, labels, batch_size, isShuffle = True):
        if isShuffle:
            data, labels = shuffle(data, labels)

        current_batch_start = 0
        batch_count = 0
        while current_batch_start < labels.shape[0]:
            end_index = np.min([current_batch_start + batch_size, labels.shape[0]])

            data_batch = data[current_batch_start : end_index, :]
            labels_batch = labels[current_batch_start : end_index]

            batch_count += 1
            current_batch_start = end_index

            yield batch_count - 1, data_batch, labels_batch

    def __read_data__(self, directory):

        class_count = 1
        mappings = {0: "non-letter"}

        data = []
        labels = []

        for folder in os.listdir(directory):
            if folder.startswith('.'):
                continue

            character = folder[9]

            print "Reading data from {} examples".format(character)

            mappings[class_count] = character
            index = class_count
            class_count += 1

            for example in os.listdir(directory + "/" + folder):
                if not example.endswith(".png"):
                    continue

                image_path = directory + "/" + folder + "/" + example

                image = Image(image_path)
                image.resize(102, 102, in_place = True)

                features = []

                i_start = 1
                while i_start + self.sample_width < image.height:
                    i_end = i_start + self.sample_width
                    j_start = 1

                    while j_start + self.sample_width < image.width:
                        j_end = j_start + self.sample_width

                        region = (i_start, j_start, i_end, j_end)
                        hog = image.calculate_hog(region, self.num_cells)

                        features.append(hog)

                        j_start += self.sample_width

                    i_start += self.sample_width

                features = np.array(features).flatten()
                features_size = np.linalg.norm(features)
                features = features / features_size

                data.append(features)
                labels.append(index)

        return np.array(data), np.array(labels), mappings

    def __read_map__(self, path):
        mappings = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                key, value = line.split("=")
                key = int(key)
                mappings[key] = value.rstrip()

        return mappings

    def __write_map__(self, path):
        with open(path, 'w') as file:
            for key in self.mappings:
                file.write(str(key) + "=" + str(self.mappings[key]) + "\n")

class Pixel_OCR_Engine(object):

    def __init__(self, training, name, num_hidden_layers=2, hidden_state_size = 500, load = False, verbose = False):

        self.name = name

        self.num_hidden_layers = num_hidden_layers
        self.hidden_state_size = hidden_state_size

        self.verbose = verbose

        checkpoint_path = None
        if load:
            checkpoint_path = "model_hl{}_hs{}".format(self.num_hidden_layers, self.hidden_state_size)
            if not os.path.exists(CHECKPOINT_DIR + "/" + checkpoint_path + ".meta"):
                checkpoint_path = None

        self.__construct_tf_graph__()

        self.sess = tf.Session()

        if checkpoint_path is None:
            self.sess.run(tf.global_variables_initializer())
            self.__train__(self.sess, training, name)

        else:
            saver = tf.train.import_meta_graph(CHECKPOINT_DIR + "/" + checkpoint_path + ".meta")
            saver.restore(self.sess, tf.train.latest_checkpoint(CHECKPOINT_DIR + "/"))

    def close(self):
        self.sess.close()

    def predict(self, image):

        resized = image.resize(102, 102)

        features = resized.image[:, :, 0].flatten()
        features_size = np.linalg.norm(features)
        features = features / features_size

        features = np.reshape(features, [1, features.shape[0]])

        feed_dict = {}
        feed_dict[self.features] = features

        probabilities = self.sess.run(self.result, feed_dict = feed_dict)
        index = np.argmax(probabilities)

        guess = self.mappings[index]
        if guess == "non-letter":
            return ''
        else:
            return guess

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
        feature_shape = [None, 102 ** 2]
        self.features = tf.placeholder(name = "features", shape = feature_shape, dtype = tf.float32)

        labels_shape = [None]
        self.labels = tf.placeholder(name = "labels", shape = labels_shape, dtype = tf.int32)

    def __define_graph__(self):

        if self.num_hidden_layers == 0:
            W_shape = [102 ** 2, 43]
            b_shape = [43]
        else:
            W_shape = [102 ** 2, self.hidden_state_size]
            b_shape = [self.hidden_state_size]

        W = tf.get_variable(name = "{}_W".format(self.name), shape = W_shape, initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = "{}_b".format(self.name), shape = b_shape, initializer = tf.contrib.layers.xavier_initializer())

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
                with tf.variable_scope("{}hidden_layer_{}".format(self.name, i)):
                    V = tf.get_variable(name = "V", shape = V_shape, initializer = tf.contrib.layers.xavier_initializer())
                    c = tf.get_variable(name = "c", shape = c_shape, initializer = tf.contrib.layers.xavier_initializer())

                    zV = tf.matmul(z, V)
                    zVc = tf.add(zV, c)

                    z = tf.tanh(zVc)

            U_shape = [self.hidden_state_size, 43]
            d_shape = [43]

            U = tf.get_variable(name = "{}_U".format(self.name), shape = U_shape, initializer = tf.contrib.layers.xavier_initializer())
            d = tf.get_variable(name = "{}_d".format(self.name), shape = d_shape, initializer = tf.contrib.layers.xavier_initializer())

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

    def __train__(self, sess, directory, name):

        data_file_name = SAVED_DATA_DIR + "/{}_data.npy".format(name)
        label_file_name = SAVED_DATA_DIR + "/{}_labels.npy".format(name)
        mapping_file_name = SAVED_DATA_DIR + "/{}_mappings".format(name)

        if os.path.exists(data_file_name):
            data = np.load(data_file_name)
            labels = np.load(label_file_name)
            self.mappings = self.__read_map__(mapping_file_name)
        else:
            data, labels, self.mappings = self.__read_data__(directory)

            np.save(data_file_name, data)
            np.save(label_file_name, labels)
            self.__write_map__(mapping_file_name)

        training_data = data
        training_labels = labels

        for i in range(0, 10):

            batches = self.__build_batches__(training_data, training_labels, 10)

            if self.verbose:
                print "Epoch {} of 10".format(i + 1)

            average_loss = 0.0
            num_batches = 0
            for batch, b_data, b_labels in batches:
                feed_dict = {}

                feed_dict[self.features] = b_data
                feed_dict[self.labels] = b_labels

                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                average_loss += loss
                num_batches += 1

            if self.verbose:
                print "\tTraining Loss: {}".format(average_loss / num_batches)

        saver = tf.train.Saver()
        saver.save(sess, CHECKPOINT_DIR + "/model_hl{}_hs{}".format(self.num_hidden_layers, self.hidden_state_size))

    def __build_batches__(self, data, labels, batch_size, isShuffle = True):
        if isShuffle:
            data, labels = shuffle(data, labels)

        current_batch_start = 0
        batch_count = 0
        while current_batch_start < labels.shape[0]:
            end_index = np.min([current_batch_start + batch_size, labels.shape[0]])

            data_batch = data[current_batch_start : end_index, :]
            labels_batch = labels[current_batch_start : end_index]

            batch_count += 1
            current_batch_start = end_index

            yield batch_count - 1, data_batch, labels_batch

    def __read_data__(self, directory):

        class_count = 1
        mappings = {0: "non-letter"}

        data = []
        labels = []

        for folder in os.listdir(directory):
            if folder.startswith('.'):
                continue

            character = folder[9]

            print "Reading data from {} examples".format(character)

            mappings[class_count] = character
            index = class_count
            class_count += 1

            for example in os.listdir(directory + "/" + folder):
                if not example.endswith(".png"):
                    continue

                image_path = directory + "/" + folder + "/" + example

                image = Image(image_path)
                resized = image.resize(102, 102)

                features = resized.image[:, :, 0].flatten()

                features_size = np.linalg.norm(features)
                features = features / features_size

                data.append(features)
                labels.append(index)

        return np.array(data), np.array(labels), mappings

    def __read_map__(self, path):
        mappings = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                key, value = line.split("=")
                key = int(key)
                mappings[key] = value.rstrip()

        return mappings

    def __write_map__(self, path):
        with open(path, 'w') as file:
            for key in self.mappings:
                file.write(str(key) + "=" + str(self.mappings[key]) + "\n")

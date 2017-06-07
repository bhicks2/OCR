# import tensorflow as tf
from image import Image
from image import *
from textimage import TextImage
import os
import sklearn.utils as sku

class Baseline(object):

    def train(self, directory, binarize=False):
        self.readTrainingData(directory, binarize)
        self.buildTensorflowGraph()

        print self.classMapping

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            self.data, self.classes = sku.shuffle(self.data, self.classes)

            print self.classes

            train_data = self.data[0:-100, :]
            train_labels = self.classes[0:-100]

            valid_data = self.data[-100:, :]
            valid_labels = self.classes[-100:]

            for epoch in range(0, 10):
                print "Epoch {} starting...".format(epoch + 1)

                train_data, train_labels = sku.shuffle(train_data, train_labels)

                averageLoss = 0.0
                for i in range(train_data.shape[0]):
                    currData = train_data[i:i+1, :]
                    currLabels = train_labels[i:i+1]

                    feed = {self.input: currData, self.input_labels: currLabels}
                    _, loss,  probability, guesses, W, b= session.run([self.train_op, self.loss, self.probability, self.guesses, self.W, self.b1], feed)
                    averageLoss += loss


                print "\tEpoch loss: {}".format(averageLoss/train_data.shape[0])

            feed = {self.input: valid_data, self.input_labels: valid_labels}
            W, probs, loss, guesses = session.run([self.W, self.probability, self.loss, self.guesses], feed)
            print loss
            print "Validation loss: {}".format(loss)

            wrong = 0.0
            for i in range(guesses.shape[0]):
                if guesses[i] != valid_labels[i]:
                    wrong += 1.0

            print "Accuracy: {}%".format(100*(100.0-wrong)/100.0)

            a = W[:, 0:1]
            b = W[:, 15:16]
            c = W[:, 2:3]
            d = W[:, 15:16]

            aMax = np.amax(a)
            aMin = np.amin(a)

            a = 255 * ((a - aMin) / (aMax - aMin))
            a = np.reshape(a, (100, 100))
            a = np.dstack([a, a, a])
            aImage = Image(a)
            aImage.displayImage()

            bMax = np.amax(b)
            bMin = np.amin(b)

            b = 255 * ((b - bMin) / (bMax - bMin))
            b = np.reshape(b, (100, 100))
            b = np.dstack([b, b, b])
            bImage = Image(b)
            bImage.displayImage()



    def buildTensorflowGraph(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 100*100], name="input")
        self.input_labels = tf.placeholder(shape=[None], dtype = tf.int32, name="labels")
        self.W = tf.get_variable("W", [100*100, 27], initializer=tf.contrib.layers.xavier_initializer())
        #self.U = tf.get_variable("U", [27, 27], initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable("b1", [27], initializer=tf.contrib.layers.xavier_initializer())
        #self.b2 = tf.get_variable("b2", [27], initializer=tf.contrib.layers.xavier_initializer())

        xW = tf.matmul(self.input, self.W)
        xWb1 = tf.add(xW, self.b1)
        #f = tf.tanh(xWb1)

        #fU = tf.matmul(f, self.U)
        #fUb2 = tf.add(fU, self.b2)

        #self.probability = tf.nn.softmax(fUb2)
        self.probability = tf.nn.softmax(xWb1)

        #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fUb2, labels = self.input_labels))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = xWb1, labels = self.input_labels))
        self.guesses = tf.arg_max(self.probability, 1)

        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        self.train_op = optimizer.minimize(self.loss)

    def readTrainingData(self, directory, binarize):

        print "Reading training data..."

        data = []
        dataClass = []
        dataClassMapping = {}
        dataClassMappingReversed = {}
        numClasses = 0

        for filename in os.listdir(directory):
            # process data
            if not (filename.endswith(".png") or filename.endswith(".jpg")):
                continue
            image = readImage(directory + "/" + filename)
            textImage = TextImage(image)

            #textImage.crop_to_contents()
            textImage.resize(100, 100)

            pixels = textImage.getFlattenedPixels()/255
            if binarize:
                pixels = (pixels < 0.01).astype(float)
            else:
                pixels = 1 - pixels
            data.append(pixels)

            underscoreIndex = filename.find("_")
            if underscoreIndex == -1:
                underscoreIndex = filename.find(".png")

            datumClassString = filename[0:underscoreIndex]

            if datumClassString not in dataClassMapping:
                dataClassMapping[datumClassString] = numClasses
                dataClassMappingReversed[numClasses] = datumClassString
                numClasses += 1

            datumClass = dataClassMapping[datumClassString]
            dataClass.append(datumClass)

        self.data = np.array(data)
        self.classes = np.array(dataClass)
        self.classMapping = dataClassMapping
        self.classMappingReverse = dataClassMappingReversed

        print "Done reading training data."

if __name__ == '__main__':
    baseline = Baseline()
    baseline.train("../resources/training_examples", binarize = True)

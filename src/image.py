import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


RED_INDEX = 0
GREEN_INDEX = 1
BLUE_INDEX = 2

RED_LUMA = 0.299
GREEN_LUMA = 0.587
BLUE_LUMA = 0.114

def readImage(filename):
    imageArray = misc.imread(filename)
    return Image(imageArray[:, :, RED_INDEX:BLUE_INDEX + 1])

class Image(object):

    def __init__(self, imageArray):
        self.image = imageArray

    def getHeight(self):
        return self.image.shape[1]

    def getWidth(self):
        return self.image.shape[0]

    def getRGB(self, x, y):
        red = self.image[x, y, RED_INDEX]
        green = self.image[x, y, GREEN_INDEX]
        blue = self.image[x, y, BLUE_INDEX]

        return (red, green, blue)

    def getGrayscaleImage(self):
        weightMatrix = np.array([RED_LUMA, GREEN_LUMA, BLUE_LUMA])
        weightMatrix = np.reshape(weightMatrix, [1, 1, weightMatrix.shape[0]])

        weightedImage = self.image * weightMatrix
        grayscale = np.sum(weightedImage, axis=2)

        grayscale = np.dstack((grayscale, grayscale, grayscale))
        return Image(grayscale)

    def getGrayscaleHistogram(self, axis=1):
        grayscaleImage = self.getGrayscaleImage()
        return np.mean(grayscaleImage.image[:, :, RED_INDEX], axis = axis)

    def resize(self, width, height):
        newImage = misc.imresize(self.image, (height, width))
        return Image(newImage)

    def saveImage(self, filename):
        image = misc.toimage(self.image, cmin=0, cmax=255, mode="RGB", channel_axis=2)
        image.save(filename)

    def displayImage(self):
        # rescale
        decArray = self.image/255
        plt.imshow(decArray)
        plt.show()

    def extractSubImage(self, x, y, width, height):
        subimage = self.image[y:y+height, x:x+width, :]
        return Image(subimage)

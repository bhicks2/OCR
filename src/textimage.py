from image import *
from image import Image

class TextImage(object):

    def __init__(self, image):
        self.image = image.getGrayscaleImage()

    def crop_to_contents(self):
        # crop vertically (i.e., make shorter)
        start, end = self.__identifyBounds__(1)
        self.image = self.image.extractSubImage(0, start, self.image.getWidth(), end - start)

        # crop horizontally (i.e., make narrower)
        start, end = self.__identifyBounds__(0)
        self.image = self.image.extractSubImage(start, 0, end - start, self.image.getHeight())

    def resize(self, width, height):
        self.image = self.image.resize(width, height)

    def getFlattenedPixels(self):
        return np.ndarray.flatten(self.image.image[:, :, 0])

    def __identifyBounds__(self, axis, threshold = 255):
        histogram = self.image.getGrayscaleHistogram(axis)

        print histogram

        # find start of the contents
        last = None
        start = 0
        for i in range(0, histogram.shape[0]):
            curr = histogram[i]
            if curr < threshold and last >= threshold:
                start = i
                break
            last = curr

        last = None
        end = histogram.shape[0]
        for j in xrange(histogram.shape[0] - 1, -1, -1):
            curr = histogram[j]

            if curr < threshold and last >= threshold:
                end = j
                break
            last = curr

        return (start, end)

    def extractLines(self, threshold=None):
        histogram = self.getImageData(1)

        last = None
        start = None
        end = None
        ranges = []

        if threshold is None:
            threshold = 0.0


        for i in range(0, histogram.shape[0]):
            curr = histogram[i]
            if curr < threshold:
                if last >= threshold:
                    start = i
            if curr >= threshold:
                if last < threshold:
                    end = i
            if start and end:
                ranges.append((start, end))
                start = None
                end = None

            last = curr

        images = []
        for span in ranges:
            images.append(self.image.extractSubImage(0, span[0], self.image.getWidth(), span[1] - span[0]))
        return images

    def getImageData(self, axis, normalize=True):
        histogram = self.image.getGrayscaleHistogram()

        if normalize:
            mean = np.mean(histogram)
            std = np.std(histogram)

            histogram = (histogram - mean)/std

        return histogram

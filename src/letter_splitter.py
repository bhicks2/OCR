import numpy as np
from image import Image

class LetterSplitter(object):

    def __init__(self, threshold = 1, binarize_threshold = None):
        self.threshold = threshold

    def split_letters(self, image):
        binarized_image = image.binarize()
        image_array = image.image

        histogram = np.min(image_array[:, :, 0], axis = 0)

        average_intensity = np.mean(histogram)
        standard_deviation = np.std(histogram)

        normalized_histogram = (histogram - average_intensity) / standard_deviation

        is_break = normalized_histogram > self.threshold

        last = True
        start = None
        end = None
        letters = []
        for i in range(len(is_break)):
            if not is_break[i] and last:
                start = i
            if is_break[i] and not last:
                end = i

            if start is not None and i == len(is_break) - 1:
                end = i + 1

            last = is_break[i]

            if start is not None and end is not None:
                new_width = end - start
                letter = image.crop(0, start, image.height, new_width)
                letters.append(letter)
                start = None
                end = None

        return letters

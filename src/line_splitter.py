import numpy as np
from image import Image

class LineSplitter(object):

    # The threshold is the z-score that is considered
    # to be a sign that a line is a line-break, and
    # not a text-line
    def __init__(self, threshold = 0, min_height = None, restrict_by_height = False):
        self.threshold = threshold
        self.min_height = min_height

        if min_height is not None or restrict_by_height:
            self.restrict_by_height = True
        else:
            self.restrict_by_height = False

    # Given an image, splits it into its component
    # lines. The return value is a set of images
    # that are considered to be text-lines
    def split_lines(self, image):
        grayscale = image.convert_to_grayscale()
        image_array = grayscale.image[:, :, 0]

        histogram = np.min(image_array, axis = 1)

        average_intensity = np.mean(histogram)
        standard_deviation = np.std(histogram)

        normalized_histogram = (histogram - average_intensity) / standard_deviation

        is_break = normalized_histogram > self.threshold

        conditional_lines = []
        line_height = []

        last = True
        start = None
        end = None
        for i in range(is_break.shape[0]):
            if is_break[i] and not last:
                end = i
            if not is_break[i] and last:
                start = i

            last = is_break[i]

            if start is not None and end is not None:
                line = grayscale.crop(start, 0, end - start, image.width)

                line_height.append(line.height)
                conditional_lines.append(line)

                start = None
                end = None

        lines = []

        if self.min_height is None:
            self.min_height = np.mean(line_height)

        if self.restrict_by_height:
            for i in range(0, len(line_height)):
                if line_height[i] >= self.min_height:
                    lines.append(conditional_lines[i])

        return lines

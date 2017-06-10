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

class WordSplitter(object):

    def __init__(self, threshold = 0, min_space = None):
        self.threshold = threshold
        self.min_space = min_space


    def split_words(self, image):
        grayscale = image.convert_to_grayscale()
        image_array = grayscale.image[:, :, 0]

        histogram = np.min(image_array, axis = 0)

        average_intensity = np.mean(histogram)
        standard_deviation = np.std(histogram)

        normalized_histogram = (histogram - average_intensity) / standard_deviation

        is_break = normalized_histogram > self.threshold

        break_widths = []

        text_image = grayscale

        start_of_text = None
        for i in range(len(is_break)):
            if not is_break[i]:
                start_of_text = i
                break

        end_of_text = None
        for i in range(is_break.shape[0] - 1, -1, -1):
            if not is_break[i]:
                end_of_text = i + 1
                break

        if start_of_text is None or end_of_text is None:
            return []

        if start_of_text == end_of_text:
            return []

        text_image = grayscale.crop(0, start_of_text, grayscale.height, end_of_text - start_of_text)
        is_break = is_break[start_of_text : end_of_text]

        running_width = 0
        for i in range(len(is_break)):
            if is_break[i]:
                running_width += 1
            if i == len(is_break) - 1 or not is_break[i] and running_width is not 0:
                break_widths.append(running_width)
                running_width = 0

        average_width = np.mean(break_widths)

        word_bounds = []

        last = True
        start = None
        end = None
        for i in range(len(is_break)):
            if not is_break[i] and last:
                start = i
            if is_break[i] and not last:
                end = i

            if start is not None and i == len(is_break) - 1:
                end = i + 1

            if start is not None and end is not None:
                word_bounds.append((start, end))
                start = None
                end = None

            last = is_break[i]

        if self.min_space is None:
            min_space = average_width
        else:
            min_space = self.min_space

        words = []
        current_bounds = word_bounds[0]
        for i in range(1, len(word_bounds)):
            check_bounds = word_bounds[i]

            if check_bounds[0] - current_bounds[1] < min_space:
                current_bounds = (current_bounds[0], check_bounds[1])
            else:
                start, end = current_bounds
                new_width = end - start
                word = text_image.crop(0, start, text_image.height, new_width)
                words.append(word)
                current_bounds = check_bounds

        start, end = current_bounds
        new_width = end - start
        word = text_image.crop(0, start, text_image.height, new_width)
        words.append(word)

        return words


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

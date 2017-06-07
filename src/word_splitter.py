import numpy as np
from image import Image
from line_splitter import LineSplitter
from letter_splitter import LetterSplitter

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
        text_image.display_image()
        is_break = is_break[start_of_text : end_of_text]

        running_width = 0
        for i in range(len(is_break)):
            if is_break[i]:
                running_width += 1
            if i == len(is_break) - 1 or not is_break[i] and running_width is not 0:
                break_widths.append(running_width)
                running_width = 0

        average_width = np.mean(break_widths)
        print average_width

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

            print "{} / {}".format(current_bounds, check_bounds)

            if check_bounds[0] - current_bounds[1] < min_space:
                print "...coalesce"
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




line_splitter = LineSplitter(min_height=10, restrict_by_height = True)
word_splitter = WordSplitter()
letter_splitter = LetterSplitter()
image = Image("../resources/training_examples_lines/processed/training_page_0001/noisy_100.jpg")
clean = image.denoise(0).normalize()

lines = line_splitter.split_lines(clean)
print "Found {} lines in the image".format(len(lines))

for line in lines:
    for word in word_splitter.split_words(line):
        word.display_image()
        for letter in letter_splitter.split_letters(word):
            letter.display_image()

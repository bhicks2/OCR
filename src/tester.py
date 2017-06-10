from image import Image
import os
import subprocess
from splitter import LineSplitter, WordSplitter, LetterSplitter
import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')

MAPPINGS = {
    '' : 0,'a' : 1,'b' : 2,'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8,
    'i' : 9, 'j' : 10, 'k' : 11, 'l' : 12, 'm' : 13, 'n' : 14, 'o' : 15, 'p' : 16,
    'q' : 17, 'r' : 18, 's' : 19, 't' : 20, 'u' : 21, 'v' : 22, 'w' : 23, 'x' : 24,
    'y' : 25, 'z' : 26, '0' : 27, '1' : 28, '2' : 29, '3' : 30, '4' : 31, '5' : 32,
    '6' : 33, '7' : 34, '8' : 35, '9' : 36, '-' : 37, ',' : 38, ';' : 39, '!' : 40,
    '.' : 41, '*' : 42
}

CLASS_MAPPINGS = ['' for i in MAPPINGS]

for key in MAPPINGS:
    index = MAPPINGS[key]
    CLASS_MAPPINGS[index] = key

class TextTester(object):

    def __init__(self, directory, engine, output):
        self.directory = directory
        self.engine = engine

        self.output = output

        self.line_splitter = LineSplitter(min_height = 10)
        self.word_splitter = WordSplitter()
        self.character_splitter = LetterSplitter()

    def test(self, verbose = False):

        average_edit_distance = 0
        average_large_edit_distance = 0
        average_medium_edit_distance = 0
        average_small_edit_distance = 0

        num_tested = 0
        num_large_tested = 0
        num_medium_tested = 0
        num_small_tested = 0

        output_path = self.directory + "/results/" + self.output
        output = open(output_path, 'w')

        for folder in os.listdir(self.directory):

            if folder.startswith("."):
                continue

            average_text_edit_distance = 0
            num_text_tested = 0

            if folder == "results":
                continue

            folder_path = self.directory + "/" + folder

            lines = None
            with open("{}/{}/text.txt".format(self.directory, folder)) as answer_key:
                lines = [line for line in answer_key.readlines()]

            key = " ".join(lines)

            for test_case in os.listdir(folder_path):
                if not test_case.endswith(".png"):
                    continue

                full_image_path = self.directory + "/" + folder + "/" + test_case

                image = Image(full_image_path)

                image.normalize(in_place = True)
                image.denoise(0, in_place = True)

                lines = self.line_splitter.split_lines(image)

                line_guesses = []
                for line in lines:
                    words = self.word_splitter.split_words(line)

                    word_guesses = []
                    for word in words:
                        characters = self.character_splitter.split_letters(word)

                        character_guesses = []
                        for character in characters:

                            character = character.binarize()
                            character = character.crop_to_contents(2)

                            guess = self.engine.predict(character)
                            character_guesses.append(guess)

                        guess = "".join(character_guesses)
                        word_guesses.append(guess)

                    guess = " ".join(word_guesses)
                    line_guesses.append(guess)

                guess = " ".join(line_guesses)

                distance = self.calculate_distance(key, guess) / float(len(key))

                if verbose:
                    result = "My guess: {}\n\t {}: {}".format(guess, test_case, distance)
                    print result
                    output.write(result + "\n")

                average_text_edit_distance += distance
                num_text_tested += 1

                if test_case[10] == "0":
                    average_small_edit_distance += distance
                    num_small_tested += 1
                if test_case[10] == "1":
                    average_medium_edit_distance += distance
                    num_medium_tested += 1
                if test_case[10] == "2":
                    average_large_edit_distance += distance
                    num_large_tested += 1

                average_edit_distance += distance
                num_tested += 1

            average_text_edit_distance = float(average_text_edit_distance) / num_text_tested


            if verbose:
                text_result = "{}: {}".format(folder, average_text_edit_distance)
                print text_result
                output.write(text_result)

        average_edit_distance = float(average_edit_distance) / num_tested
        average_large_edit_distance = float(average_large_edit_distance) / num_large_tested
        average_medium_edit_distance = float(average_medium_edit_distance) / num_medium_tested
        average_small_edit_distance = float(average_small_edit_distance) / num_small_tested

        full_result = "Average distance: {}\n\tSmall: {}\tMedium: {}\tLarge: {}".format(average_edit_distance, average_small_edit_distance, average_medium_edit_distance, average_large_edit_distance)
        print full_result
        output.write(full_result)




    def calculate_distance(self, gold, guess):
        edit_distance = np.zeros(shape = [len(gold) + 1, len(guess) + 1])

        for i in range(len(gold) + 1):
            edit_distance[i, 0] = i

        for j in range(len(guess) + 1):
            edit_distance[0, j] = j

        for i in range(1, len(gold) + 1):
            for j in range(1, len(guess) + 1):
                curr_score = 0 if gold[i - 1] == guess[j - 1] else 2

                above = edit_distance[i - 1, j] + 1
                left = edit_distance[i, j - 1] + 1
                diag = edit_distance[i - 1, j - 1] + curr_score

                edit_distance[i, j] = np.min([above, left, diag])

        i_max = len(gold)
        j_max = len(guess)
        return edit_distance[i_max, j_max]

class LetterTester(object):
    def __init__(self, directory, engine, output):
        self.directory = directory
        self.engine = engine

        self.output = output

    def test(self, verbose = False):

        output_path = self.directory + "/results/" + self.output
        output = open(output_path, 'w')

        num_correct = 0
        num_tested = 0

        confusion = np.zeros([len(MAPPINGS), len(MAPPINGS)])

        for folder in os.listdir(self.directory):

            num_correct_char = 0
            num_tested_char = 0

            if folder.startswith('.'):
                continue

            if folder == "results":
                continue

            character = folder[8]

            if character == '_':
                character = ''

            folder_path = self.directory + "/" + folder

            for image in os.listdir(folder_path):
                if not image.endswith(".png"):
                    continue

                full_image_path = folder_path + "/" + image

                image = Image(full_image_path)

                image = image.denoise(0)
                image = image.normalize()

                guess = self.engine.predict(image)

                if guess == character:
                    num_correct += 1
                    num_correct_char += 1

                num_tested += 1
                num_tested_char += 1

                correct_index = MAPPINGS[character]
                guessed_index = MAPPINGS[guess]

                confusion[correct_index, guessed_index] += 1

            if verbose:
                char_result = "Character '{}': {} / {} = {}%".format(character, num_correct_char, num_tested_char, 100 * float(num_correct_char) / num_tested_char)
                print char_result
                output.write(char_result + "\n")

        overall_result = "Overall: {} / {} = {}%".format(num_correct, num_tested, 100 * float(num_correct) / num_tested)
        print overall_result
        output.write(overall_result + "\n")

        output.close()


        plt.matshow(confusion)
        plt.colorbar()

        tick_marks = np.arange(len(MAPPINGS))

        plt.xticks(tick_marks, CLASS_MAPPINGS)
        plt.yticks(tick_marks, CLASS_MAPPINGS)

        plt.show()

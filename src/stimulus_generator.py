from image import Image
import os
import subprocess
import numpy as np

CHARACTERS_TO_GENERATE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
'.', ',', '!', ';', '-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ', ' ', ' ', ' ', ' ', ' ']

FONTS_TO_MAKE = ['American Typewriter', 'Andale Mono', 'Arial Black', 'Arial Narrow',
'Arial Rounded MT Bold', 'Arial Unicode MS', 'Avenir', 'Avenir Next', 'Avenir Next Condensed',
'Baskerville', 'Big Caslon',
'Bradley Hand', 'Brush Script MT', 'Chalkboard', 'Chalkboard SE', 'Chalkduster',
'Cochin', 'Comic Sans MS', 'Copperplate', 'Courier', 'Courier New', 'Didot', 'Futura',
'Geneva', 'Georgia', 'Gill Sans', 'Helvetica', 'Helvetica Neue', 'Herculanum', 'Hoefler Text',
'Impact', 'Lucida Grande', 'Luminari', 'Marker Felt', 'Menlo', 'Microsoft Sans Serif',
'Monaco', 'Noteworthy', 'Optima', 'Palatino', 'Papyrus', 'Phosphate', 'PT Mono',
'PT Serif', 'PT Serif Caption', 'Savoye LET', 'SignPainter', 'Skia', 'Snell Roundhand',
'STIXGeneral', 'Tahoma', 'Times', 'Times New Roman', 'Trebuchet MS',
'Verdana', 'Zapfino']

FONT_SIZES = ['normalsize', 'large', 'LARGE']

STIMULI = ["i am brian", "potato\n\nwatermelon\n\nbanana", "madrid, spain\n\nlisbon, portugal\n\nmoscow, russia\n\nberlin, germany"]

MAX_LINES = 12
MAX_LINE_LENGTH = 10

NUM_STIMULI = 30

class StimulusGenerator(object):

    def __init__(self, preamble, closing):
        self.preamble = preamble
        self.closing = closing

    # path should be a directory
    def create_stimulus(self, path, text, font, size, name):
        latex_file_path = path + "/latex.tex"
        text_file_path = path + "/text.txt"

        with open(latex_file_path, 'w') as target:
            target.write(self.preamble)

            target.write("\\setromanfont[Mapping=tex-text]{" + font + "}\n")

            target.write("\\{}\n".format(size))

            print text
            target.write(text)

            target.write(self.closing)

        with open(text_file_path, 'w') as target:
            target.write(text)

        tex_production = subprocess.Popen(['xelatex', "--interaction=batchmode", "--output-directory={}".format(path), path + "/latex.tex"])
        tex_production.communicate()

        image_conversion = subprocess.Popen(['convert', path + "/latex.pdf", path + "/stimulus_clean.png"])
        image_conversion.communicate()

        white_background = subprocess.Popen(['convert', path + "/stimulus_clean.png", "-flatten",  path + "/{}.png".format(name)])
        white_background.communicate()

        os.remove(path + "/latex.pdf")
        os.remove(path + "/latex.log")
        os.remove(path + "/latex.aux")
        os.remove(path + "/latex.tex")

preamble = "%!TEX TS-program = xelatex\n%!TEX encoding = UTF-8 Unicode\n\\documentclass[extrafontsizes, 36pt]{memoir}\n\\usepackage[margin=1in]{geometry}\n\\geometry{letterpaper}\n\\usepackage{graphicx}\n\\usepackage{amssymb}\n\\usepackage{nopageno}\n\\usepackage{fontspec,xltxtra,xunicode}\n\\defaultfontfeatures{Mapping=tex-text}\n\\setromanfont[Mapping=tex-text]{Times New Roman}\n\\begin{document}\n"
closing = "\n\\end{document}"

generator = StimulusGenerator(preamble, closing)

stimuli = []

directory = "../resources/testing_data/text"

for i in range(len(STIMULI)):

    folder = "testing_text_{}".format(i)

    if not os.path.exists(directory + "/" + folder):
        os.mkdir(directory + "/" + folder)

    stimulus = STIMULI[i]

    for j in range(len(FONTS_TO_MAKE)):
        for k in range(len(FONT_SIZES)):

            size = FONT_SIZES[k]
            font = FONTS_TO_MAKE[j]

            example_name = "example_{}_{}.png".format(j, k)

            path = directory + "/" + folder

            generator.create_stimulus(path, stimulus, font, size, example_name)

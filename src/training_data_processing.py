from image import Image
import os
import subprocess

DIRECTORY = "../resources/training_examples"

CHARACTERS_TO_GENERATE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'U', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', ',', '!', ':', ';', '-',
'1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

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

for character in CHARACTERS_TO_GENERATE:
    filename = "training_{}_example".format(character)

    full_path = DIRECTORY + "/" + filename
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    path_to_output = full_path + "/example"

    with open(path_to_output + ".tex", 'w') as file:
        file.write("%!TEX TS-program = xelatex\n")
        file.write("%!TEX encoding = UTF-8 Unicode\n")

        file.write("\\documentclass[extrafontsizes, 60pt]{memoir}\n")

        # Write packages
        file.write("\\usepackage[margin=1in]{geometry}\n")
        file.write("\\geometry{letterpaper}\n")
        file.write("\\usepackage{graphicx}\n")
        file.write("\\usepackage{amssymb}\n")
        file.write("\\usepackage{nopageno}\n")

        file.write("\\usepackage{fontspec,xltxtra,xunicode}\n")
        file.write("\\defaultfontfeatures{Mapping=tex-text}\n")

        file.write("\\setromanfont[Mapping=tex-text]{Times New Roman}\n")

        file.write("\\begin{document}\n")


        for font in FONTS_TO_MAKE:
            file.write("\\setromanfont[Mapping=tex-text]{" + font + "}\n")
            file.write("{\\Huge " + character + "}\n")
            file.write("\\newpage\n")

        file.write("\\end{document}")

        file.close()

        #os.system("pdftex {}".format(path_to_output))
        tex_production = subprocess.Popen(['xelatex', "--interaction=batchmode", "--output-directory={}".format(full_path), path_to_output + ".tex"])
        tex_production.communicate()

        print "Completed file"

        image_conversion = subprocess.Popen(['convert', path_to_output + ".pdf", path_to_output + ".png"])
        image_conversion.communicate()

        print "Completed image conversion"

        for i in range(len(FONTS_TO_MAKE)):
            image_name = full_path + "/example-{}.png".format(i)
            new_image_name = full_path + "/example_{}.png".format(i)

            white_background = subprocess.Popen(['convert', image_name, "-flatten", new_image_name])
            white_background.communicate()

            os.remove(image_name)

            image = Image(new_image_name).crop_to_contents(2)
            image.save_image(new_image_name)

        os.remove(full_path + "/example.pdf")
        os.remove(full_path + "/example.log")
        os.remove(full_path + "/example.aux")
        os.remove(full_path + "/example.tex")

from image import Image
import os

# takes in a set of training examples from the provided folder
# and generates a larger set of training examples by adding
# noise, changing brightness, and other changes

print "Please enter path:"
folder = raw_input()

original_folder = folder + "/originals"
for filename in os.listdir(original_folder + ""):
    path = original_folder + "/" + filename

    if not (filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    original = Image(path)
    print "Currently Processessing:",filename
    # search from end to make sure its actually the extension
    start_of_extension = filename[::-1].find(".")
    name = filename[0:-(start_of_extension + 1)]

    noisy_20 = original.add_noise(20)
    noisy_40 = original.add_noise(40)
    noisy_60 = original.add_noise(60)
    noisy_80 = original.add_noise(80)
    noisy_100 = original.add_noise(100)

    dark_20 = original.darken(20)
    dark_40 = original.darken(40)
    dark_60 = original.darken(60)

    bright_20 = original.brighten(20)
    bright_40 = original.brighten(40)
    bright_60 = original.brighten(60)

    processed_path = folder + "/processed/" + name
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    noisy_20.save_image(processed_path + "/" + "noisy_20.jpg")
    noisy_40.save_image(processed_path + "/" + "noisy_40.jpg")
    noisy_60.save_image(processed_path + "/" + "noisy_60.jpg")
    noisy_80.save_image(processed_path + "/" + "noisy_80.jpg")
    noisy_100.save_image(processed_path + "/" + "noisy_100.jpg")

    dark_20.save_image(processed_path + "/" + "dark_20.jpg")
    dark_40.save_image(processed_path + "/" + "dark_40.jpg")
    dark_60.save_image(processed_path + "/" + "dark_60.jpg")

    bright_20.save_image(processed_path + "/" + "bright_20.jpg")
    bright_40.save_image(processed_path + "/" + "bright_40.jpg")
    bright_60.save_image(processed_path + "/" + "bright_60.jpg")

    original.save_image(processed_path + "/original.jpg")

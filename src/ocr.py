from image import Image
from splitter import WordSplitter, LineSplitter, LetterSplitter
import ocr_engine as ocr
from ocr_engine import HoG_OCR_Engine, Pixel_OCR_Engine
from tester import TextTester, LetterTester

# Load the OCR Engine
hog_engine = Pixel_OCR_Engine(ocr.DEFAULT_TRAINING_DIR, "basic_pixel_normalized", num_hidden_layers = 2, hidden_state_size = 1000, verbose = True)
pixel_engine = HoG_OCR_Engine(ocr.DEFAULT_TRAINING_DIR, "basic_hog_normalized", num_cells = 9, sample_width = 4, num_hidden_layers = 2, hidden_state_size = 1000, verbose = True)


print "HoG Tests:"
tester = LetterTester("../resources/testing_data/letters", hog_engine, "hog_letter_test")
tester.test(verbose = True)

tester = TextTester("../resources/testing_data/text", hog_engine, "hog_text_test")
tester.test(verbose = True)

print "Pixel Tests"
tester = LetterTester("../resources/testing_data/letters", pixel_engine, "pixel_letter_test")
tester.test(verbose = True)

tester = TextTester("../resources/testing_data/text", pixel_engine, "pixel_text_test")
tester.test(verbose = True)

engine.close()

import image.Image;
import image.Pixel;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * Created by brianhicks on 5/10/17.
 */
public class ImageTest {

    private static final int RGB_TOLERANCE = 1;
    private static final int ALPHA_TOLERANCE = 0;

    @Test
    public void constructorTest(){
        boolean failed = false;

        Image image = null;
        try {
            image = new Image("/Users/brianhicks/IdeaProjects/OCR/src/test/resources/color_test.jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }

        Pixel redPixel = image.getPixel(0, 0);
        Pixel greenPixel = image.getPixel(0, 1);
        Pixel bluePixel = image.getPixel(0, 2);

        // Check red pixel
        boolean redPass = checkPixelValues(redPixel, 255, 0, 0, 255, "RedPixel");
        System.out.print("ImageTest/Test 1a: Red Accuracy --- ");
        if(redPass){
            System.out.println("Passed");
        } else {
            System.out.println("Failed");
            failed = true;
        }

        boolean bluePass = checkPixelValues(bluePixel, 0, 0, 255, 255, "BluePixel");
        System.out.print("ImageTest/Test 1b: Blue Accuracy --- ");
        if(bluePass){
            System.out.println("Passed");
        } else {
            System.out.println("Failed");
            failed = true;
        }

        boolean greenPass = checkPixelValues(greenPixel, 0, 255, 0, 255, "GreenPixel");
        System.out.print("ImageTest/Test 1c: Green Accuracy --- ");
        if(greenPass){
            System.out.println("Passed");
        } else {
            System.out.println("Failed");
            failed = true;
        }

        assert(!failed);

        image.saveImage("/Users/brianhicks/test.png");


        try {
            image = new Image("/Users/brianhicks/IdeaProjects/OCR/src/test/resources/color_test.jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }

        Image adjustedImage = image.adjustSize(1, 6);
        image.saveImage("/Users/brianhicks/test1.png");
        adjustedImage.saveImage("/Users/brianhicks/test2.png");
        Pixel pix = adjustedImage.getPixel(0, 0);
        System.out.println(pix.getRed());
        System.out.println(pix.getGreen());
        System.out.println(pix.getBlue());
        System.out.println(pix.getAlpha());

    }

    private boolean isWithinTolerance(int value, int expected, int tolerance){
        // Check if bounded from below by tolerance
        if(value >= expected - tolerance){

            // Check if bounded from above by tolerance
            if(value <= expected + tolerance){
                return true;
            }
        }
        return false;
    }

    private boolean checkPixelValues(Pixel pixel, int red, int green, int blue, int alpha, String pixelName){
        int pRed = pixel.getRed();
        int pGreen = pixel.getGreen();
        int pBlue = pixel.getBlue();
        int pAlpha = pixel.getAlpha();

        boolean passes = true;
        if(!isWithinTolerance(pRed, red, RGB_TOLERANCE)){
            System.err.println(pixelName + "'s red value is outside of tolerance");
            System.err.println("\t Actual: " + red + "; Expected Range: " + (red - RGB_TOLERANCE) + " - " + (red + RGB_TOLERANCE));
            passes = false;
        }

        if(!isWithinTolerance(pBlue, blue, RGB_TOLERANCE)){
            System.err.println(pixelName + "'s blue value is outside of tolerance");
            System.err.println("\t Actual: " + blue + "; Expected Range: " + (blue - RGB_TOLERANCE) + " - " + (blue + RGB_TOLERANCE));
            passes = false;
        }

        if(!isWithinTolerance(pGreen, green, RGB_TOLERANCE)){
            System.err.println(pixelName + "'s green value is outside of tolerance");
            System.err.println("\t Actual: " + green + "; Expected Range: " + (green - RGB_TOLERANCE) + " - " + (green + RGB_TOLERANCE));
            passes = false;
        }

        if(!isWithinTolerance(pAlpha, alpha, ALPHA_TOLERANCE)){
            System.err.println(pixelName + "'s alpha value is outside of tolerance");
            System.err.println("\t Actual: " + alpha + "; Expected Range: " + (alpha - ALPHA_TOLERANCE) + " - " + (alpha + ALPHA_TOLERANCE));
            passes = false;
        }

        return passes;
    }


}

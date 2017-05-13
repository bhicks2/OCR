import image.Image;
import image.Pixel;
import org.testng.annotations.Test;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by brianhicks on 5/10/17.
 */
public class ImageTest {

    private static final int RGB_TOLERANCE = 1;
    private static final int ALPHA_TOLERANCE = 0;

    @Test
    public void constructorTest(){
        String testFilePath = getClass().getResource("color_test.jpg").getPath();

        Image image = null;
        try {
            image = new Image(testFilePath);
        } catch (IOException e){
            e.printStackTrace();

            // Print the report
            System.out.println("Constructor Test: FAILED");
            System.out.println("\t[1] Load Image: FAILED");
            return;
        }

        // check dimensions
        boolean heightPass = (image.getHeight() == 1);
        boolean widthPass = (image.getWidth() == 3);

        // Check that pixels have the correct values
        Pixel redPixel = image.getPixel(0, 0);
        Pixel greenPixel = image.getPixel(0,1);
        Pixel bluePixel = image.getPixel(0, 2);

        boolean redPass = checkPixelValues(redPixel, 255, 0, 0,255, "RedPixel");
        boolean greenPass = checkPixelValues(greenPixel, 0, 255, 0,255, "GreenPixel");
        boolean bluePass = checkPixelValues(bluePixel, 0,0,255,255,"BluePixel");

        // Ensure correct behavior when the image is not found.
        String fakeImagePath = "feaf";
        boolean fakePass = false;
        try {
            image = new Image(fakeImagePath);
        } catch(FileNotFoundException e){
            fakePass = true;
        } catch(IOException e){
            e.printStackTrace();
        }

        // Print the report
        boolean passed = heightPass && widthPass && redPass && greenPass && bluePass && fakePass;
        System.out.println("Constructor Test: " + (passed ? "PASSED" : "FAILED"));
        System.out.println("\t[1] Load Image: PASSED");
        System.out.println("\t[2] Height Check: " + (heightPass ? "PASSED" : "FAILED"));
        System.out.println("\t[3] Width Check: " + (widthPass ? "PASSED" : "FAILED"));
        System.out.println("\t[4] Red Pixel Check: " + (redPass ? "PASSED" : "FAILED"));
        System.out.println("\t[5] Green Pixel Check: " + (greenPass ? "PASSED" : "FAILED"));
        System.out.println("\t[6] Blue Pixel Check: " + (bluePass ? "PASSED" : "FAILED"));
        System.out.println("\t[7] FileNotFound Check: " + (fakePass ? "PASSED" : "FAILED"));
    }

    private boolean isWithinTolerance(int value, int expected, int tolerance){
        return value >= expected - tolerance && value <= expected + tolerance;
    }

    private boolean checkPixelValues(Pixel pixel, int red, int green, int blue, int alpha, String pixelName){
        int pRed = pixel.getRed();
        int pGreen = pixel.getGreen();
        int pBlue = pixel.getBlue();
        int pAlpha = pixel.getAlpha();

        boolean redPasses = isWithinTolerance(pRed, red, RGB_TOLERANCE);
        boolean bluePasses = isWithinTolerance(pBlue, blue, RGB_TOLERANCE);
        boolean greenPasses = isWithinTolerance(pGreen, green, RGB_TOLERANCE);
        boolean alphaPasses = isWithinTolerance(pAlpha, alpha, ALPHA_TOLERANCE);

        return redPasses && bluePasses && greenPasses && alphaPasses;
    }


}

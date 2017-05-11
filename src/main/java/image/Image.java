package image;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by brianhicks on 5/10/17.
 */
public class Image {

    private static final int ALPHA_MASK = 0xFF000000;
    private static final int RED_MASK = 0x00FF0000;
    private static final int GREEN_MASK = 0x0000FF00;
    private static final int BLUE_MASK = 0x000000FF;

    private static final int ALPHA_SHIFT = 24;
    private static final int RED_SHIFT = 16;
    private static final int GREEN_SHIFT = 8;
    private static final int BLUE_SHIFT = 0;

    private Pixel[][] image;
    private int height;
    private int width;

    // ====================================================
    // constructor Image(String)
    // ====================================================
    // This constructor takes in a filename as a string,
    // and constructs an Image object from the image in
    // the given file.
    // ====================================================
    // Exceptions: this file with throw a FileNotFound
    // exception if the provided file does not exist, and
    // an IOException if the image cannot be read.
    // ====================================================
    public Image(String filename) throws IOException {
        BufferedImage image;

        // Make sure the file exists before trying to
        // read it.
        File file = new File(filename);
        if(!file.exists()){
            throw new FileNotFoundException("No file " + file.getAbsolutePath() + " could be found");
        }

        // Try to read the file
        try{
            image = ImageIO.read(file);
        } catch (IOException e) {
            throw new IOException("Unable to read image from file " + filename);
        }


        // Set the height and width of the image
        this.height = image.getHeight();
        this.width = image.getWidth();

        // Convert BufferedImage to a 2D array of Pixels
        this.image = convertToArray(image);
    }

    public Image(Pixel[][] image){
        this.image = image;

        this.height = image.length;
        this.width = this.height == 0 ? 0 : image[0].length;
    }

    public Image(Image image){
        this.width = image.width;
        this.height = image.height;

        this.image = new Pixel[this.height][this.width];
        for(int row = 0; row < this.height; row++){
            for(int col = 0; col < this.width; col++){
                this.image[row][col] = new Pixel(image.image[row][col]);
            }
        }
    }

    public Image extractSubImage(int row, int col, int height, int width){
        Pixel[][] subimage = new Pixel[height][width];
        for(int r = row; r < row + height; r++){
            for(int c = col; c < col + width; c++){
                Pixel pixel = new Pixel(image[r][c]);

                int newRow = r - row;
                int newCol = c - col;

                pixel.setColumn(newCol);
                pixel.setRow(newRow);

                subimage[newRow][newCol] = pixel;
            }
        }

        return new Image(subimage);
    }

    // ====================================================
    // method getPixel(int row, int col)
    // ====================================================
    // This function returns the pixel located at the
    // given row and col. Note that the origin (0, 0) is
    // located at the upper left of the image.
    // ====================================================
    // Exceptions: this function throws an ArrayOutOfBounds
    // exception if the given row or col is outside of the
    // bounds of the image.
    // ====================================================
    public Pixel getPixel(int row, int col){
        return image[row][col];
    }

    // ====================================================
    // method saveImage(String filename)
    // ====================================================
    // This function saves the image to the specified file,
    // as defined in the filename.
    //
    // Note that filename is expected to have an extension.
    // Although the program will work without one, it will
    // default to using .png as the extension.
    public void saveImage(String filename){
        int startOfExtension = filename.lastIndexOf(".");

        String extension = "png";
        if(startOfExtension != -1){
            extension = filename.substring(startOfExtension + 1);
        } else {
            filename = filename.concat("." + extension);
        }

        BufferedImage image = buildBufferedImage();

        File file = new File(filename);

        try {
            System.out.println("Writing");
            ImageIO.write(image, extension, file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Done Writing");
    }

    public int getHeight(){
        return this.height;
    }

    public int getWidth(){
        return this.width;
    }

    public Image adjustSize(int newHeight, int newWidth){
        Pixel[][] newImage = new Pixel[newHeight][newWidth];

        double gridHeight = ((double) height)/newHeight;
        double gridWidth = ((double) width) / newWidth;

        double area = (gridHeight * gridWidth);

        for(int newRow = 0; newRow < newHeight; newRow++){
            for(int newCol = 0; newCol < newWidth; newCol++){
                double rowLower = gridHeight * newRow;
                double rowUpper = gridHeight * (newRow + 1);

                double colLower = gridWidth * newCol;
                double colUpper = gridWidth * (newCol + 1);

                int pixelRowLower = (int) Math.floor(rowLower);
                int pixelRowUpper = (int) Math.ceil(rowUpper);

                int pixelColLower = (int) Math.floor(colLower);
                int pixelColUpper = (int) Math.ceil(colUpper);

                double red = 0.0;
                double green = 0.0;
                double blue = 0.0;
                double alpha = 0.0;

                for(int pixelRow = pixelRowLower; pixelRow < pixelRowUpper; pixelRow++){
                    for(int pixelCol = pixelColLower; pixelCol < pixelColUpper; pixelCol++){
                        double includedHeight = Math.min(1, Math.min(rowUpper - pixelRow, Math.min(pixelRow + 1 - rowLower, rowUpper - rowLower)));
                        double includedWidth = Math.min(1, Math.min(colUpper - pixelCol, Math.min(pixelCol + 1 - colLower, colUpper - colLower)));

                        double includedArea = includedHeight*includedWidth;

                        Pixel pixel = image[pixelRow][pixelCol];

                        int pixelRed = pixel.getRed();
                        int pixelBlue = pixel.getBlue();
                        int pixelGreen = pixel.getGreen();
                        int pixelAlpha = pixel.getAlpha();

                        red += includedArea * pixelRed;
                        green += includedArea * pixelGreen;
                        blue += includedArea * pixelBlue;
                        alpha += includedArea * pixelAlpha;
                    }
                }

                red = red/area;
                blue = blue/area;
                green = green/area;
                alpha = alpha/area;

                newImage[newRow][newCol] = new Pixel(newRow, newCol, (int) red, (int) green, (int) blue, (int) alpha);
            }
        }

        return new Image(newImage);
    }

    // ====================================================
    // method convertToArray(BufferedImage)
    // ====================================================
    // Utility function that takes in a BufferedImage image
    // and converts it to a 2D array of Pixel objects
    // representing the individual pixels of the image.
    //
    // The array generated stores pixels in matrix notation
    // so that a pixel at (i, j) means that its in the ith
    // row and the jth column of the image, with the origin
    // at the upper left of the image.
    // ====================================================
    // Exceptions: this function should never throw an
    // an exception
    // ====================================================
    private Pixel[][] convertToArray(BufferedImage image){
        Pixel[][] array = new Pixel[this.height][this.width];

        for(int row = 0; row < this.height; row++){
            for(int col = 0; col < this.width; col++){
                Color color = new Color(image.getRGB(col, row));
                array[row][col] = new Pixel(row, col, color);
            }
        }
        return array;
    }

    private BufferedImage buildBufferedImage() {
        BufferedImage newImage = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_ARGB);
        for (int row = 0; row < this.height; row++) {
            for (int col = 0; col < this.width; col++) {
                Pixel pixel = image[row][col];

                int redComp = (pixel.getRed() << RED_SHIFT) & RED_MASK;
                int blueComp = (pixel.getBlue() << BLUE_SHIFT) & BLUE_MASK;
                int greenComp = (pixel.getGreen() << GREEN_SHIFT) & GREEN_MASK;
                int alphaComp = (pixel.getAlpha() << ALPHA_SHIFT) & ALPHA_MASK;
                int rgb = redComp | blueComp | greenComp | alphaComp;

                newImage.setRGB(col, row, rgb);
            }
        }

        return newImage;
    }
}

package image;

import java.awt.*;

public class Pixel{

    private int red;
    private int green;
    private int blue;
    private int alpha;
    private int row;
    private int col;

    public Pixel(int row, int col, int red, int green, int blue, int alpha){
        this.red = red;
        this.green = green;
        this.blue = blue;
        this.alpha = alpha;
        this.row = row;
        this.col = col;
    }

    public Pixel(int x, int y, Color color){
        this.row = row;
        this.col = col;
        this.red = color.getRed();
        this.blue = color.getBlue();
        this.green = color.getGreen();
        this.alpha = color.getAlpha();
    }

    public Pixel(Pixel pixel){
        this.red = pixel.red;
        this.green = pixel.green;
        this.blue = pixel.blue;
        this.alpha = pixel.alpha;
        this.row = pixel.row;
        this.col = pixel.col;
    }

    public Color getColor(){
        return new Color(red, green, blue, alpha);
    }

    public int getRed(){
        return red;
    }

    public int getGreen(){
        return green;
    }

    public int getBlue(){
        return blue;
    }

    public int getAlpha(){
        return alpha;
    }

    public int getRow(){
        return row;
    }

    public int getColumn(){
        return col;
    }

    public int getGrayscale(){
        return (int)(0.299 * red + 0.587 * green + 0.114 * blue);
    }

    public void setRow(int row){
        this.row = row;
    }

    public void setColumn(int col){
        this.col = col;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Pixel pixel = (Pixel) o;

        if (red != pixel.red) return false;
        if (green != pixel.green) return false;
        if (blue != pixel.blue) return false;
        if (alpha != pixel.alpha) return false;
        if (row != pixel.row) return false;
        return col == pixel.col;
    }

    @Override
    public int hashCode() {
        int result = red;
        result = 31 * result + green;
        result = 31 * result + blue;
        result = 31 * result + alpha;
        result = 31 * result + row;
        result = 31 * result + col;
        return result;
    }
}
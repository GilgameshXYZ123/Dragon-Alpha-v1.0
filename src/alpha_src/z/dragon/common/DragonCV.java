/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.util.Objects;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import z.util.math.vector.Vector;

/**
 * @author Gilgamesh
 */
public class DragonCV 
{
    private DragonCV() {}
    
    public static DragonCV instance() { return cv; }
    public static final DragonCV cv = new DragonCV();
 
    //<editor-fold defaultstate="collapsed" desc="enum: Byte Image Types">
    /**
     * <pre> 
     * 3Byte: (Blue, Green, Red). 
     * Represents an image with 8-bit RGB color components, corresponding
     * to a Windows-style BGR color model, with the colors Blue, Green,
     * and Red stored in 3 bytes.  There is no alpha.  The image has a
     * <code>ComponentColorModel</code>.
     * When data with non-opaque alpha is stored
     * in an image of this type,
     * the color data must be adjusted to a non-premultiplied form
     * and the alpha discarded,
     * as described in the
     * {@link java.awt.AlphaComposite} documentation.
     * </pre>
     */
    public final int TYPE_3BYTE_BGR = BufferedImage.TYPE_3BYTE_BGR;
    
    /**
     * <pre> 
     * 4Byte: (Alpha, Blue, Green ,Red). 
     * Represents an image with 8-bit RGBA color components with the colors
     * Blue, Green, and Red stored in 3 bytes and 1 byte of alpha.  The
     * image has a <code>ComponentColorModel</code> with alpha.  The
     * color data in this image is considered not to be premultiplied with
     * alpha.  The byte data is interleaved in a single
     * byte array in the order A, B, G, R
     * from lower to higher byte addresses within each pixel.
     * </pre>
     */
    public final int TYPE_4BYTE_ABGR = BufferedImage.TYPE_4BYTE_ABGR;
    
    /**
     * <pre>
     * 4Byte: (Alpha, Blue, Green, Red). 
     * The color data in this image is considered to be premultiplied with alpha.
     * Represents an image with 8-bit RGBA color components with the colors
     * Blue, Green, and Red stored in 3 bytes and 1 byte of alpha.  The
     * image has a <code>ComponentColorModel</code> with alpha. The color
     * data in this image is considered to be premultiplied with alpha.
     * The byte data is interleaved in a single byte array in the order
     * A, B, G, R from lower to higher byte addresses within each pixel.
     * </pre>
     */
    public final int TYPE_4BYTE_ABGR_PRE = BufferedImage.TYPE_4BYTE_ABGR_PRE;
    
    /**
     * <pre>
     *  Represents an indexed byte image.  When this type is used as the
     * <code>imageType</code> argument to the <code>BufferedImage</code>
     * constructor that takes an <code>imageType</code> argument
     * but no <code>ColorModel</code> argument, an
     * <code>IndexColorModel</code> is created with
     * a 256-color 6/6/6 color cube palette with the rest of the colors
     * from 216-255 populated by grayscale values in the
     * default sRGB ColorSpace.
     *
     * <p> When color data is stored in an image of this type,
     * the closest color in the colormap is determined
     * by the <code>IndexColorModel</code> and the resulting index is stored.
     * Approximation and loss of alpha or color components
     * can result, depending on the colors in the
     * <code>IndexColorModel</code> colormap.
     * </pre>
     */
    public final int TYPE_BYTE_BINARY = BufferedImage.TYPE_BYTE_BINARY;
    
    /**
     * <pre>
     * Represents a unsigned byte grayscale image, non-indexed.  This
     * image has a <code>ComponentColorModel</code> with a CS_GRAY
     * {@link ColorSpace}.
     * When data with non-opaque alpha is stored
     * in an image of this type,
     * the color data must be adjusted to a non-premultiplied form
     * and the alpha discarded,
     * as described in the
     * {@link java.awt.AlphaComposite} documentation.
     * </pre>
     */
    public final int TYPE_BYTE_GRAY = BufferedImage.TYPE_BYTE_GRAY;
    
     /**
     * <pre>
     * Represents an opaque byte-packed 1, 2, or 4 bit image.  The
     * image has an {@link IndexColorModel} without alpha.  When this
     * type is used as the <code>imageType</code> argument to the
     * <code>BufferedImage</code> constructor that takes an
     * <code>imageType</code> argument but no <code>ColorModel</code>
     * argument, a 1-bit image is created with an
     * <code>IndexColorModel</code> with two colors in the default
     * sRGB <code>ColorSpace</code>: {0,&nbsp;0,&nbsp;0} and
     * {255,&nbsp;255,&nbsp;255}.
     *
     * <p> Images with 2 or 4 bits per pixel may be constructed via
     * the <code>BufferedImage</code> constructor that takes a
     * <code>ColorModel</code> argument by supplying a
     * <code>ColorModel</code> with an appropriate map size.
     *
     * <p> Images with 8 bits per pixel should use the image types
     * <code>TYPE_BYTE_INDEXED</code> or <code>TYPE_BYTE_GRAY</code>
     * depending on their <code>ColorModel</code>.

     * <p> When color data is stored in an image of this type,
     * the closest color in the colormap is determined
     * by the <code>IndexColorModel</code> and the resulting index is stored.
     * Approximation and loss of alpha or color components
     * can result, depending on the colors in the
     * <code>IndexColorModel</code> colormap.
     * </pre>
     */
    public static final int TYPE_BYTE_INDEXED = BufferedImage.TYPE_BYTE_INDEXED;
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: read image">
    public BufferedImage imread(InputStream in) throws IOException {
        return ImageIO.read(in);
    }
    public BufferedImage imread(String path) throws IOException {
        return ImageIO.read(new File(path)); 
    }
    public BufferedImage imread(File file) throws IOException {
        return ImageIO.read(file);
    }
    public BufferedImage imread(URL url) throws IOException {
        return ImageIO.read(url);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: write image file">
    public void imwrite(BufferedImage image, String format, OutputStream output) throws IOException 
    {
        if(!ImageIO.write(image, format, output)) 
            throw new IOException("No matched ImageWriter for:" + format);
    }
    public void imwrite(BufferedImage image, String format, String path) throws IOException 
    {
        File file = new File(path);
        if(!ImageIO.write(image, format, file)) 
            throw new IOException("No matched ImageWriter for:" + format);
    }
    public void imwrite(BufferedImage image, String path) throws IOException 
    {
        File file = new File(path);
        String name = file.getName();
        String format = name.substring(name.lastIndexOf('.') + 1);
        imwrite(image, format, path);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: info of image">
    public int channels(BufferedImage img) {
        return img.getColorModel().getNumComponents();
    }
    
    public int[] dim(BufferedImage img) {
        return new int[]{img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents()};
    }
    
    public int pixel_size(BufferedImage img) {
        return img.getColorModel().getPixelSize();//for precision and depth of color
    }
    
    public int[] channel_pixel_size(BufferedImage img) {
        return img.getColorModel().getComponentSize();
    }
    
    public String brief(BufferedImage img) 
    {
        StringBuilder sb = new StringBuilder(256);
        sb.append("Image { type = ").append(img.getType()).append(", ");
        sb.append("dim = [");Vector.append(sb, dim(img)); sb.append("], ");
        sb.append("pixelSize = ").append(pixel_size(img)).append(", channelPixelSize = [");
        Vector.append(sb, channel_pixel_size(img)); sb.append("]}");
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: show image">
    public void imshow(Image image) { imshow(image, "Image show"); }
    public void imshow(Image image, String title)
    {
        if(image == null) throw new NullPointerException("image is null");

        JFrame frame = new JFrame(title);
        JPanel panel = new JPanel();
        JLabel label = new JLabel();
        label.setIcon(new ImageIcon(image));
        panel.add(label);
        frame.add(panel);
        frame.pack();
        
        if(frame.getWidth()<256) frame.setSize(256, frame.getHeight());
        if(frame.getHeight()<256) frame.setSize(frame.getWidth(), 256);
        
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: reshape image">
    public BufferedImage reshape(BufferedImage img, double scaleY, double scaleX) {
        int height = (int) (img.getHeight() * scaleY);
        int width = (int) (img.getWidth() * scaleX);
        return reshape(img, height, width);
    }
    
    public BufferedImage reshape(BufferedImage img, double scaleY, double scaleX, Color bgColor)  {
        int height = (int) (img.getHeight() * scaleY);
        int width = (int) (img.getWidth() * scaleX);
        return reshape(img, height, width, bgColor);
    }
    
    public BufferedImage reshape(BufferedImage img, int height, int width) 
    {
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage reshape(BufferedImage img, int height, int width, Color bgColor) 
    {
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, bgColor, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage reshape(BufferedImage img, int height, int width, int type) 
    {
        BufferedImage dst = new BufferedImage(width, height, type);
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, null);
        graph.dispose();
        return dst;
    }
    public BufferedImage reshape(BufferedImage img, int height, int width, int type, Color bgColor) 
    {
        BufferedImage dst = new BufferedImage(width, height, type);
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, bgColor, null);
        graph.dispose();
        return dst;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: sub image">
    public BufferedImage subImage(BufferedImage img, 
            double scaleY1, double scaleX1, 
            double scaleY2, double scaleX2) 
    {
        int sh = img.getHeight(), sw = img.getWidth();
        int y1 = (int) (sh*scaleY1), y2 = (int) (sh*scaleY2);
        int x1 = (int) (sw*scaleX1), x2 = (int) (sw*scaleX2);
        return subImage(img, y1, x1, y2, x2);
    }
    
    public BufferedImage subImage(BufferedImage img, 
            double scaleY1, double scaleX1, 
            double scaleY2, double scaleX2,
            Color bgColor) 
    {
        int sh = img.getHeight(), sw = img.getWidth();
        int y1 = (int) (sh*scaleY1), y2 = (int) (sh*scaleY2);
        int x1 = (int) (sw*scaleX1), x2 = (int) (sw*scaleX2);
        return subImage(img, y1, x1, y2, x2, bgColor);
    }
    
    public BufferedImage subImage(BufferedImage img, int y1, int x1, int y2, int x2)
    {
        int sh = img.getHeight(), sw = img.getWidth();
        if(x1 >= sw) x1 = sw - 1; if(y1 >= sh) y1 = sh - 1;
        if(x2 >= sw) x2 = sw - 1; if(y2 >= sh) y2 = sh - 1;
        if(x1 > x2) {int t = x1; x1 = x2; x2 = t;}
        if(y1 > y2) {int t = y1; y1 = y2; y2 = t;}
        
        int width = x2 - x1 + 1, height = y2 - y1 + 1;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width - 1, height - 1, x1, y1, x2, y2, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage subImage(BufferedImage img, int y1, int x1, int y2, int x2, Color bgColor)
    {
        int sh = img.getHeight(), sw = img.getWidth();
        if(x1 >= sw) x1 = sw - 1; if(y1 >= sh) y1 = sh - 1;
        if(x2 >= sw) x2 = sw - 1; if(y2 >= sh) y2 = sh - 1;
        if(x1 > x2) {int t = x1; x1 = x2; x2 = t;}
        if(y1 > y2) {int t = y1; y1 = y2; y2 = t;}
        
        int width = x2 - x1 + 1, height = y2 - y1 + 1;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width - 1, height - 1, x1, y1, x2, y2, bgColor, null);
        graph.dispose();
        
        return dst;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine Transform">
    public AffineTransformer affine() { return new AffineTransformer(); }
    
    //<editor-fold defaultstate="collapsed" desc="class: Affine">
    public static final class AffineTransformer  
    {
        AffineTransform tf = new AffineTransform();
        
        //<editor-fold defaultstate="collapsed" desc="Basic-functions">
        public AffineTransform getTransfrom() {
            return tf;
        }
        public void setTransfrom(AffineTransform transfrom) {
            this.tf = transfrom;
        }
        
        public AffineTransformer copy() {
            AffineTransformer copied =  new AffineTransformer();
            copied.tf = (AffineTransform) this.tf.clone();
            return copied;
        }
        
        @Override 
        public int hashCode() { 
            return 95 + Objects.hashCode(this.tf); 
        }

        @Override
        public boolean equals(Object obj)  {
            if(!(obj instanceof AffineTransformer)) return false;
            AffineTransformer af = ( AffineTransformer) obj;
            return af.tf.equals(this.tf);
        }
       
        public void append(StringBuilder sb) 
        {
            double m00 = tf.getScaleX(), m01 = tf.getShearX(), m02 = tf.getTranslateX();
            double m10 = tf.getShearY(), m11 = tf.getScaleY(), m12 = tf.getTranslateY();
            sb.append("Affine Transfomer: \n");
            sb.append("[x'] = [").append(m00).append(", ").append(m01).append(", ").append(m02).append("] [x]\n");
            sb.append("[y'] = [").append(m10).append(", ").append(m11).append(", ").append(m12).append("] [y]");
        }
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(128);
            this.append(sb);
            return sb.toString();
        }
        //</editor-fold>
       
        //<editor-fold defaultstate="collapsed" desc="affine transforms">
        public AffineTransformer toIndentity() {
            tf.setToIdentity();
            return this;
        }
        
        public AffineTransformer invert() {
            try { tf.invert(); } 
            catch (NoninvertibleTransformException ex) {throw new RuntimeException(ex);}
            return this;
        }
        
        public AffineTransformer move(int moveY, int moveX) {
            tf.translate(moveX, moveY);
            return this;
        }

        public AffineTransformer scale(double scaleX, double scaleY) {
            tf.scale(scaleX, scaleY);
            return this;
        }
        
        public AffineTransformer rotate(double theta) {
            tf.rotate(theta);
            return this;
        }
        
        public AffineTransformer rotate(double theta, double anchorY, double anchorX) {
            tf.rotate(theta, anchorX, anchorY);
            return this;
        }
        
        public AffineTransformer angleRotate(double theta) {
            tf.rotate(theta * Math.PI / 180);
            return this;
        }
        
        public AffineTransformer angleRotate(double theta, double anchorY, double anchorX) {
            tf.rotate(theta * Math.PI / 180, anchorY, anchorX);
            return this;
        }

        public AffineTransformer shear(double shearY, double shearX) {
            tf.shear(shearX, shearY);
            return this;
        }
        //</editor-fold>
        
        public void draw(BufferedImage src, BufferedImage dst) {
            Graphics2D graph = dst.createGraphics();
            graph.drawImage(src, tf, null);
            graph.dispose();
        }
        
        public BufferedImage transform(BufferedImage src)  {
            int sw = src.getWidth(), sh = src.getHeight();
            double m00 = tf.getScaleX(), m01 = tf.getShearX(), m02 = tf.getTranslateX();
            double m10 = tf.getShearY(), m11 = tf.getScaleY(), m12 = tf.getTranslateY();
                 
            int width  = (int) (m00*sw + m01*sh + m02);
            int height = (int) (m10*sw + m11*sh + m12); 
            return transform(src, height, width);
        }
        
        public BufferedImage transform(BufferedImage src, int height, int width)  {
            BufferedImage dst = new BufferedImage(width, height, src.getType());
            Graphics2D graph = dst.createGraphics();
            graph.drawImage(src, tf, null);
            graph.dispose();
            return dst;
        }
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ColorSpace Transform">
    private final ColorConvertOp gray_op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
    private final ColorConvertOp bgr_op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_sRGB), null);
    
    public BufferedImage to_gray(BufferedImage img) {
        return gray_op.filter(img, null); 
    }
    
    public BufferedImage to_BGR(BufferedImage img) { 
        BufferedImage dst = new BufferedImage(img.getWidth(), img.getHeight(), TYPE_3BYTE_BGR);
        return bgr_op.filter(img, dst); 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image & pixel[]">
    public byte[] pixel(BufferedImage img) {
        DataBufferByte dataBuf = (DataBufferByte)img.getRaster().getDataBuffer();
        return dataBuf.getData();
    }
    
    private static final int[] BYTE4 = {8, 8, 8, 8};
    private static final int[] ARGB = {1, 2, 3, 0};
    private static final int[] ABGR = {3, 2, 1, 0};
    
    public BufferedImage ARGB(byte[] pixel, int width, int height) throws IOException {
        return channel4(pixel, width, height, ARGB);
    }
    
    public BufferedImage ABGR(byte[] pixel, int width, int height) throws IOException {
        return channel4(pixel, width, height, ABGR);
    }
    
    private BufferedImage channel4(byte[] pixel, int width, int height, int[] channel_order) throws IOException
    {
        if(pixel == null || pixel.length != width * height * 4) 
            throw new IllegalArgumentException("invalid image description for pictures with 4 channels");
        
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE4, true, false, //(hasAlpha = true)
                Transparency.TRANSLUCENT, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, width, height, width*4, 4, channel_order, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    
    private static final int[] BYTE3 = {8, 8, 8};
    private static final int[] RGB = {0, 1, 2};
    private static final int[] BGR = {2, 1, 0};

    public BufferedImage RGB(byte[] pixel, int width, int height) throws IOException {
        return channel3(pixel, width, height, RGB);
    }
    
    public BufferedImage BGR(byte[] pixel, int width, int height) throws IOException {
        return channel3(pixel, width, height, BGR);
    }
    
    private BufferedImage channel3(byte[] pixel, int width, int height, int[] channel_order) throws IOException
    {
        if(pixel == null || pixel.length != width * height * 3) 
            throw new IllegalArgumentException("invalid image description for RGBs");
      
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE3, false, false, 
                Transparency.OPAQUE, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, width, height, width*3, 3, channel_order, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    
    private static final int[] BYTE1 = {8};
    private static final int[] GRAY = {0};
    
    public BufferedImage gray(byte[] pixel, int width, int height) throws IOException
    {
        if(pixel == null || pixel.length != width * height) 
            throw new IllegalArgumentException("invalid image description for RGBs");
          
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_GRAY);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE1, false, false, 
                Transparency.OPAQUE, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, width, height, width, 1, GRAY, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    //</editor-fold>
}

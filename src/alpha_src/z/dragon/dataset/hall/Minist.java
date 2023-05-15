/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.dataset.hall;

import java.awt.image.BufferedImage;
import java.io.IOException;
import static z.dragon.alpha.Alpha.Datas.data;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonFile.fl;
import z.dragon.data.DataSet;
import z.dragon.data.container.ListContainer;

/**
 *
 * @author Gilgamesh
 */
public final class Minist 
{
    private Minist() {}
    
    public static final int num_class = 10;//use one hot coding
    public static final int[] picture_dim = {28, 28, 1};
    
    public static final int test_set_size = 10000;
    public static final String test_image_file = "t10k-images.idx3-ubyte";
    public static final String test_label_file = "t10k-labels.idx1-ubyte";
    
    public static final int train_set_size = 60000;
    public static final String train_image_file = "train-images.idx3-ubyte";
    public static final String train_label_file = "train-labels.idx1-ubyte";
    
    //<editor-fold defaultstate="collapsed" desc="data-load">
    private static void load(ListContainer<byte[], Integer> conta, int length, 
            String imageName, String labelName) throws IOException 
    {
        String alpha_home = alpha.home();
        String imagePath = alpha_home + "\\data\\minist\\" + imageName;
        String labelPath = alpha_home + "\\data\\minist\\" + labelName;

        byte[] imageArr = fl.create(imagePath).toBytes();
        byte[] labelArr = fl.create(labelPath).toBytes();
        
        int imageOffset = 16, labelOffset = 8;
        for(int i=0; i<length; i++) 
        {
            byte[] pixel = new byte[784]; 
            System.arraycopy(imageArr, imageOffset, pixel, 0, 784);
            imageOffset += 784;
            
            int label =  labelArr[labelOffset++];
            conta.add(pixel, label);
        }
    }
    //</editor-fold>
    public static DataSet<byte[], Integer> test()  {
        ListContainer<byte[], Integer> con = data.list(byte[].class, Integer.class, test_set_size);
        try { load(con, test_set_size, test_image_file, test_label_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        
        return data.dataset(con)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.onehot_transform(num_class));
    }
    
    public static DataSet<byte[], Integer> train() {
        ListContainer<byte[], Integer> con = new ListContainer<>(byte[].class, Integer.class, train_set_size);
        try { load(con, train_set_size, train_image_file, train_label_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(con)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.onehot_transform(num_class));
    }
}

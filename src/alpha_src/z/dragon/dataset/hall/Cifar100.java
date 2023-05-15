/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.dataset.hall;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import static z.dragon.alpha.Alpha.Datas.data;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.common.DragonFile.BufferedFile;
import static z.dragon.common.DragonFile.fl;
import z.dragon.data.DataSet;
import z.dragon.data.container.ListContainer;

/**
 *
 * @author Gilgamesh
 */
public final class Cifar100 
{
    private Cifar100() {}
    
    public static final int num_class_fine = 100;
    public static final int[] picture_dim = {32, 32, 3};

    public static final String test_file = "test.bin";
    public static int test_set_size = 10000;
    
    public static int train_set_size = 50000;
    public static final String train_file = "train.bin";
    
    //<editor-fold defaultstate="collapsed" desc="dataset-load">
    private static void load_fine(ListContainer<byte[], Integer> ds, String name, int length) throws IOException 
    {
        String path = alpha.home() + "\\data\\cifar100\\" + name;
        byte[] arr = fl.create(path).toBytes();
        for(int i=0, offset = 0; i<length; i++) {
            int label = arr[offset]; offset += 2;
            
            byte[] pixel = new byte[3072];
            for(int j=0; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=1; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=2; j<3072; j+=3) pixel[j] = arr[offset++];
            
            ds.add(pixel, label);
        }
    }
    //</editor-fold>
    
    public static DataSet<byte[], Integer> test() {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, test_set_size);
        try { load_fine(conta, test_file, test_set_size); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta,
                data.image_transform(picture_dim),
                data.onehot_transform(num_class_fine));
    }
   
    public static DataSet<byte[], Integer> train() {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, train_set_size);
        try { load_fine(conta, train_file, train_set_size); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta,
                data.image_transform(picture_dim),
                data.onehot_transform(num_class_fine));
    }
    
    public static Map<Integer, String> fine_labels() throws IOException {
        HashMap<Integer, String> mt = new HashMap<>();
        BufferedFile file = fl.create(alpha.home() + "\\data\\cifar100\\fine_label_names.txt");
        String line = null; int index = 0;
        while((line = file.nextLine()) != null) mt.put(index++, line);
        return mt;
    }
}

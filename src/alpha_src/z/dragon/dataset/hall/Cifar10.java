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
 * @author Gilgamesh
 */
public final class Cifar10 
{
    private Cifar10() {}
    
    public static final int num_class = 10;//use one hot coding
    public static final int[] picture_dim = {32, 32, 3};
    
    public static final int test_set_size = 10000;
    public static final String test_set_file = "test_batch.bin";
    
    public static final int train_set_size = 50000;
    public static final String[] train_set_files = { 
        "data_batch_1.bin", 
        "data_batch_2.bin",  
        "data_batch_3.bin", 
        "data_batch_4.bin", 
        "data_batch_5.bin"
    };
    
    //<editor-fold defaultstate="collapsed" desc="dataset-load">
    private static void load(ListContainer<byte[], Integer> con, String name) throws IOException 
    {
        String path = alpha.home() + "\\data\\cifar10\\" + name;
        byte[] arr = fl.create(path).toBytes();
        
        for(int i=0, offset = 0; i<10000; i++) {
            int label = arr[offset++];
            
            byte[] pixel = new byte[3072]; 
            for(int j=0; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=1; j<3072; j+=3) pixel[j] = arr[offset++];
            for(int j=2; j<3072; j+=3) pixel[j] = arr[offset++];
            
            con.add(pixel, label);
        }
    }
    //</editor-fold>
    public static DataSet<byte[], Integer> test() {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, test_set_size);
        try { load(conta, test_set_file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.onehot_transform(num_class));
    }
    
    public static DataSet<byte[], Integer> train() {
        ListContainer<byte[], Integer> conta = data.list(byte[].class, Integer.class, train_set_size);
        try { for(String file : train_set_files) load(conta, file); }
        catch(IOException e) { throw new RuntimeException(e); }
        return data.dataset(conta)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.onehot_transform(num_class));
    }
    
    public static Map<Integer, String> labels() throws IOException  {
        HashMap<Integer, String> mt = new HashMap<>();
        BufferedFile file = fl.create(alpha.home() + "\\data\\cifar10\\batches.meta.txt");
        String line; int index = 0;
        while((line = file.nextLine()) != null)  mt.put(index++, line);
        return mt;
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg19;

import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.DataSet;
import z.dragon.data.TensorIter;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.dataset.hall.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static int batch_size = 128;
    
    public static void test() throws IOException
    {
        VGG19 net = new VGG19().eval().init(eg).println(); net.load();
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        DataSet<byte[], Integer> dataset = Cifar10.train();
//        DataSet<byte[], Integer> dataset = Cifar10.test();
        
        double accuracy = 0;
        int batchIdx = 0;
        for(TensorIter iter = dataset.batch_iter(); iter.hasNext(); ) {
            TensorPair pair = iter.next(eg, batch_size);
            Tensor yh = net.forward(pair.input)[0];
            
            float ls = loss.loss(yh, pair.label).get();
            System.out.println("loss = " + ls);
            
            Tensor pre = eg.row_max_index(yh).c();//<int 32>
            Tensor lab = eg.row_max_index(pair.label).c();
            
            Vector.println("Y1: ", pre.value_int32(), 0, batch_size);
            Vector.println("Y2: ", lab.value_int32(), 0, batch_size);
            
            float eq = eg.straight_equal(pre, lab).get();
            System.out.println("eq = " + eq);
            accuracy += eq;
            
            net.gc(); pre.delete(); lab.delete(); 
            batchIdx++;
        }
        
        System.out.println("accuracy = " + accuracy / batchIdx);
    }
    
    public static void main(String[] args)
    {
        try
        {
            test();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}

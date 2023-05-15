/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg16;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.DataSet;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.dataset.hall.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 128);
    
    static float lr = 0.001f;//learning_rate
    static int batch_size = 512;
    
    public static void train(int epoch)
    {
        VGG16 net = new VGG16().train().init(eg).println();
        Optimizer opt = alpha.optim.Adam(net.params(), lr);
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        DataSet<byte[], Integer> train_set = Cifar10.train();
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        eg.sync(false).check(false);
        
        int batchIdx = 0; 
       
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<epoch; i++)  {
            System.out.println("epoch = " + i);
            for(iter.reset(true); iter.hasNext(); batchIdx++) {
                TensorPair pair = iter.next();
                Tensor x = pair.input;
                Tensor y = pair.label;
                
                Tensor yh = net.forward(x)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(yh, y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(yh, y));
                opt.update().clear_grads();
                net.gc();
            }
            System.out.println("\n");
        }
        long div = timer.record().timeStampDifMills();
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f * div / 1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            train(20);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}

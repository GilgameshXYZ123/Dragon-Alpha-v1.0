package test.cuda;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_optimizer_test 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] W = Vector.randomFloatVector(length, 0, 1f);
        float[] V = Vector.randomFloatVector(length, 0, 1f);
        float[] S = Vector.randomFloatVector(length, 0, 1f);
        float[] G = Vector.randomFloatVector(length, 0, 1f);
        float[] deltaW = Vector.randomFloatVector(length, -1f, 1f);
        
        Tensor tW = eg.tensor(W, 1, 1, height, width);
        Tensor tV = eg.tensor(V, 1, height, width);
        Tensor tS = eg.tensor(S, 1, height, width);
        Tensor tG = eg.tensor(G, 1, height, width);
        Tensor tdeltaW = eg.tensor(deltaW, height, width);
        
        
        float a1 = exr.nextFloat(0, 1f), a2 = 1 - a1;
        float b1 = exr.nextFloat(0, 1f), b2 = 1 - b1;
        float c1 = exr.nextFloat(0, 1f), c2 = 1 - c1;
        float lr_t  = exr.nextFloat(0, 1f);
        
        //CPU-------------------------------------------------------------------
//        Vector.Momentum(W, deltaW, V, a1, a2, lr_t, length);
//        Vector.RMSprop(W, deltaW, S, a1, a2, 1e-8f, lr_t, length);
        Vector.Adam(W, deltaW, V, a1, a2, S, b1, b2, 1e-8f, lr_t, length);    
//        Vector.Adamod(W, deltaW, V, a1, a2, S, b1, b2, 1e-8f, G, c1, c2, lr_t, length);
//        Vector.SGDMN(W, deltaW, V, a1, b1, c1, lr_t, length);;

     
        //GPU-------------------------------------------------------------------
//        tW = eg.momentum(tW, tV, a1, a2, tdeltaW, lr_t);
//        tW = eg.rmsprop(tW, tS, a1, a2, 1e-8f, tdeltaW, lr_t);
//        tW = eg.rmsprop(tW, tS, a1, a2, 1e-8f, tdeltaW, lr_t, 0, 0);
        tW = eg.adam(tW, tV, a1, a2, tS, b1, b2, 1e-8f, tdeltaW, lr_t).c();
//        tW = eg.adam_decay(tW, tV, a1, a2, tS, b1, b2, 1e-8f, tdeltaW, lr_t, 0, 0).c();
//        tW = eg.adamod(tW, tV, a1, a2, tS, b1, b2, 1e-8f, tG, c1, c2, tdeltaW, lr_t);
//        tW = eg.adamod(tW, tV, a1, a2, tS, b1, b2, 1e-8f, tG, c1, c2, tdeltaW, lr_t, 0, 0);
//        tW = eg.sgdmn(tW, tV, a1, b1, c1, tdeltaW, lr_t);
//        tW = eg.sgdmn(tW, tV, a1, b1, c1, tdeltaW, lr_t, 0, 0);
        
        
        //compare---------------------------------------------------------------
        float[] W2 = eg.valueOf(tW);
        float[] V2 = eg.valueOf(tV);
        float[] S2 = eg.valueOf(tS);
        float[] G2 = eg.valueOf(tG);
        
        System.out.print("CPU: W: "); Vector.println(W, 0, 10);
        System.out.print("GPU: W: "); Vector.println(W2, 0, 10);
         
        float spW = Vector.samePercentRelative(W, W2, 1e-4f); System.out.println("spW:" + spW);
        float spV = Vector.samePercentAbsolute(V, V2); System.out.println("spV:" + spV);
        float spS = Vector.samePercentAbsolute(S, S2); System.out.println("spS:" + spS);
        float spG = Vector.samePercentAbsolute(G, G2); System.out.println("spG:" + spS);
        
        if(spW < 0.99) throw new RuntimeException();
        if(spV < 0.99) throw new RuntimeException();
        if(spS < 0.99) throw new RuntimeException();
        if(spG < 0.99) throw new RuntimeException();
      
        eg.delete(tW, tV, tS, tG, tdeltaW);
    }
    
    public static void testSpeed(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] W = Vector.randomFloatVector(length, 0, 1f);
        float[] V = Vector.randomFloatVector(length, 0, 1f);
        float[] S = Vector.randomFloatVector(length, 0, 1f);
        float[] deltaW = Vector.randomFloatVector(length, -1, 1);
     
        //GPU-------------------------------------------------------------------
        Tensor tW = eg.tensor(W, height, width);
        Tensor tV = eg.tensor(V, height, width);
        Tensor tS = eg.tensor(S, height, width);
        Tensor tdeltaW = eg.tensor(deltaW, height, width);

        float a1 = exr.nextFloat(0, 1f), a2 = 1 - a1;
        float b1 = exr.nextFloat(0, 1f), b2 = 1 - b1;
        float lr_t  = exr.nextFloat(0, 1f);
        
        int nIter = 3000;
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        for(int i=0; i < nIter; i++)
        {
//            tW = eg.momentum(tW, tV, a1, a2, tdeltaW, lr_t);
//            tW = eg.rmsprop(tW, tS, a1, a2, 1e-8f, tdeltaW, lr_t);
            tW = eg.adam(tW, tV, a1, a2, tS, b1, b2, 1e-8f, tdeltaW, lr_t);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 7;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tW, tV, tS);
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            for(int h=30; h<=50; h++)
                for(int w=40; w<=256; w++) testCorrect(h, w);
         
            testCorrect(55, 443);
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

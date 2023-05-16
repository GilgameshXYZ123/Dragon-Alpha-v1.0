package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
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
public class Cuda_reduce_field_test2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
   static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();

    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
        
//        float[] Y1 = Matrix.field_var(mX)[0];
        float[] Y1 = Matrix.field_stddev(mX)[0];
        
        System.out.print("CPU: "); Vector.println(Y1, 0, 10);
        //GPU-------------------------------------------------------------------
//        Tensor tY = eg.field_var(tX);
        Tensor tY = eg.field_std(tX);
    
        float[] Y2 = eg.valueOf(tY);
        System.out.print("GPU: "); Vector.println(Y2, 0, 10);
        //compare---------------------------------------------------------------
        
        float sp0 = Vector.samePercentRelative(Y1, Y2, 1e-5f); System.out.println("sp0:" + sp0);
        if(sp0 < 0.99) throw new RuntimeException();
      
        eg.delete(tY, tX);
    }
    
    public static void testSpeed(int height, int width)
    {
        eg.check(false);
        eg.sync(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, -1f);
        Tensor tX = eg.tensor(X, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.field_binomialSum(tX, 2 * 3 * width, 1, 2, 3).sync(); eg.delete(tY);
//            Tensor tY = eg.field_max(tX, 2 * 3 * width).sync(); eg.delete(tY);
            Tensor tY = eg.field_min(tX, width).c(); eg.delete(tY);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 1;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX);
    }
    public static void main(String[] args)
    {
        try
        {
//            for(int h=1; h<=20; h++)
//                for(int w=1; w<=256; w++) testCorrect(h, w);
////            
            Vector.PRINT_DIFFERENT = true;
            for(int h=100; h<=105; h++)
                for(int w= 128; w<=256; w++) testCorrect(h, w);
        
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

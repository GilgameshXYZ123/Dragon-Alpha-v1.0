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
public class Cuda_function_test1 
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
        
        float[] X1 = Vector.randomFloatVector(length, 0.5f, 1f); 
        float[] X2 = Vector.randomFloatVector(length, 0.5f, 1f);
        
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);
        
        float alpha = exr.nextFloat(-0.5f, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(-0.5f, 0.5f), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = -0.1f;
        
        //CPU-------------------------------------------------------------------
//        Vector.tan(alpha, X, beta, Y1, length);
//        Vector.cot(alpha, X, beta, Y1, length);
//        Vector.asin(alpha, X, beta, Y1, length);
//        Vector.arctan(alpha, X, beta, Y1, length);

//        Vector.sqrt(alpha, X, beta, Y1, length);
//        float[] Y1 = Vector.sqrt_quadratic2(X1, X2, alpha, alpha2, beta, beta2, gamma, k);

//        Vector.exp(alpha, X, beta, Y1, length);

//        Vector.println(X);
//        Vector.println(X2);
        
//        float[] Y1 = Vector.linear_greater(alpha, X1, beta);
        float[] Y1 = Vector.linear_greater2(X1, X2, alpha, beta, gamma);

        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.tan(false, alpha, tX, beta);
//        Tensor tY2 = eg.tan(true, alpha, tX, beta);
//        Tensor tY1 = eg.cot(false, alpha, tX, beta);
//        Tensor tY2 = eg.cot(true, alpha, tX, beta);
//        Tensor tY1 = eg.arcsin(false, alpha, tX, beta);
//        Tensor tY2 = eg.arcsin(true, alpha, tX, beta);
//        Tensor tY1 = eg.arctan(false, alpha, tX, beta);
//        Tensor tY2 = eg.arctan(false, alpha, tX, beta);
                
//        Tensor tY1 = eg.sqrt(false, alpha, tX, beta);
//        Tensor tY2 = eg.sqrt(true, alpha, tX, beta);
//        Tensor tY1 = eg.exp(false, alpha, tX, beta);
//        Tensor tY2 = eg.exp(true, alpha, tX, beta);
//        Tensor tY1 = eg.log(false, alpha, tX, beta);
//        Tensor tY2 = eg.log(true, alpha, tX, beta);
      
//        Tensor tY1 = eg.sqrt_quadratic2(false, tX1, tX2, alpha, alpha2, beta, beta2, gamma, k);
//        Tensor tY2 = eg.sqrt_quadratic2(true, tX1, tX2, alpha, alpha2, beta, beta2, gamma, k);

//        Tensor tY1 = eg.linear_greater(false, alpha, tX1, beta);
//        Tensor tY2 = eg.linear_greater(false, alpha, tX1, beta);
        
        Tensor tY1 = eg.linear_greater2(true, tX1, tX2, alpha, beta, gamma);
        Tensor tY2 = eg.linear_greater2(false, tX1, tX2, alpha, beta, gamma);
        
        //compare---------------------------------------------------------------
        float sum2 = eg.straight_sum(tY1).get();
        float sum3 = eg.straight_sum(tY2).get();
        System.out.println("sum(tY1) = " + sum2);
        System.out.println("sum(tY2) = " + sum3);
        
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
      
        float[] Y3 = eg.valueOf(tY2);
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        
        float sp1 = Vector.samePercentRelative(Y1, Y2, 0.01f); System.out.println("sp1:" + sp1);
        float sp2 = Vector.samePercentRelative(Y1, Y3, 0.01f); System.out.println("sp2:" + sp2);
         
        //delete----------------------------------------------------------------
        eg.delete(tX1, tX2, tY1);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        
        if(Float.isNaN(sum2)) throw new RuntimeException();
        if(Float.isNaN(sum3)) throw new RuntimeException();
    }
    
    public static void testSpeed(int height, int width)
    {
        eg.check(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length);
        float[] X2 = Vector.randomFloatVector(length);
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX2 = eg.tensor(X, 1, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.tan(true, 1, tX, 2);
//            Tensor tY = eg.arcsin(true, tX);
//            Tensor tY = eg.arctan(true, i, tX, i);
//            Tensor tY = eg.sqrt(true, 1, tX, 2);
//            Tensor tY = eg.exp(true, 1, tX, 2);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 2;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX);
    }
    public static void main(String[] args)
    {
        try
        {
            for(int h=20; h<=30; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
        
            testCorrect(1024, 1024);
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}
 
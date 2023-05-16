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
public class Cuda_reduce_field_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();

    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * 2 * 3 * stride;
        int length  = height * 2 * 3 * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, 3f);
        float[] X2 = Vector.randomFloatVector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, 2, 3, width);
        Tensor tX2 = eg.tensor(X2, height, 2, 3, width);
        
        float alpha = exr.nextFloat(-1f, 1f);
        float beta = exr.nextFloat(-1f, 1f);
        float gamma = exr.nextFloat(-1f, 1f);
        float alpha2 = exr.nextFloat(-1f, 1f);
        float beta2 = exr.nextFloat(-1f, 1f);
        float gamma2 = exr.nextFloat(-1f, 1f);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, 2 * 3 * width);
        float[][] mX2 = Matrix.toMatrix(X2, 2 * 3 * width);
        float[] Y1 = new float[2 * 3 * width];
        
//        Matrix.field_mean(mX, Y1);
        Matrix.field_squareMean(mX, Y1);

//        Matrix.field_linear(mX, alpha, beta, Y1);
//        Matrix.field_linear2(mX, mX2, alpha, beta, gamma, Y1);
//        Matrix.field_quadratic(mX, alpha, beta, gamma, Y1);
//        Matrix.field_quadratic2(mX, mX2, alpha, beta, gamma, alpha2, beta2, gamma2, Y1);
//        Matrix.field_max(mX, Y1);
//        Matrix.minOfEachField(mX, Y1);
     
        System.out.print("CPU: "); Vector.println(Y1, 0, 10);
        //GPU-------------------------------------------------------------------
//        Tensor tY = eg.field_linear(tX, 2 * 3 * width, alpha, beta).c();
//        Tensor tY = eg.field_linear2(tX, tX2, 2 * 3 * width, alpha, beta, gamma);
//        Tensor tY = eg.field_quadratic(tX, 2 * 3 * width, alpha, beta, gamma).c();
//        Tensor tY = eg.field_linear_quadratic(tX, 2 * 3 * width, alpha2, beta2, alpha, beta, gamma)[1].c();
//        Tensor tY = eg.field_mean(tX, 2 * 3 * width);
        Tensor tY = eg.field_sqmean(tX, 2 * 3 * width);
        
//        Tensor tY = eg.field_quadratic2(tX, tX2, 2 * 3 * width, alpha, beta, gamma, alpha2, beta2, gamma2);
//        Tensor tY = eg.field_max(tX, 2 * 3 * width).c();
//        Tensor tY = eg.reduce_field_min(tX, 2 * 3 * width).c();
    
        float[] Y2 = eg.valueOf(tY);
        System.out.print("GPU: "); Vector.println(Y2, 0, 10);
        //compare---------------------------------------------------------------
        
        float sp0 = Vector.samePercentRelative(Y1, Y2, 1e-3f); System.out.println("sp0:" + sp0);
        if(sp0 < 0.99) throw new RuntimeException();
      
        eg.delete(tY, tX, tX2);
    }
    
    public static void testSpeed(int height, int width)
    {
        eg.check(false).sync(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Correct:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, -1f);
        Tensor tX = eg.tensor(X, height, width);
        
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.field_linear(tX, 1.0f, 2.0f).c(); tY.delete();
            Tensor tY = eg.field_quadratic(tX, 1, 2, 3).c(); tY.delete();
        }
        Cuda.deviceSynchronize();
        
        float time = (float) timer.record().timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 1;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX);
    }
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            for(int h=1; h<=20; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            for(int h=100; h<=105; h++)
                for(int w= 128; w<=256; w++) testCorrect(h, w);
            for(int h=1024; h<=1028; h++)
                for(int w= 233; w<=256; w++) testCorrect(h, w);
            
            //71.839073 GB/s
            testCorrect(1024, 1024);//[512, 16, 16, 128]
            testSpeed(256*16*16, 128);  
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

package test.cuda;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
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
public class Cuda_function_field_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());     
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, 0, 4f);
        float[] X2 = Vector.randomFloatVector(height, 0, 4f);
        
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        float[][] mX1 = Matrix.toMatrix(X, width);
        float[][] mY1 = new float[height][width];
       
        Matrix.fieldVectorBinomial(mX1, X2, alpha, beta, alpha2, beta2, k, gamma, mY1, height, width);
//        Matrix.fieldVectorDiv(alpha, mX1, beta, alpha2, X2, beta2, gamma, mY1, height, width);
        
        float[] Y1 = Matrix.toVector(mY1);
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
       
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX2 = eg.tensor(X2, height);
        Tensor tY1 = eg.quadratic2_field(true, tX, tX2, alpha, beta, alpha2, beta2, k, gamma);
//        Tensor tY1 = eg.div_field(true, alpha, tX, beta, alpha2, tX2, beta2, gamma);

        //compare---------------------------------------------------------------
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercentRelative(Y1, Y2, 1e-3f); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        eg.delete(tX, tX2, tY1);
    }
    
    public static void testSpeed(int height, int width)
    {
        eg.check(false);
        eg.sync(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length);
        float[] X2 = Vector.randomFloatVector(height);
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX2 = eg.tensor(X2, height);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY1 = eg.binomial_field(true, tX, tX2, 1, 1, 1, 1, 1, 1);
            Tensor tY1 = eg.div_field(true, 1, tX, 1, 1, tX2, 1, 1);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 3;
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
            
            for(int h = 100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
        
            testCorrect(1023, 1023);
            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

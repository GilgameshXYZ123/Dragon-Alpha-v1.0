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
public class Cuda_function_softmax_test 
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
        
        float[] X = Vector.randomFloatVector(length, 0, 0.7f); Tensor tX = eg.tensor(X, height, width);
        float[] X2 = Vector.randomFloatVector(length, 0, 0.7f); Tensor tX2 = eg.tensor(X2, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX1 = Matrix.toMatrix(X, width);
        float[][] mY1 = new float[height][width];
//        
        Matrix.softmax(mX1, mY1, height, width);
        float[] Y1 = Matrix.toVector(mY1);//Y = softmax(X)

        float[] grad = new float[length];
//        Vector.sub(Y1, X2, grad, length);
        
//        Vector.crossEntropy(Y1, X2, Y1);//Y1 = crossEntropy(Y1, X2)
//        Vector.crossEntropy_deltaYh(Y1, X2, Y1);
//        Vector.sigmoid_Deri(Y1, Y1, length);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
       
        //GPU-------------------------------------------------------------------
//        Tensor tdeltaX1 = eg.softmax(false, tX, width).c();
//        Tensor tdeltaX2 = eg.softmax(true, tX, width).c();
//        
        Tensor tdeltaX2 = eg.softmax_crossEntropy_deltaX_naive(true, tX, tX2, width).c();
//        Tensor tdeltaX2 = eg.softmaxCrossEntropy_deltaX(tX, tX2, width).c();
        
        //tY = crossEntropy(softmax(X), X2);

        //compare---------------------------------------------------------------
//        float[] Y1 = eg.valueOf(tdeltaX1);
        float[] Y2 = eg.valueOf(tdeltaX2);
        
//        System.out.print("GPU1: "); Vector.println(Y1, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y2, 0, 10);
        
        float sp1 = Vector.samePercentRelative(Y1, Y2, 1e-4f); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        //delete----------------------------------------------------------------
        eg.delete(tX, tdeltaX2, tX2);
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
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX2 = eg.tensor(X, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.softmax(true, tX,  width).c();
            Tensor tY = eg.softmax_crossEntropy(tX, tX2, length).c(); eg.delete(tY);
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
            Vector.PRINT_DIFFERENT = true;
            
            for(int h=2; h<=20; h++)
                for(int w=2; w<=256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
        
            testCorrect(1024, 1024);
            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

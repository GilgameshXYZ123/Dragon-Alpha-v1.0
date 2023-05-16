/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_deltaX_dual_test 
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
        
        float[] X1 = Vector.randomFloatVector(length, -1, 1); 
        float[] X2 = Vector.randomFloatVector(length, -1, 1); 
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);
        
//        Vector.println("X1 = ", X1, 0, 10);
//        Vector.println("X1 = ", X2, 0, 10);
//        Vector.println("deltaY = ", deltaY, 0, 10);
        
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        float alpha1 = exr.nextFloat(0, 0.5f);
        float beta1 = exr.nextFloat(1, -1);
        float alpha2 = exr.nextFloat(1, -1);
        float beta2 = exr.nextFloat(0, -1.5f);
        float gamma = exr.nextFloat(0, -1.5f);
        
        //CPU-------------------------------------------------------------------
        float[] deriX1 = new float[length];
        float[] deriX2 = new float[length];
        float[] deltaX1 = new float[length];
        float[] deltaX2 = new float[length];
        
        Vector.div_Deri(deriX1, deriX2, X1, alpha1, beta1, X2, alpha2, beta2, length);
//        Vector.binomial_Deri(deriX1, deriX2, X1, X2, alpha1, beta1, alpha2, beta2, gamma, length);
        
        Vector.elementMul(deltaY, deriX1, deltaX1);
        Vector.elementMul(deltaY, deriX2, deltaX2);
        System.out.print("CPU[deltaX1]: "); Vector.println(deltaX1, 0, 10);
        System.out.print("CPU[deltaX2]: "); Vector.println(deltaX2, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor[] tdeltaX = eg.div_deltaX(true, tdeltaY, tX1, alpha1, beta1, tX2, alpha2, beta2);
//        Tensor[] tdeltaX = eg.quadratic2_deltaX(tdeltaY, tX1, tX2, alpha1, beta1, alpha2, beta2, gamma);
        
        
        float[] deltaX1_gpu = eg.valueOf(tdeltaX[0]);
        float[] deltaX2_gpu = eg.valueOf(tdeltaX[1]);
        
        System.out.print("GPU[deltaX1]: "); Vector.println(deltaX1_gpu, 0, 10);
        System.out.print("GPU[deltaX2]: "); Vector.println(deltaX2_gpu, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(deltaX1, deltaX1_gpu); System.out.println("sp1:" + sp1);
        float sp2 = Vector.samePercentRelative(deltaX2, deltaX2_gpu); System.out.println("sp2:" + sp2);
       
        //delete----------------------------------------------------------------
        eg.delete(tdeltaY, tX1, tX2);
        eg.delete(tdeltaX);
        
        if(sp1 < 0.99 || sp2 < 0.99) throw new RuntimeException();
    }
    
     public static void testSpeed(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X1 = Vector.randomFloatVector(length);
        float[] X2 = Vector.randomFloatVector(length);
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);

        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor[] tdeltaX = eg.quadratic2_deltaX(tdeltaY, tX1, tX2, 1, 2, 3, 4, 5);
            Tensor[] tdeltaX = eg.div_deltaX(false, tdeltaY, tX1, 1, 2, tX2, 1, 2);
            eg.delete(tdeltaX);
        }
        timer.record();
        
        float time = (float) timer.timeStampDifMills() / nIter;
	int data_size = (lengthv) * 4 * 5;
        
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        
        eg.delete(tdeltaY, tX1, tX2);
    }
    
    public static void main(String[] args)
    {
        
//        for(int h=1; h<=10; h++)
//            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
        testSpeed(1024, 1024);
    }
}

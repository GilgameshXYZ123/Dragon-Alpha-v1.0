package test.cuda;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
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
public class Cuda_function_deltaX_test1 
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
        
        float[] Y = Vector.randomFloatVector(length, -5, 5);  Vector.println("Y = ", Y, 0, 10);
        Tensor tY = eg.tensor(Y, height, width);
        
        float[] deltaY = Vector.randomFloatVector(length, 0, 1); Vector.println("deltaY = ", deltaY, 0, 10);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        float alpha = exr.nextFloat(0, 0.5f); System.out.println("alpha = " + alpha);
        float beta = exr.nextFloat(0, 0.5f); System.out.println("beta = " + beta);
        float gamma = exr.nextFloat(1, -1);
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        float[] deriY = new float[length];
        float[] deltaX1 = new float[length];

//        Vector.quadratic_Deri(Y, alpha, beta, deriY, length);
//        Vector.rpl_Deri(Y, alpha, beta, deriY, length);

//        Vector.abs_Deri(Y, alpha, deriY, length);
//        Vector.relu_deri(Y, deriY, length);
//        Vector.leakyRelu_Deri(Y, k, deriY, length);
//         Vector.leakyRelu_Deri(Y, k, deriY, length);
//        Vector.elu_Deri(Y, alpha, k, deriY, length);
//        Vector.softPlus_Deri(Y, deriY, length);
//        Vector.ln_Deri(alpha, Y, beta, deriY, length);

//        Vector.sqrt_Deri(alpha, Y, beta, deriY, length);
//        Vector.sqrt_Deri(Y, alpha, deriY, length);

//        Vector.tanh_Dri(Y, deriY, length);
        Vector.sigmoid_Deri(Y, deriY, length);

//        Vector.sin_Deri(Y, alpha, beta, deriY, length);
//        Vector.tan_Deri(alpha, Y, beta, deriY, length);
//        Vector.cot_Deri(alpha, Y, beta, deriY, length);
//        Vector.halfSin_Deri(alpha*alpha, Y, deriY, length);
//        Vector.asin_Deri(alpha, Y, beta, deriY, length);
//        Vector.atan_Deri(alpha, Y, beta, deriY, length);
        
        Vector.elementMul(deltaY, deriY, deltaX1);
        System.out.print("CPU:  "); Vector.println(deltaX1, 0, 10);
        
        //GPU-------------------------------------------------------------------
//        tY = eg.cot(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.cot_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.cot_deltaX(true, tdeltaY, tY, alpha);
        
//        tY = eg.tan(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.tan_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.tan_deltaX(true, tdeltaY, tY, alpha);
        
//        tY = eg.arctan(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.arctan_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.arctan_deltaX(true, tdeltaY, tY, alpha);
        
//        tY = eg.arcsin(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.arcsin_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.arcsin_deltaX(true, tdeltaY, tY, alpha);
        
//        tY = eg.sqrt(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.sqrt_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.sqrt_deltaX(true, tdeltaY, tY, alpha);
                
//        tY = eg.log(true, alpha, tY, beta);
//        Tensor tdeltaX1 = eg.log_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.log_deltaX(false, tdeltaY, tY, alpha);
        
//        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
//        Tensor tdeltaX1 = eg.quadratic_deltaX(false, tdeltaY, tY, alpha, beta);
//        Tensor tdeltaX2 = eg.quadratic_deltaX(true, tdeltaY, tY, alpha, beta);
        
//        tY = eg.rpl(true, alpha, tY, beta, gamma).c();
//        Tensor tdeltaX1 = eg.rpl_deltaX(false, tdeltaY, tY, alpha, gamma);
//        Tensor tdeltaX2 = eg.rpl_deltaX(true, tdeltaY, tY, alpha, gamma);
        
//        Tensor tdeltaX1 = eg.abs_deltaX(false, tdeltaY, tY, alpha);
//        Tensor tdeltaX2 = eg.abs_deltaX(true, tdeltaY, tY, alpha);
        
//        Tensor tdeltaX1 = eg.relu_deltaX_v1(false, tdeltaY, tY);
//        Tensor tdeltaX2 = eg.relu_deltaX_v1(true, tdeltaY, tY);
        
//        Tensor tdeltaX1 = eg.sin_deltaX(false, tdeltaY, tY, alpha, beta);
//        Tensor tdeltaX2 = eg.sin_deltaX(true, tdeltaY, tY, alpha, beta);

//        Tensor tdeltaX1 = eg.leakyRelu_deltaX_v1(false, tdeltaY, tY, k);
//        Tensor tdeltaX2 = eg.leakyRelu_deltaX_v1(true, tdeltaY, tY, k);

//        Tensor tdeltaX1 = eg.elu_deltaX_v1(false, tdeltaY, tY, alpha, k);
//        Tensor tdeltaX2 = eg.elu_deltaX_v1(true, tdeltaY, tY, alpha, k);

//        Tensor tdeltaX1 = eg.softplus_deltaX_v1(false, tdeltaY, tY);
//        Tensor tdeltaX2 = eg.softplus_deltaX_v1(true, tdeltaY, tY);

//        Tensor tdeltaX1 = eg.tanh_deltaX_v1(false, tdeltaY, tY);
//        Tensor tdeltaX2 = eg.tanh_deltaX_v1(true, tdeltaY, tY);

        Tensor tdeltaX1 = eg.sigmoid_deltaX_v1(false, tdeltaY, tY);
        Tensor tdeltaX2 = eg.sigmoid_deltaX_v1(true, tdeltaY, tY);
        
//        Tensor tdeltaX1 = eg.halfSin_deltaX(false, tdeltaY, tY, alpha, alpha);
//        Tensor tdeltaX2 = eg.halfSin_deltaX(true, tdeltaY, tY, alpha, alpha);
        
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        System.out.print("GPU1: "); Vector.println(deltaX2, 0, 10);
        
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        System.out.print("GPU2: "); Vector.println(deltaX3, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(deltaX1, deltaX2); System.out.println("sp1:" + sp1);
        float sp2 = Vector.samePercentRelative(deltaX1, deltaX3); System.out.println("sp2:" + sp2);
       
        //delete----------------------------------------------------------------
        eg.delete(tY, tdeltaX2, tdeltaX1);
        
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
        
        float[] Y = Vector.randomFloatVector(length);
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);

        Tensor tY = eg.tensor(Y, height, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
            tdeltaY = eg.sqrt_deltaX(true, tdeltaY, tY, 1);
//            tdeltaY = eg.cot_deltaX(true, tdeltaY, tY, 1);
//            tdeltaY = eg.tan_deltaX(true, tdeltaY, tY, 1);
//            tdeltaY = eg.arctan_deltaX(true, tdeltaY, tY, 1);
//            tdeltaY = eg.arcsin_deltaX(true, tdeltaY, tY, 1);
//            tdeltaY = eg.sqrt_deltaX(true, tdeltaY, tY, i);
//            tdeltaY = eg.log_deltaX(true, tdeltaY, tY, i);
//            tdeltaY = eg.quadratic_deltaX(true, tdeltaY, tY, 1, 2);
//            tdeltaY = eg.sin_deltaX(true, tdeltaY, tY, 1.0f, 1.0f);
//            tdeltaY = eg.abs_deltaX(true, tdeltaY, tY, 1.0f);
//            tdeltaY = eg.relu_deltaX(true, tdeltaY, tY);
//            tdeltaY = eg.leakyRelu_deltaX(true, tdeltaY, tY, 0.01f);
//            tdeltaY = eg.elu_deltaX(true, tdeltaY, tY, 1.0f, 0.01f);
//            tdeltaY = eg.softplus_deltaX(true, tdeltaY, tY);
//            tdeltaY = eg.tanh_deltaX(true, tdeltaY, tY);
//            tdeltaY = eg.sigmoid_deltaX(true, tdeltaY, tY);
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        float time = (float) timer.timeStampDifMills() / nIter;
	int data_size = (lengthv) * 4 * 3;
        
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        
        eg.delete(tdeltaY, tY);
    }
    public static void main(String[] args)
    {
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
        testSpeed(1024, 1024);
    }

}

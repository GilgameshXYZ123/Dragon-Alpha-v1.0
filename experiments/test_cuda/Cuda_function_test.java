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
public class Cuda_function_test 
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
        
        float[] X = Vector.randomFloatVector(length, -1f, 1f); Tensor tX = eg.tensor(X, height, width);
        float[] X2 = Vector.randomFloatVector(length, -1f, 1f); Tensor tX2 = eg.tensor(X2, height, width);
        
        float[] deltaY = Vector.randomFloatVector(length, -1, 1);
        
        float alpha = exr.nextFloat(0, 0.5f), alpha2 = exr.nextFloat();
        float beta = exr.nextFloat(1, -1), beta2 = exr.nextFloat();
        float gamma = exr.nextFloat();
        
        float vmin = exr.nextFloat(0, -1.5f);
        float vmax = exr.nextFloat(0, 1.5f);
        float k = exr.nextFloat();
        
        //CPU-------------------------------------------------------------------
        
//        Vector.linear(alpha, X, beta, Y1, length);
//        float[] Y1 = Vector.rpl(alpha, X, beta, gamma);
//        Vector.div(alpha, X, beta, alpha2, X2, beta2, gamma, Y1, length);
//         Vector.binomial(X, X2, alpha, beta, alpha2, beta2, gamma, k, Y1, length);

//        Vector.quadratic(X, alpha, beta, gamma, Y1, length);
//        Vector.sign(Y1, Y1, length);
//        Vector.ceil(Y1, Y1, length);
//        Vector.floor(Y1, Y1, length);
//        Vector.abs(Y1, Y1, length);

//        Vector.min(Y1, vmin, Y1, length);
//        Vector.max(Y1, vmax, Y1, length);
//        Vector.clip(Y1, vmin, vmax, Y1, length);
        
//        Vector.relu(X, Y1, length);
        float[] Y1 = Vector.leakyRelu(X, k);
//        Vector.elu(X, alpha, k, Y1, length);
//        Vector.softplus(Y1, X, length);
//        Vector.tanh(X, Y1, length);
//        Vector.sigmoid(X, Y1, length);

//        Vector.L1(X, X2, Y1);
//        Vector.L1_deltaYh(X, X2, Y1);
//        Vector.L2(X, X2, Y1);
//        Vector.L2_deltaYh(X, X2, Y1);
//        Vector.sigmoid(X, X, length);
//        Vector.crossEntropy(X, X2, Y1);
//        Vector.crossEntropy_deltaYh(X, X2, Y1);
//        Vector.smoothL1(X, X2, Y1);
//        Vector.smoothL1_deltaYh(X, X2, Y1);
//        Vector.balancedCrossEntropy(X, X2, alpha, beta, Y1);
//        Vector.balancedCrossEntropy_deltaYh(X, X2, alpha, beta, Y1);

//        Vector.sin(alpha, X, beta, Y1, length);
//        Vector.cos(alpha, X, beta, Y1, length);
//        Vector.cos(X, Y1, length);
//        Vector.halfSin(alpha, alpha, X, beta, Y1, length);
//        Vector.sigmoid(X, X, length);  Vector.crossEntropy(X, X2, Y1);
        
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        
//        Vector.println(X);
//        Vector.println(X2);
        
        //GPU-------------------------------------------------------------------
       
//        System.out.println(tX.lengthv() + ":" + tX2.lengthv());
        
//        Tensor tY1 = eg.linear(false, alpha, tX, beta).sync();
//        Tensor tY2 = eg.linear(true, alpha, tX, beta).sync();
//        Tensor tY1 = eg.quadratic(false, tX, alpha, beta, gamma);
//        Tensor tY2 = eg.quadratic(true, tX, alpha, beta, gamma);
//        Tensor tY1 = eg.rpl(false, alpha, tX, beta, gamma);
//        Tensor tY2 = eg.rpl(true, alpha, tX, beta, gamma);

//        Tensor tY1 = eg.div(false, alpha, tX, beta, alpha2, tX2, beta2, gamma);
//        Tensor tY2 = eg.div(eg.empty(1, height, width), alpha, tX, beta, alpha2, tX2, beta2, gamma);
//        Tensor tY1 = eg.binomial(tX, tX2, alpha, beta, alpha2, beta2, gamma, k);
//        Tensor tY2 = eg.binomial(eg.tensor(1, height, width), tX, tX2, alpha, beta, alpha2, beta2, gamma, k);
        
//        Tensor tY1 = eg.sign(false, alpha, tX, beta).sync();
//        Tensor tY2 = eg.sign(true, alpha, tX, beta).sync();
//        Tensor tY1 = eg.ceil(false, alpha, tX, beta).sync();
//        Tensor tY2 = eg.ceil(true, alpha, tX, beta).sync();
//        Tensor tY1 = eg.floor(false, alpha, tX, beta).sync();
//        Tensor tY2 = eg.floor(true, alpha, tX, beta).sync();
//        Tensor tY1 = eg.abs(false, alpha, tX, beta).sync();
//        Tensor tY2 = eg.abs(true, alpha, tX, beta).sync();

//        Tensor tY1 = eg.min(false, alpha, tX, beta, vmin).sync();
//        Tensor tY2 = eg.min(true, alpha, tX, beta, vmin).sync();
//        Tensor tY1 = eg.max(false, alpha, tX, beta, vmax).sync();
//        Tensor tY2 = eg.max(true, alpha, tX, beta, vmax).sync();
//        Tensor tY1 = eg.clip(false, alpha, tX, beta, vmin, vmax).sync();
//        Tensor tY2 = eg.clip(true, alpha, tX, beta, vmin, vmax).sync();

//        Tensor tY1 = eg.relu(false, tX).c();
//        Tensor tY2 = eg.relu(true, tX).c();
        Tensor tY1 = eg.leakyRelu(false, tX, k).c();
        Tensor tY2 = eg.leakyRelu(true, tX, k).c();
//        Tensor tY1 = eg.elu(false, tX, alpha, k).sync();
//        Tensor tY2 = eg.elu(true, tX, alpha, k).sync();
//        Tensor tY1 = eg.softplus(false, tX).sync();
//        Tensor tY2 = eg.softplus(true, tX).sync();
//        Tensor tY1 = eg.tanh(false, tX).sync();
//        Tensor tY2 = eg.tanh(true, tX).sync();
//        Tensor tY1 = eg.sigmoid(false, tX).c();
//        Tensor tY2 = eg.sigmoid(true, tX).c();
        
//        Tensor tY1 = eg.L1(tX, tX2);
//        Tensor tY1 = eg.L1_deltaYh(tX, tX2);
//        Tensor tY1 = eg.L2(tX, tX2);
//        Tensor tY1 = eg.L2_deltaYh(tX, tX2);
//        Tensor tY1 = eg.crossEntropy(tX, tX2);       
//        Tensor tY1 = eg.crossEntropy_deltaYh(tX, tX2);
//        Tensor tY1 = eg.balancedCrossEntropy(tX, tX2, alpha, beta);
//        Tensor tY1 = eg.balancedCrossEntropy_deltaYh(tX, tX2, alpha, beta);
//        Tensor tY1 = eg.smoothL1(tX, tX2);
//        Tensor tY1 = eg.smoothL1_deltaYh(tX, tX2);
//        Tensor tY1 = eg.sigmoidCrossEntropy(tX, tX2);
//        Tensor tY1 = eg.sigmoidCrossEntropy_deltaX(tX, tX2);


//        Tensor tY1 = eg.sin(true, alpha, tX, beta);
//        Tensor tY1 = eg.cos(true, alpha, tX, beta);
//        Tensor tY1 = eg.cos(true, tX);
//        Tensor tY1 = eg.halfSin(true, alpha, alpha, tX, beta);
        

        //compare---------------------------------------------------------------
        float[] Y2 = eg.valueOf(tY1);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        float sp1 = Vector.samePercentAbsolute(Y1, Y2); System.out.println("sp1:" + sp1);
        if(sp1 < 0.99) throw new RuntimeException();
        
        float[] Y3 = eg.valueOf(tY2);
        System.out.print("GPU1: "); Vector.println(Y3, 0, 10);
        float sp2 = Vector.samePercentRelative(Y2, Y3); System.out.println("sp2:" + sp2);
        if(sp2 < 0.99) throw new RuntimeException();
        //delete----------------------------------------------------------------
        eg.delete(tX, tX2, tY1);
        
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
        
        float[] X1 = Vector.randomFloatVector(length);
        float[] X2 = Vector.randomFloatVector(length);
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X1, 1, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
//            Tensor tY = eg.quadratic(true, tX, 1, 2, 3);
            
//            tX = eg.linear(true, 1.0f, tX, 1.0f);
//            tX = eg.rpl(true, 1.0f, tX, 1.1f, 2.1f);
//            tX = eg.div(tX, 1.0f, tX, 1.0f, 2.0f, tX, 2.0f, 3f);
//            tX = eg.binomial(tX, tX, tX, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
            
//            tX = eg.sign(true, 1.0f, tX, 1.0f);
//            tX = eg.ceil(true, 1.0f, tX, 1.0f);
//            tX = eg.floor(true, 1.0f, tX, 1.0f);
//            tX = eg.abs(true, 1.0f, tX, 1.0f);
//            Tensor tY1 = eg.halfSin(true, 1, tX, 2);
            
            
//            tX = eg.min(true, 1.0f, tX, 1.0f, 0.5f);
//            tX = eg.max(true, 1.0f, tX, 1.0f, 0.5f);
//            tX = eg.clip(true, 1.0f, tX, 1.0f, 0.0f, 0.5f);
              
//            tX = eg.relu(true, tX);
//            tX = eg.leakyRelu(true, tX, 0.5f);
//            tX = eg.elu(true, tX, 1.0f, 0.5f);
//            tX = eg.softplus(true, tX);
//            tX = eg.tanh(true, tX);
//            tX = eg.sigmoid(true, tX);
            
//            Tensor tL = eg.L1(tX, tX2); eg.delete(tL);
//            Tensor tL = eg.L2(tX, tX2); eg.delete(tL);
//            Tensor tL = eg.crossEntropy(tX, tX2); eg.delete(tL);
//            Tensor tL = eg.smoothL1(tX, tX2); eg.delete(tL);
            
//            Tensor tY1 = eg.L1_deltaYh(tX, tX2); eg.delete(tY1);
//            Tensor tY1 = eg.L2_deltaYh(tX, tX2); eg.delete(tY1);
//            Tensor tY1 = eg.crossEntropy_deltaYh(tX, tX2); eg.delete(tY1);
//            Tensor tY1 = eg.smoothL1_deltaYh(tX, tX2); eg.delete(tY1);
//            Tensor tY1 = eg.balancedCrossEntropy_deltaYh(tX, tX2, 1, 2); eg.delete(tY1);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 2;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX1);
    }
    public static void main(String[] args)
    {
        try
        {
            for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
        
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
 
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_batchnorm_test6 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * 2 * 3 * stride;
        int length = height * 2 * 3 * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
//        float[] X = Vector.randomFloatVector(length, 0, 1f);
        float[] X = Vector.randomFloatVector(length, 0, 0.001f);
        float[] A = Vector.randomFloatVector(width, 0, 1f);
        float[] B = Vector.randomFloatVector(width, 0, 1f);
        float[] deltaY = Vector.randomFloatVector(length, 0, 1f);
        float eps = 1e-5f;
        
        Tensor tX = eg.tensor(X, height, 2, 3, width);
        Tensor tA = eg.tensor(A, width);
        Tensor tB = eg.tensor(B, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, 2, 3, width);
        
        //forward prop----------------------------------------------------------
        Tensor tX_mean = eg.field_mean(tX).c();
        Tensor tX_squareMean = eg.field_sqmean(tX).c();
        Tensor tY = eg.sqBatchNorm(false, tX, tX_mean, tX_squareMean, 1e-5f, tA, tB).c();
        
        //GPU1------------------------------------------------------------------
        Tensor[] grads1 = eg.sqBatchNorm_gradients_v1(false, tdeltaY, tY, tX_mean, tX_squareMean, eps, tA, tB);
        Tensor tdeltaX1 = grads1[0].c();
        Tensor tdeltaA1 = grads1[1].c();
        Tensor tdeltaB1 = grads1[2].c();
        
        //GPU2------------------------------------------------------------------
        Tensor[] grads2 = eg.sqBatchNorm_gradients_v2(false, tdeltaY, tX, tX_mean, tX_squareMean, eps, tA);
        Tensor tdeltaX2 = grads2[0].c();
        Tensor tdeltaA2 = grads2[1].c();
        Tensor tdeltaB2 = grads2[2].c();
        
        //compare---------------------------------------------------------------
        float[] dX1 = tdeltaX1.value(), dX2 = tdeltaX2.value();
        float[] dA1 = tdeltaA1.value(), dA2 = tdeltaA2.value();
        float[] dB1 = tdeltaB1.value(), dB2 = tdeltaB2.value();
        
        float sp0 = Vector.samePercentRelative(dX1, dX2);
        float sp1 = Vector.samePercentRelative(dA1, dA2);
        float sp2 = Vector.samePercentRelative(dB1, dB2);
        
        Vector.println("dX1 = ", dX1, 0, 10);
        Vector.println("dX2 = ", dX2, 0, 10);
        
        Vector.println("dA1 = ", dA1, 0, 10);
        Vector.println("dA2 = ", dA2, 0, 10);
        
        Vector.println("dB1 = ", dB1, 0, 10);
        Vector.println("dB2 = ", dB2, 0, 10);
        
        System.out.println("sp0(X) = " + sp0 + ", " + tdeltaX1.sum() + ", " + tdeltaX2.sum());
        System.out.println("sp1(A) = " + sp1);
        System.out.println("sp2(B) = " + sp2);
        
        if(sp0 < 0.95f || sp1 < 0.9f || sp2 < 0.95f) throw new RuntimeException();
       
        if(tdeltaX1.hasNan().get()) throw new RuntimeException();
        if(tdeltaX2.hasNan().get()) throw new RuntimeException();
       
        if(tdeltaA1.hasNan().get()) throw new RuntimeException();
        if(tdeltaA2.hasNan().get()) throw new RuntimeException();
          
        if(tdeltaB1.hasNan().get()) throw new RuntimeException();
        if(tdeltaB2.hasNan().get()) throw new RuntimeException();
        
        tdeltaX1.delete(); tdeltaX1.delete();
        tdeltaA1.delete(); tdeltaA1.delete();
        tdeltaB1.delete(); tdeltaB1.delete();
        
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        //20.559
        try
        {
            testCorrect(8, 4);
            
//            for(int h=1; h<=10; h++)
//                for(int w=1; w<=256; w++) testCorrect(h, w);
            
            for(int h=30; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
            testCorrect(512*8*8, 256);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

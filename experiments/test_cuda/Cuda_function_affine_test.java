/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static test.cuda.Cuda_function_batchnorm_test.eg;
import static test.cuda.Cuda_function_batchnorm_test.testCorrect;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_affine_test 
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
        
        float[] X = Vector.randomFloatVector(length, 0, 1f);
        float[] X_mean = Vector.randomFloatVector(2 * 3 * width, 0, 1f);
        float[] X_square_mean = Vector.randomFloatVector(2 * 3 * width, 1, 2f);
        float[] A = Vector.randomFloatVector(2 * 3 * width, 0, 1f);
        float[] B = Vector.randomFloatVector(2 * 3 * width);
        float[] deltaY = Vector.randomFloatVector(length, 0, 1f);
        
        Tensor tX = eg.tensor(X, height, 2, 3, width);
        Tensor tX_mean = eg.tensor(X_mean, 2, 3, width);
        Tensor tX_square_mean = eg.tensor(X_square_mean, 2, 3, width);
        Tensor tA = eg.tensor(A, 2, 3, width);
        Tensor tB = eg.tensor(B, 2, 3, width);
        Tensor tdeltaY = eg.tensor(deltaY, height, 2, 3, width);
        float eps = 1e-5f;
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.sqBatchNorm(false, tX, tX_mean, tX_square_mean, eps, tA, tB).c();
        Tensor tdeltaA1 = eg.sqBatchNorm_deltaA_v1(tdeltaY, tY, tA, tB).c();
        Tensor tdeltaA2 = eg.sqBatchNorm_deltaA_v2(tdeltaY, tX, tX_mean, tX_square_mean, eps).c();
        Tensor tdeltaB1 = eg.field_sum(tdeltaY, 2*3*width);
        
        Tensor[] delta1 = eg.sqBatchNorm_deltaAB_v1(tdeltaY, tY, tA, tB);
        Tensor[] delta2 = eg.sqBatchNorm_deltaAB_v2(tdeltaY, tX, tX_mean, tX_square_mean, eps);
        Tensor tdeltaA3 = delta1[0], tdeltaB2 = delta1[1];
        Tensor tdeltaA4 = delta2[0], tdeltaB3 = delta2[1];

        //compare Y-------------------------------------------------------------
        float[] deltaA1 = tdeltaA1.value();
        float[] deltaA2 = tdeltaA2.value();
        float[] deltaA3 = tdeltaA3.value();
        float[] deltaA4 = tdeltaA4.value();
        
        float spA1 = Vector.samePercentRelative(deltaA1, deltaA2, 1e-3f);
        float spA2 = Vector.samePercentRelative(deltaA1, deltaA3, 1e-3f);
        float spA3 = Vector.samePercentRelative(deltaA1, deltaA4, 1e-3f);
        
        float[] deltaB1 = tdeltaB1.value();
        float[] deltaB2 = tdeltaB2.value();
        float[] deltaB3 = tdeltaB3.value();
                 
        float spB1 = Vector.samePercentRelative(deltaB1, deltaB2, 1e-3f);
        float spB2 = Vector.samePercentRelative(deltaB1, deltaB3, 1e-3f);
        
        System.out.print("GPU - deltaA1: "); Vector.println(deltaA1, 0, 10);
        System.out.print("GPU - deltaA2: "); Vector.println(deltaA2, 0, 10);
        System.out.print("GPU - deltaA3: "); Vector.println(deltaA3, 0, 10);
        System.out.print("GPU - deltaA4: "); Vector.println(deltaA4, 0, 10);
        System.out.println("tdeltaA1 = " + tdeltaA1.sum());
        System.out.println("tdeltaA2 = " + tdeltaA2.sum());
        System.out.println("tdeltaA3 = " + tdeltaA3.sum());
        System.out.println("tdeltaA4 = " + tdeltaA4.sum());
        
        System.out.println("spA1:" + spA1);     
        System.out.println("spA2:" + spA2);
        System.out.println("spA3:" + spA3);     
        
        System.out.print("GPU - deltaB1: "); Vector.println(deltaB1, 0, 10);
        System.out.print("GPU - deltaB2: "); Vector.println(deltaB2, 0, 10);
        System.out.print("GPU - deltaB3: "); Vector.println(deltaB3, 0, 10);
        
        System.out.println("spB1:" + spB1);     
        System.out.println("spB2:" + spB2);
        
        //delete---------------------------------------------------------------
        eg.delete(tX, tX_mean, tX_square_mean, tA, tB, tdeltaY);   
        eg.delete(tdeltaA1, tdeltaA2, tdeltaA3, tdeltaA4);
        eg.delete(tdeltaB1, tdeltaB2, tdeltaB3);
        
        if(spA1 < 0.99f) throw new RuntimeException();
        if(spA2 < 0.99f) throw new RuntimeException();
        if(spA3 < 0.99f) throw new RuntimeException();
            
        if(spB1 < 0.99f) throw new RuntimeException();
        if(spB2 < 0.99f) throw new RuntimeException();
    }

    public static void main(String[] args)
    {
        
//        Vector.PRINT_DIFFERENT = true;
        
        try
        {
            for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

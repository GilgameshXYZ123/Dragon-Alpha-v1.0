package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
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
public class Cuda_function_batchnorm_test 
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
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, 2 * 3 * width);
        float[][] mY = new float[height][2 * 3 * width];
         
        float[][] mdeltaY = Matrix.toMatrix(deltaY, 2 * 3 *width);
        float[] deltaA1 = new float[2 * 3 * width];
        
        Matrix.batchNorm(mX, X_mean, X_square_mean, A, B, mY, height, 2 * 3 * width);
        Matrix.batchNorm_deltaA(mdeltaY, mY, A, B, height, 2 * 3 * width, deltaA1);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.sqBatchNorm(false, tX, tX_mean, tX_square_mean, eps, tA, tB).c();
//        Tensor tdeltaA1 = eg.batchNorm_deltaA_v1(tdeltaY, tY, tA, tB).c();
        Tensor tdeltaA2 = eg.sqBatchNorm_deltaA_v2(tdeltaY, tX, tX_mean, tX_square_mean, eps).c();
        Tensor tdeltaA1 = eg.sqBatchNorm_gradients_v1(false, tdeltaY, tY, tX_mean, tX_square_mean, eps, tA, tB)[1].c();

        //compare Y-------------------------------------------------------------
        float[] Y1 = Matrix.toVector(mY); 
        float[] Y2 = eg.valueOf(tY);
        float spY = Vector.samePercentRelative(Y1, Y2, 1e-3f); 
        
        System.out.print("CPU - Y: "); Vector.println(Y1, 0, 10);
        System.out.print("GPU - Y: "); Vector.println(Y2, 0, 10);
        System.out.println("spY:" + spY);
        
        //compare deltaA--------------------------------------------------------
        float[] deltaA2 = tdeltaA1.value();
        float[] deltaA3 = tdeltaA2.value();
        float spA1 = Vector.samePercentRelative(deltaA1, deltaA2, 1e-3f);
        float spA2 = Vector.samePercentRelative(deltaA1, deltaA3, 1e-3f);
//        
        System.out.print("CPU - deltaA : "); Vector.println(deltaA1, 0, 10);
        System.out.print("GPU - deltaA1: "); Vector.println(deltaA2, 0, 10);
        System.out.print("GPU - deltaA2: "); Vector.println(deltaA3, 0, 10);
        System.out.println("spA1:" + spA1);     
        System.out.println("spA2:" + spA2);
        
        //delete---------------------------------------------------------------
        eg.delete(tX, tX_mean, tX_square_mean, tA, tB);   
        eg.delete(tX, tdeltaA1, tdeltaA2, tdeltaY, tA, tB);   
        
        if(spY < 0.95f) throw new RuntimeException();
//        if(spA1 < 0.99f) throw new RuntimeException();
//        if(spA2 < 0.99f) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        
        Vector.PRINT_DIFFERENT = true;
        
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

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
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_batchnorm_test3 
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
        
        //CPU--------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, 2 * 3 * width);
        float[][] mY = new float[height][2 * 3 *width];
        float[][] mdeltaY = Matrix.toMatrix(deltaY, 2 * 3 *width);
        
        float[][] mdeltaX1 = new float[height][2 * 3 * width];
        Matrix.batchNorm_deltaX1(mdeltaY, mX, X_mean, X_square_mean, eps, mdeltaX1, height, 2 * 3 * width);
        
        float[][] mdeltaX2 = new float[height][2 * 3 * width];
        Matrix.batchNorm_deltaX2(mdeltaY, mX, X_mean, X_square_mean, eps, mdeltaX2, height, 2 * 3 * width);
     
        //compare---------------------------------------------------------------
        float[] dX0 = Matrix.toVector(mdeltaX1);
        float[] dX1 = Matrix.toVector(mdeltaX2);
        
        Vector.println("dX0 = ", dX0, 0, 10);
        Vector.println("dX1 = ", dX1, 0, 10);
        
        float sp0 = Vector.samePercentRelative(dX0, dX1, 1e-3f);
        float zp0 = Vector.zeroPercent(dX0);
        float zp1 = Vector.zeroPercent(dX1);
        
        System.out.println("sp1 = " + sp0);
        System.out.println("zp0 = " + zp0);
        System.out.println("zp1 = " + zp1);
        
        if(sp0 < 0.9f) throw new RuntimeException();
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
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

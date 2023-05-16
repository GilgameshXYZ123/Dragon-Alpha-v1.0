package test.cuda;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
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
public class Cuda_function_layernorm_test 
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
        
        float[] X = Vector.randomFloatVector(length, 0, 1f);
        float[] X_mean = Vector.randomFloatVector(height, 0, 1f);
        float[] X_square_mean = Vector.randomFloatVector(height, 1, 2f);
        float[] A = Vector.randomFloatVector(width, 0, 1f);
        float[] B = Vector.randomFloatVector(width);
        
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX_mean = eg.tensor(X_mean, height);
        Tensor tX_square_mean = eg.tensor(X_square_mean, height);
        Tensor tA = eg.tensor(A, width);
        Tensor tB = eg.tensor(B, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
        float[][] mY = new float[height][width];
         
        Matrix.layerNorm(mX, X_mean, X_square_mean, A, B, mY, height, width);
//        Matrix.layerNorm_deltaX(mX, X_mean, X_square_mean, A, mY, height, width);

        //GPU-------------------------------------------------------------------
        Tensor tY = eg.layerNorm(false, tX, tX_mean, tX_square_mean, 1e-5f, tA, tB).c();
//        Tensor tY = eg.layerNorm_deltaX(true, tX, tX_mean, tX_square_mean, 1e-5f, tA).c();

        //compare---------------------------------------------------------------
        float[] Y1 = Matrix.toVector(mY); 
        System.out.print("CPU: "); Vector.println(Y1, 0, 10);
        
        float[] Y2 = eg.valueOf(tY);
        System.out.print("GPU: "); Vector.println(Y2, 0, 10);
        
        float sp0 = Vector.samePercentAbsolute(Y1, Y2); System.out.println("spW:" + sp0);
        if(sp0 < 0.99) throw new RuntimeException();
      
        eg.delete(tX, tX_mean, tX_square_mean, tA);   
    }
    
    public static void main(String[] args)
    {
        try
        {
            for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
              
            testCorrect(1024, 1024);  
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

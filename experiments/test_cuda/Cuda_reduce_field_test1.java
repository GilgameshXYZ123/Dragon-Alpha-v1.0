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
public class Cuda_reduce_field_test1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();

    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length  = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
        
//        float[][] Ys = Matrix.field_var(mX);
        float[][] Ys = Matrix.field_stddev(mX);
        
        float[] var1 = Ys[0];
        float[] mean1 = Ys[1];
        float[] squareMean1 = Ys[2];
        
        //GPU-------------------------------------------------------------------
//        Tensor[] tYs = eg.field_var_mean_squareMean(tX, width);
        Tensor[] tYs = eg.field_std_mean_sqmean(tX, width);
        
        float[] var2 = tYs[0].value();
        float[] mean2 = tYs[1].value();
        float[] squareMean2 = tYs[2].value();
        
        //compare---------------------------------------------------------------
        
        float sp1 = Vector.samePercentRelative(var1, var2);
        float sp2 = Vector.samePercentRelative(mean1, mean2);
        float sp3 = Vector.samePercentRelative(squareMean1, squareMean2);
        
        System.out.println("sp(var) = " + sp1);
        Vector.println("var1 = ", var1, 0, 10);
        Vector.println("var2 = ", var2, 0, 10);
        
        System.out.println("sp(mean) = " + sp2);
        Vector.println("mean1 = ", mean1, 0, 10);
        Vector.println("mean2 = ", mean2, 0, 10);
        
        System.out.println("sp(squareMean) = " + sp3);
        Vector.println("squareMean1 = ", squareMean1, 0, 10);
        Vector.println("squareMean2 = ", squareMean2, 0, 10);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        if(sp3 < 0.99) throw new RuntimeException();
        
        if(Float.isNaN(eg.straight_sum(tYs[0]).get())) throw new RuntimeException();
      
        eg.delete(tYs);
        eg.delete(tX);
    }
  
    public static void main(String[] args)
    {
        try
        {
            for(int h=5; h<=20; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
////            
            Vector.PRINT_DIFFERENT = true;
            for(int h=100; h<=105; h++)
                for(int w= 128; w<=256; w++) testCorrect(h, w);
        
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

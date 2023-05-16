/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static test.cuda.Cuda_reduce_field_test.testCorrect;
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
public class Cuda_reduce_field_test3 
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
        Vector.println(X, 0, 10);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
//        Object[] result1 = Matrix.field_max_indexed(mX);
        Object[] result1 = Matrix.field_min_indexed(mX);
        
        float[] Y1 = (float[]) result1[0];
        int[] Index1 = (int[]) result1[1];
        
        //GPU-------------------------------------------------------------------
//        Tensor[] result2 = eg.field_max_indexed(tX) ;
        Tensor[] result2 = eg.field_min_indexed(tX);
        
        float[] Y2 = result2[0].value();
        int[] Index2 = result2[1].value_int32();
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(Y1, Y2);
        float sp2 = Vector.samePercentAbsolute(Index1, Index2);
        
        System.out.println("sp(Y) = " + sp1);
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        
        System.out.println("sp(Index) = " + sp2);
        Vector.println("Index1 = ", Index1, 0, 10);
        Vector.println("Index2 = ", Index2, 0, 10);
        
        if(sp1 < 0.99) throw new RuntimeException();
        if(sp2 < 0.99) throw new RuntimeException();
        
        if(result2[0].hasNan().get()) throw new RuntimeException();
      
        eg.delete(result2);
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
        
            for(int h=1024; h<=1028; h++)
                for(int w= 233; w<=256; w++) testCorrect(h, w);
            
//            testSpeed(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

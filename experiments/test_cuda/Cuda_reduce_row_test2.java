package test.cuda;


import static test.cuda.Cuda_reduce_row_test.testCorrect;
import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
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
public class Cuda_reduce_row_test2 
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
        
        float[] X = Vector.randomFloatVector(length, -3f, 6f);
        Tensor tX = eg.tensor(X, height, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, width);
//        Object[] result1 = Matrix.row_max_indexed(mX);
        Object[] result1 = Matrix.row_min_indexed(mX);
        
        float[] Y1 = (float[]) result1[0];
        int[] Index1 = (int[]) result1[1];

        //GPU-------------------------------------------------------------------
//        Tensor[] result2 = eg.row_max_indexed(tX);
        Tensor[] result2 = eg.row_min_indexed(tX);
        
        float[] Y2 = result2[0].value();
        int[] Index2 = result2[1].value_int32();
        
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentAbsolute(Y1, Y2);
        float sp2 = Vector.samePercentAbsolute(Index1, Index2);
        
        System.out.println("sp(Y) = " + sp1);
        Vector.println("Y1 = ", Y1, 0, 10);
        Vector.println("Y2 = ", Y2, 0, 10);
        
        System.out.println("sp(Index) = " + sp2);
        Vector.println("Index1 = ", Index1, 0, 10);
        Vector.println("Index2 = ", Index2, 0, 10);
        
        if(sp1 < 0.99f) throw new RuntimeException();
        if(sp2 < 0.99f) throw new RuntimeException();
        if(result2[0].hasNan().get()) throw new RuntimeException();
      
        eg.delete(tX);
        eg.delete(result2);
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
//            (3, 1), (3, 2,
            for(int h = 1; h <= 20; h++)
                for(int w = 1; w <= 256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);

            for(int h=300; h<=305; h++)
                for(int w=7; w<=12; w++) testCorrect(h, w);            
            
            for(int h=300; h<=305; h++)
                for(int w= 140; w<=164; w++) testCorrect(h, w);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

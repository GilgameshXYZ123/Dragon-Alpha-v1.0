/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_equal_test2 
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
        
        int[] X1 = Vector.randomIntVector(length);
        int[] X2 = Vector.randomIntVector(length);
        
        Tensor tX1 = eg.tensor_int32(X1, height, width);
        Tensor tX2 = eg.tensor_int32(X2, height, width);
      
//        int vmin = exr.nextInt(0, 50);
//        int vmax = exr.nextInt(50, 100);
        int vmin = 0, vmax = 0;
        
        //CPU-------------------------------------------------------------------
        float[] Y1 = Vector.equal(X1, X1, vmin, vmax);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.equal_abs_int32(tX1, tX1, vmin, vmax);
        
        float[] Y2 = tY.value();
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(Y1, Y2, 0.01f); 
        
        System.out.println("sp1:" + sp1);
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        System.out.print("GPU : "); Vector.println(Y2, 0, 10);
         
        //delete----------------------------------------------------------------
        eg.delete(tX1, tX2, tY);
        if(sp1 < 0.99) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        try
        {
            for(int h=20; h<=30; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            
            for(int h=320; h<=330; h++)
                for(int w=234; w<=256; w++) testCorrect(h, w);
        
            testCorrect(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

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
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_equal_test1 
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
        
        float[] X1 = Vector.randomFloatVector(length, 0.5f, 1f); 
        float[] X2 = Vector.randomFloatVector(length, 0.5f, 1f);
        
        Tensor tX1 = eg.tensor(X1, height, width);
        Tensor tX2 = eg.tensor(X2, height, width);
      
//        float vmin = exr.nextFloat(0, 0.1f);
//        float vmax = exr.nextFloat(1, 0.2f);
        float vmin = 0, vmax = 0;
        
        //CPU-------------------------------------------------------------------
        float[] Y1 = Vector.equal(X1, X1, vmin, vmax);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.equal_abs(tX1, tX1, vmin, vmax);
        
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

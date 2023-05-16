/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import java.util.Arrays;
import static test.cuda.Cuda_reduce_straight_test.testCorrect;
import static test.cuda.Cuda_reduce_straight_test.testSpeed;
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

public class Cuda_reduce_straight_test1 {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width) 
    {
        int stride = (width + 3) >> 2 << 2;
        int length = height * width;
        int lengthv = height * stride;
        
        System.out.println("testCorrect: ");
        System.out.format("[length, lengthv] = [%d, %d]\n", length, lengthv);
        
        float[] X = Vector.randomFloatVector(length, -3, 0);
        Tensor tX = eg.tensor(X, length);
        
        //CPU-------------------------------------------------------------------
//        float[] y1 = Vector.var(X);
        float[] y1 = Vector.stddev(X);
        
        //GPU-------------------------------------------------------------------
//        float[] y2 = eg.straight_var_mean_squareMean(tX).get();
        float[] y2 = eg.straight_std_mean_sqmean(tX).get();
        
        //compare---------------------------------------------------------------
        System.out.println("CPU: " + Arrays.toString(y1));
        System.out.println("GPU: " + Arrays.toString(y2));
        
        float sp = Vector.samePercentRelative(y1, y2);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void main(String[] args) 
    {
        try
        {
            for(int h=1; h<=20; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
//            
             for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
        
            testSpeed(1024, 1024);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    } 
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_deltaX_dual_out_test1 
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
        
        float[] X = Vector.randomFloatVector(length, -1, 1); 
        float alpha1 = exr.nextFloat(0, 0.5f);
        float beta1 = exr.nextFloat(1, -1);
        float alpha2 = exr.nextFloat(1, -1);
        float beta2 = exr.nextFloat(0, -1.5f);
        
        Tensor tX = eg.tensor(X, height, width);
        
        //GPU1------------------------------------------------------------------
        Tensor tA1 = eg.linear(false, alpha1, tX, beta1).c();
        Tensor tB1 = eg.linear(false, alpha2, tX, beta2).c();
        
        //GPU2------------------------------------------------------------------
        Tensor[] outs = eg.linear_2out(true, tX, alpha1, beta1, alpha2, beta2);
        Tensor tA2 = outs[0];
        Tensor tB2 = outs[1];
        
        Cuda_expk2.checkMemAlign(tA2);
        Cuda_expk2.checkMemAlign(tB2);
         
        //compare---------------------------------------------------------------
        float[] A1 = tA1.value(), B1 = tB1.value();
        float[] A2 = tA2.value(), B2 = tB2.value();
        
        float sp1 = Vector.samePercentRelative(A1, A2); 
        float sp2 = Vector.samePercentRelative(B1, B2); 
        
        Vector.println("A1 = ", A1, 0, 10);
        Vector.println("A2 = ", A2, 0, 10);
        System.out.println();
        
        Vector.println("B1 = ", B1, 0, 10);
        Vector.println("B2 = ", B2, 0, 10);
        System.out.println();
        
        System.out.println("sp1:" + sp1);
        System.out.println("sp2:" + sp2);
       
        //delete----------------------------------------------------------------
        eg.delete(tA1, tA2, tB1, tB2);
        
        if(sp1 < 0.99 || sp2 < 0.99) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
    }
}

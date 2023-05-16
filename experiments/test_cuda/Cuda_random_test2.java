package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.lang.SimpleTimer;
import z.util.math.ExRandom;
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
public class Cuda_random_test2 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int length)
    {
        System.out.println("\nTest Corret:");
        System.out.format("(length) = %d\n", length);
        
        float[] X = Vector.randomFloatVector(length);
        Tensor tX = eg.tensor(X, length).c();
        
        //GPU-------------------------------------------------------------------
        Tensor[] outs = eg.bernouli_mul(tX, 0.3f, 0.9f, 0f);
        Tensor tY = outs[0];
        Tensor tR = outs[1];
     
        Cuda_expk2.checkMemAlign(tX);
        Cuda_expk2.checkMemAlign(tR);
        
        //compare---------------------------------------------------------------
        float[] Y = tY.value();
        float[] R = tR.value();
        
        float zp0 = Vector.zeroPercent(R);
        float zp1 = Vector.zeroPercent(Y);
        
        Vector.println("X = ", X, 0, 10);
        Vector.println("Y = ", Y, 0, 10);
        Vector.println("R = ", R, 0, 10);
        
        System.out.println("zp0:" + zp0);
        System.out.println("zp1:" + zp1);
        
        //delete----------------------------------------------------------------
        eg.delete(tX);
        eg.delete(outs);
        if(zp1 != zp0) throw new RuntimeException(); 
    }
    
  
    public static void main(String[] args)
    {
        for(int length = 1; length <= 2048; length ++) testCorrect(length);
                
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_summary_test 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int num, int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        Tensor[] Xs = new Tensor[num];
        for(int i=0; i<Xs.length; i++) Xs[i] = eg.Uniform(height, width);
        
        //way1------------------------------------------------------------------
        Tensor Y1 = eg.zeros(height, width);
        for(int i=0; i<num; i++) Y1 = eg.add(true, Y1, Xs[i]).c();
        
        //way2------------------------------------------------------------------
        Tensor Y2 = eg.summary(false, Xs).c();
        Tensor Y3 = eg.summary(true, Xs).c();
        
        //compare---------------------------------------------------------------
        float[] v1 = Y1.value();
        float[] v2 = Y2.value();
        float[] v3 = Y3.value();
        
        Vector.println("Y1: ", v1, 0, 10);
        Vector.println("Y2: ", v2, 0, 10);
        Vector.println("Y3: ", v3, 0, 10);
        
        float sp1 = Vector.samePercentRelative(v1, v2);
        float sp2 = Vector.samePercentRelative(v2, v3);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        System.out.println(sp1 + " : " + sp2);
        if(sp1 < 0.99 || sp2 < 0.99) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        for(int n = 2; n<=4; n++)   
        for(int h=1; h<=10; h++)
                for(int w=1; w<=256; w++) testCorrect(n, h, w);
    }
    
}

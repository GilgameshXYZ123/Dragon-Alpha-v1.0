package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
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
public class Cuda_reduce_field_test4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();

    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * 2 * 3 * stride;
        int length  = height * 2 * 3 * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -2f, 3f);
        Tensor tX = eg.tensor(X, height, 2, 3, width);
        
        float alpha1 = exr.nextFloat(-1f, 1f);
        float beta1 = exr.nextFloat(-1f, 1f);
        float alpha2 = exr.nextFloat(-1f, 1f);
        float beta2 = exr.nextFloat(-1f, 1f);
        float gamma2 = exr.nextFloat(-1f, 1f);
        
        //way1------------------------------------------------------------------
        Tensor A1 = eg.field_linear(tX, alpha1, beta1);
        Tensor B1 = eg.field_quadratic(tX, alpha2, beta2, gamma2);
        
        //way2------------------------------------------------------------------
        Tensor[] Y = eg.field_linear_quadratic(tX, alpha1, beta1, alpha2, beta2, gamma2);
        Tensor A2 = Y[0];
        Tensor B2 = Y[1];
        
        //compare---------------------------------------------------------------
        float[] a1 = A1.value();
        float[] a2 = A2.value();
        Vector.println("A1: ", a1, 0, 10);
        Vector.println("A2: ", a2, 0, 10);
        
        float[] b1 = B1.value();
        float[] b2 = B2.value();
        Vector.println("B1: ", b1, 0, 10);
        Vector.println("B2: ", b2, 0, 10);
        
        float spA = Vector.samePercentRelative(a1, a2);
        float spB = Vector.samePercentRelative(b1, b2);
        System.out.println("spA = " + spA);
        System.out.println("spB = " + spB);
        
        eg.delete(tX, A1, A2, B1, B2);
        if(spA < 0.99 || spB < 0.99) throw new RuntimeException();
    }

    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
            for(int h=1; h<=20; h++)
                for(int w=1; w<=256; w++) testCorrect(h, w);
            for(int h=100; h<=105; h++)
                for(int w= 128; w<=256; w++) testCorrect(h, w);
            for(int h=1024; h<=1028; h++)
                for(int w= 233; w<=256; w++) testCorrect(h, w);
            
            testCorrect(1024, 1024);//[512, 16, 16, 128]
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

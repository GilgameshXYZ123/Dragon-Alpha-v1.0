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
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_zeronan_test 
{
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
        
        float[] X = Vector.nan(length);
//        for(int i=0; i<4; i++) X[i] = 10;
        
        Vector.println(X, 0, 10);
        
        Tensor tX = eg.tensor(X, height, width);
        tX = eg.zero_nan(true, tX).c();
        float[] X2 = tX.value();
        Vector.println(X2, 0, 10);
        
        float zp = Vector.zeroPercent(X2);
        System.out.println("zp = " + zp);
        
        eg.delete(tX);
    }
    
       public static void testSpeed(int height, int width)
    {
        eg.check(false);
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Speed:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length);
        float[] X2 = Vector.randomFloatVector(length);
        Tensor tX = eg.tensor(X, height, width);
        Tensor tX2 = eg.tensor(X, 1, height, width);
        
        SimpleTimer timer = new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
            tX = eg.zero_nan(true, tX);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (lengthv) * 4 * 2;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tX);
    }
    
    
    public static void main(String[] args)
    {
//        for(int h=1; h<=100; h++)
//        for(int w=1; w<=100; w++) testCorrect(h, w);
        testCorrect(1024, 1024);
        testSpeed(1024, 1024);
//        float nan = Float.NaN;
//        float inf = Float.POSITIVE_INFINITY;
//        System.out.println((int)inf);
    }
}

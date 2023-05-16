package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
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
public class Cuda_random_test1 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int length)
    {
        System.out.println("\nTest Corret:");
        System.out.format("(length) = %d\n", length);
        float alpha = exr.nextFloat(-1, 0);
        float beta = exr.nextFloat() + 2;
        
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.Bernouli(0.3f, 0, 1f, length);
//        Tensor tX = eg.uniform(alpha, beta, length);
//        Tensor tX = eg.sparse_uniform(0.7f, alpha, beta, length);
//        Tensor tX = eg.gaussian(1, 2, length);
//        Tensor tX = eg.sparse_gaussian(0.7f, alpha, beta, length);
        
        float[] Y = eg.valueOf(tX);
        System.out.print("GPU1: "); Vector.println(Y, 0, 10);
        
        //compare---------------------------------------------------------------
        float zp1 = Vector.zeroPercent(Y); System.out.println("zp1:" + zp1);
        float sp1 = Vector.matchPercent(Y, (Float x)->{return x>=alpha && x<=beta;});
                
        //delete----------------------------------------------------------------
        eg.delete(tX);
//        if(zp1 > 0.51) throw new RuntimeException(); 
    }
    
    public static void testSpeed(int length)
    {
        eg.check(false);
        System.out.println("\nTest Corret:");
        System.out.format("(length) = %d\n", length);
        float alpha = exr.nextFloat();
        float beta = exr.nextFloat() + 1;
        Tensor tX = eg.empty(length);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i < nIter; i++)
        {
            //Tensor tX = eg.uniform(alpha, beta, length).sync(); eg.delete(tX);
            tX = eg.bernouli(tX, 0.3f, 0, 1f);
//           tX = eg.uniform(tX, alpha, beta);
//           tX = eg.sparse_uniform(tX, 0.4f, 1, 2);
//            tX = eg.gaussian(tX, 1, 2);
//            tX = eg.sparse_gaussian(tX, 0.6f, 1, 2);
        }
        Cuda.deviceSynchronize();
        timer.record();
        float time = (float) timer.timeStampDifMills()/nIter;
	int data_size = (length) * 4 * 1;
	float speed =  ((float)data_size) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
    }
    public static void main(String[] args)
    {
        for(int length = 1; length <= 2048; length ++) testCorrect(length);
        testSpeed(1024 * 1024);
       
                
    }
}

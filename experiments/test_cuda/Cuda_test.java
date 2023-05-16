package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.lang.SimpleTimer;
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
public class Cuda_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());

    public static void test1()
    {
        Tensor ts;
//        ts = eg.tensor(13, 23);
//        System.out.println(ts);
        
        ts = eg.tensor(Vector.randomFloatVector(5*7), 5, 7);
        System.out.println(ts);
        ts = eg.reshape(false, ts, 7 , 5);
        System.out.println(ts);
    }
    
    public static void test2()
    {
        System.out.println(eg);
        for(int dim1 = 1; dim1 <= 64; dim1++)
        for(int dim2 = 1; dim2 <= 8; dim2++)
        for(int dim3 = 1; dim3 <= 8; dim3++)
        for(int dim4 = 1; dim3 <= 32; dim3++)
        {
            float[] value = Vector.randomFloatVector(dim1*dim2*dim3*dim4);
            Tensor ts = eg.tensor(value, dim1, dim2, dim3, dim4);
            eg.delete(ts);
        }
        System.out.println(eg);
    }
    
    //<editor-fold defaultstate="collapsed" desc="float-test">
    public static void testCorrectFloat(int N, int IH, int IW, int IC) 
    {
        int size = N*IH*IW*IC;
        float[] X = Vector.randomFloatVector(size);
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        float[] X2 = eg.valueOf(tX);
        float sp = Vector.samePercentAbsolute(X, X2);
        
        Vector.println("CPU: ", X, 0, 10);
        Vector.println("GPU: ", X2, 0, 10);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testCorrect_constant(int N, int IH, int IW, int IC, float v) 
    {
        System.out.format("testCorrectConstant[%d, %d, %d, %d]\n", N, IH, IW, IC);
        
        Tensor tX = eg.constant(v, N, IH, IW, IC).c();
        float value[] = tX.value();
        for(int i=0; i<value.length; i++)  
            if(value[i] != v)  throw new RuntimeException();
        tX.delete();
    }
    
    public static void testSpeedFloat(int N, int IH, int IW, int IC) 
    {
        int length = N * IH * IW * IC;
        float[] X = Vector.randomFloatVector(length);
        
        int nIter = 30;
        
        SimpleTimer timer = SimpleTimer.instance().record();
        for(int i=0; i < nIter; i++)
        {
            Tensor tX = eg.tensor(X, N, IH, IW, IC);
            eg.delete(tX);
        }
        long div = timer.record().timeStampDifMills();
        
        float time = 1.0f * div / nIter;
	int data_size = length << 2;
	double speed = (1.0 *(data_size) / (1 << 30)) / (time*1e-3);
        System.out.println("Time = " + time + " msec, Speed = " + speed + "GB/s");
    }
    
    public static void testCorrectByte(int N, int IH, int IW, int IC)
    {
        int size = N*IH*IW*IC;
        byte[] X = Vector.randomByteVector(size);
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        float[] X2 = eg.valueOf(tX);
        
        float[] fX = Vector.toFloatVector(X);
        float sp = Vector.samePercentAbsolute(fX, X2);
        
        Vector.println("CPU: ", fX, 0, 10);
        Vector.println("GPU: ", X2, 0, 10);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testCorrectInt(int N, int IH, int IW, int IC)
    {
        int size = N*IH*IW*IC;
        int[] X = Vector.randomIntVector(size, 100);
        
        float[] X1 = Vector.toFloatVector(X);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        float[] X2 = eg.valueOf(tX);
       
        float sp = Vector.samePercentAbsolute(X1, X2);
        
        Vector.println("CPU: ", X1, 0, 10);
        Vector.println("GPU: ", X2, 0, 10);
        System.out.println("sp = " + sp);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testSpeedByte(int N, int IH, int IW, int IC) 
    {
        int size = N * IH * IW * IC;
        byte[] X = Vector.randomByteVector(size);
        
        int nIter = 30;
        
        SimpleTimer timer = SimpleTimer.instance().record();
        for(int i=0; i < nIter; i++)
        {
            Tensor tX = eg.tensor(X, N, IH, IW, IC);
            eg.delete(tX);
        }
        long div = timer.record().timeStampDifMills();
        
        float time = 1.0f * div / nIter;
	int data_size = size << 2;
	double speed = (1.0 *(data_size) / (1 << 30)) / (time*1e-3);
        System.out.println("Time = " + time + " msec, Speed = " + speed + "GB/s");
    }
    //</editor-fold>
    
    public static void testCorrect_int32(int N, int IH, int IW, int IC) 
    {
        System.out.format("testCorrect_int32: (N, IH, IW, IC) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
        int size = N * IH * IW * IC;
        int[] X = Vector.randomIntVector(size);
        Tensor tX = eg.tensor_int32(X, N, IH, IW, IC);
        int[] X2 = eg.valueOf_int32(tX);
        float sp = Vector.samePercentAbsolute(X, X2);
        
        Vector.println(X, 0, 10);
        Vector.println(X2, 0, 10);
        System.out.println("sp = " + sp);
        System.out.println(tX);
        
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testCorrect_int8(int N, int IH, int IW, int IC) 
    {
        eg.sync(false);
        int size = N * IH * IW * IC;
        
        System.out.format("testCorrect_int8: (N, IH, IW, IC) = (%d, %d, %d, %d)\n", N, IH, IW, IC);
        System.out.println("size = " + size);
        
        byte[] X1 = Vector.randomByteVector(size);
        Tensor tX = eg.tensor_int8(X1, N, IH, IW, IC);
        Cuda_expk2.checkMemAlign(tX);
        
        byte[] X2 = eg.valueOf_int8(tX);
        float sp = Vector.samePercentAbsolute(X1, X2);
        
        Vector.println("X1: ", X1 , 0, 10);
        Vector.println("X2: ", X2, 0, 10);
        System.out.println("sp = " + sp);
        System.out.println(tX);
        tX.delete();
        
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        try
        {
//            testCorrect_constant(1, 5, 7, 5, 1.1f);
//            for(int n=1; n<32; n++)
//            for(int ih=1; ih<8; ih++)
//            for(int iw=1; iw<8; iw++)
//            for(int ic=1; ic<8; ic++)
//                testCorrect_constant(n, ih, iw, ic, 1.1f);
            
//            for(int n=1; n<32; n++)
//            for(int ih=1; ih<8; ih++)
//            for(int iw=1; iw<8; iw++)
//            for(int ic=1; ic<8; ic++)
//                testCorrect_int32(n, ih, iw, ic);
            
//            testCorrect_int8(1, 1, 1, 4);
//            for(int n=1; n<32; n++)
//            for(int ih=1; ih<8; ih++)
//            for(int iw=1; iw<8; iw++)
//            for(int ic=1; ic<8; ic++)
//                testCorrect_int8(n, ih, iw, ic);
            
//            testCorrectFloat(64, 128, 128, 16);
//            testSpeedFloat(64, 128, 128, 16);
            
//            for(int n=1; n<32; n++)
//            for(int ih=1; ih<8; ih++)
//            for(int iw=1; iw<8; iw++)
//            for(int ic=1; ic<8; ic++)
//                testCorrectByte(n, ih, iw, ic);
            
//            for(int n=1; n<32; n++)
//            for(int ih=1; ih<8; ih++)
//            for(int iw=1; iw<8; iw++)
//            for(int ic=1; ic<8; ic++)
//                testCorrectInt(n, ih, iw, ic);
            
//            
//            testCorrectByte(64, 128, 128, 16);
//            testSpeedByte(64, 128, 128, 16);
            
            for(int n=1; n<=4; n++)
            for(int ih=1; ih<=32; ih++)
            for(int iw=1; iw<=32; iw++)
                testCorrect_int8(n, ih, iw, 3);
            
//            testCorrect_int8(13, 1, 1, 3);
//            testCorrect_int8(13, 1, 1, 3);
            
            System.out.println(eg.engineBase());
            
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}

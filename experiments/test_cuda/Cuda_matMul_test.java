package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Matrix;
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
public class Cuda_matMul_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
     
    static void multiply(float[][] A, float[][] B, float[][] C)
    {
        //step A = K
        //step B,C = M
        int N=A.length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j]+=A[i][k]*B[k][j];
    }
    //A(K*N), B(K*M), C(N*M), A^T(N*K)
    //C = A^T * B
    static void multiplyT1(float[][] A, float[][] B, float[][] C)
    {
        //step A = N
        int N=A[0].length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j]+=A[k][i]*B[k][j];
    }
    
    public static void testCorrect(int N, int M, int K)
    {
        System.out.println("TestCorrect(N, M, K): "+ N + ", " + M + ", " + K);
        float[] A = Vector.randomFloatVector(N*K); 
        float[] B = Vector.randomFloatVector(K*M); 
        
        Tensor tA = eg.tensor(A, N, K);
        Tensor tB = eg.tensor(B, K, M);
        
        //CPU-------------------------------------------------------------------
        float[][] mA = Matrix.toMatrix(A, K);
        float[][] mB = Matrix.toMatrix(B, M);
        float[][] mC = new float[N][M];
       
        multiply(mA, mB, mC);
        
        float[] C1 = Matrix.toVector(mC, N*M);
        
        //GPU-------------------------------------------------------------------
        Tensor tC1 = eg.matMul(tA, tB).c();
        Tensor tC2 = eg.matMul(eg.empty(N, M).c(), tA, tB).c();
        
        float[] C2 = eg.valueOf(tC1);
        float[] C3 = eg.valueOf(tC2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentAbsolute(C1, C2);
        float sp2 = Vector.samePercentAbsolute(C1, C3);
        
        System.out.print("CPU : "); Vector.println(C1, 0, 10);
        System.out.print("GPU1: "); Vector.println(C2, 0, 10);
        System.out.print("GPU2: "); Vector.println(C3, 0, 10);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.999f || sp2 < 0.999f) { throw new RuntimeException(N + " "+ M + " "+K); }
    }
    
    public static void testSpeed(int N, int M, int K)
    {
        eg.check(false).sync(false);
        float[] A=Vector.randomFloatVector(N*K);
        float[] B=Vector.randomFloatVector(K*M);
        
        Tensor tA = eg.tensor(A, N, K).c();
        Tensor tB = eg.tensor(B, K, M).c();
        Tensor tC = eg.empty(N, M).c();
       
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.matMul(tC, tA, tB);
        }
        Cuda.deviceSynchronize();
        long dif = timer.record().timeStampDifMills();
        
        System.out.println(eg);
        System.out.println("total time = " + dif);
        
        float time = (float) (1.0*dif / nIter);
	double sizeV = N * M * K;
	float performance = (float) ((2 * sizeV * 1.0e-9f) / (time / 1000.0f));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)(sizeV/(1024*1024*1024)), time, performance);
    }
    
    public static void main(String[] args)
    {
//          test3();
//        //20  52  2
        int N = 1024, M = 1148, K = 1024;
        testCorrect(N, M, K);
        testSpeed(N, M, K);
        System.out.println(CudaException.lastException());
//        
        //1.3526666 msec, 1587.5927 GFlop/s
//        for(int N=1; N<=64;N++)
//            for(int M=4; M<=64; M++)  
//                for(int K=2;K<=32;K++) testCorrect(N, M, K);
        //32  128  1
       
        
    }
}

package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
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
public class Cuda_matMul_test_T1 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    //A(K*N), B(K*M), C(N*M), A^T(N*K)
    //C = A^T * B
    static void multiplyT1(float[][] A, float[][] B, float[][] C)
    {
        //step A = N
        int N=A[0].length, M=B[0].length, K=B.length;
        for(int k=0;k<K;k++)
            for(int i=0;i<N;i++)
                for(int j=0;j<M;j++)
                    C[i][j] += A[k][i]*B[k][j];
    }
    
    public static void testCorrect(int N, int M, int K)
    {
        eg.sync(false);
        System.out.println("TestCorrect:" + N +",  " + M + ",  " + K);
        float[] A=Vector.randomFloatVector(K * N);
        float[] B=Vector.randomFloatVector(K * M);
        
        //CPU-------------------------------------------------------------------
        float[][] mA = Matrix.toMatrix(A, N);
        float[][] mB = Matrix.toMatrix(B, M);
        float[][] mC = new float[N][M];
        
        multiplyT1(mA, mB, mC);
        
        float[] C1 = Matrix.toVector(mC, N*M);
        
        //GPU-------------------------------------------------------------------
        Tensor tA = eg.tensor(A, K, N).c();
        Tensor tB = eg.tensor(B, K, M).c();
        Tensor tC1 = eg.matMulT1(tA, tB).c();
        Tensor tC2 = eg.matMulT1(eg.empty(N, M).c(), tA, tB);
        
        float[] C2 = eg.valueOf(tC1);
        float[] C3 = eg.valueOf(tC2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(C2, C1, 1e-4f);
        float sp2 = Vector.samePercentRelative(C3, C1, 1e-4f);
        
        System.out.print("CPU :"); Vector.println(C1, 0, 10);
        System.out.print("GPU1:" ); Vector.println(C2, 0, 10);
        System.out.print("GPU2:" ); Vector.println(C3, 0, 10);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.999f || sp2 < 0.999f) { throw new RuntimeException(N+" "+M+" "+K); }
    }
    
    public static void testSpeed(int N, int M, int K)
    {
        eg.check(false).sync(false);

        float[] A = Vector.randomFloatVector(K*N);
        float[] B = Vector.randomFloatVector(K*M);
        
        Tensor tA = eg.tensor(A, K, N).c();
        Tensor tB = eg.tensor(B, K, M).c();
        Tensor tC = eg.empty(N, M).c();
       
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.matMulT1(tC, tA, tB).c();
        }
        Cuda.deviceSynchronize();
        long dif = timer.record().timeStampDifMills();
        
        System.out.println(eg);
        System.out.println("total time = " + dif);
        
        float msecPerMatrixMul = (float) (1.0*dif / nIter);
	double flopsPerMatrixMul = 2.0 * N * M * K;
	float gigaFlops = (float) ((flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f));
        System.out.print(msecPerMatrixMul + " msec, ");
        System.out.println(gigaFlops+" GFlop/s");
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        try
        {
//            test3();
//        //20  52  2
//        int N = 380, M = 380, K = 1024*16;
//        testCorrect(N, M, K);
//        testSpeed(N, M, K);
//        System.out.println(CudaException.lastException());
//        
//            for(int N=1; N<=64;N++)
//                for(int M=4; M<=64; M++)  
//                    for(int K=2; K<=32; K++) testCorrect(N, M, K);
            
//            testCorrect(9, 9, 512);
            
            for(int N=1; N<=64;N++)
                for(int M=4; M<=64; M++)  
                    for(int K=512; K<=517; K++) testCorrect(N, M, K);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}

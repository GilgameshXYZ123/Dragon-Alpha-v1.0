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
public class Cuda_batchMatMulT1_test
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static void matMulT1(float[][] A, float[][] B, float[][] C)
    {
        int N = A[0].length, M = B[0].length, K = B.length;
        for(int k = 0; k <K; k++)
            for(int i = 0; i < N; i++)
                for(int j = 0; j < M; j++)
                    C[i][j] += A[k][i] * B[k][j];
    }
    
    static void bmmT1(float[][][][] A, float[][][][] B, float[][][][] C) 
    {
        int dim0 = A.length;
        int dim1 = A[0].length;
        
        for(int d0=0; d0<dim0; d0++) 
            for(int d1=0; d1<dim1; d1++)
                matMulT1(A[d0][d1], B[d0][d1], C[d0][d1]);
    }
    
    public static void testCorrect(int Batch0, int Batch1, int N, int M, int K)
    {
        System.out.println("TestCorrect:");
        System.out.format("[Batch0, Batch1, N, M, K] = [%d, %d, %d, %d, %d]\n",
                Batch0, Batch1, N, M, K);
        
        int sizeA = Batch0 * Batch1 * K * N;
        int sizeB = Batch0 * Batch1 * K * M;
        float[] A = Vector.randomFloatVector(sizeA);
        float[] B = Vector.randomFloatVector(sizeB);
        Tensor tA = eg.tensor(A, Batch0, Batch1, K, N).c();//N -> 4x
        Tensor tB = eg.tensor(B, Batch0, Batch1, K, M).c();//M -> 4x
        
        //CPU-------------------------------------------------------------------
        float[][][][] mA = Vector.to4D(A, Batch0, Batch1, K, N);
        float[][][][] mB = Vector.to4D(B, Batch0, Batch1, K, M);
        float[][][][] mC = new float[Batch0][Batch1][N][M];// M -> 4x, so we need a new parameter CN
        
        bmmT1(mA, mB, mC);
        
        float[] C1 = Vector.toVector(mC);
        System.out.print("CPU = "); Vector.println(C1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tC = eg.batchMatMulT1(tA, tB).c();
        
        float[] C2 = tC.value();
        System.out.print("GPU = "); Vector.println(C2, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp = Vector.samePercentAbsolute(C2, C1);
        System.out.println("sp = " + sp);
        
        eg.delete(tA, tB, tC);
        if(sp != 1.0f) {throw new RuntimeException(N+" "+M+" "+K);}
    }
    
    public static void testSpeed(int Batch0, int Batch1, int N, int M, int K) 
    {
        eg.check(false).sync(false);
        
        System.out.println("TestCorrect:");
        System.out.format("[Batch0, Batch1, N, M, K] = [%d, %d, %d, %d, %d]\n",
                Batch0, Batch1, N, M, K);
        
        int sizeA = Batch0 * Batch1 * K * N;
        int sizeB = Batch0 * Batch1 * K * M;
        float[] A = Vector.randomFloatVector(sizeA);
        float[] B = Vector.randomFloatVector(sizeB);
        Tensor tA = eg.tensor(A, Batch0, Batch1, K, N).c();
        Tensor tB = eg.tensor(B, Batch0, Batch1, K, M).c();
        Tensor tC = eg.empty(Batch0, Batch1, N, M).c();
        
        int nIter = 500;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<nIter; i++) {
            tC = eg.batchMatMulT1(tC, tA, tB);
        }
        Cuda.deviceSynchronize();
        
        long dif = timer.record().timeStampDifMills();
        float msecPerMatrixMul = (float) (1.0*dif / nIter);
	double flopsPerMatrixMul = Batch0 * Batch1 * 2.0 * N * M * K;
	float gigaFlops = (float) ((flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f));
        float size = (float) (flopsPerMatrixMul / 2 / 1024 / 1024 / 1024);
        
        System.out.print("size = " + size );
        System.out.print(", Time = " + msecPerMatrixMul + " msec");
        System.out.println(", Performance = "+ gigaFlops+" GFlop/s");
    }
    
    public static void main(String[] args) 
    {
//        Vector.PRINT_DIFFERENT = true;
        for(int K=7; K<=9; K++)
        for(int N=1; N <= 82; N++)
            for(int M=1; M <= 96; M++)  
                testCorrect(5, 3, N, M, K);
        
        //[64, 256, 128]
//        [64, 8, 256, 32] * [64, 8, 32, 256]
//        int Batch0 = 16, Batch1 = 8;
//        int N = 224, M = 192, K = 256;
//        testCorrect(Batch0, Batch1, N, M, K);
//        testSpeed(Batch0*2, Batch1, N, M, K);
    }
}

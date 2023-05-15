/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
public final class Cuda_batchMatMul 
{
    private Cuda_batchMatMul() {}

    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize(int N, int M) //N%4, M%4 == 0
    {
        int sn = 0, sm = 0;
        if((N & (N - 1)) == 0) sn = 1;
        else for(;;) {
            if(N > 127) sn++; N &= 127; if(N == 0) break;
            if(N >  63) sn++; N &=  63; if(N == 0) break;
            if(N >  31) sn++; N &=  31; if(N == 0) break;
            if(N >  15) sn++; N &=  15; if(N == 0) break;
            if(N >   7) sn++; N &=   7; if(N == 0) break;
            if(N >   3) sn++; break;
        }
        
        if((M & (M - 1)) == 0) sm = 1;
        else for(;;) {
            if(M > 127) sm++; M &= 127; if(M == 0) break;
            if(M >  63) sm++; M &=  63; if(M == 0) break;
            if(M >  31) sm++; M &=  31; if(M == 0) break;
            if(M >  15) sm++; M &=  15; if(M == 0) break;
            if(M >   7) sm++; M &=   7; if(M == 0) break;
            if(M >   3) sm++; break;
        }
        
        int size = sn * sm; if(size < 1) size = 1;
        return (size < 10 ? size : 10);//the max stream size is 10
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batchMatMul">
    /**
     * <pre>
     * (1) A belongs to Tensor[Batch,  N, AK]
     * (2) B belongs to Tensor[Batch, BK,  M]
     * (3) C belongs to Tensor[Batch,  N,  M]
     * (4) for i from 1 to Batch
     *      C[i] = A[i] * B[i]
     * (5) (Batch, M, AK)%4 == 0 && >= 4.
     * ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 16, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 3]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (82, 96), K = 20, Batch = [5 * 3]:  correct
     * 3D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [12]:  correct
     * 
     * for[Batch0, Batch1, K] = [32, 8, 256]: (nIter = 1000)
     * [N, M] = [128, 128]: size = 1.0, Time = 1.419 msec, Performance = 1513.3782 GFlop/s
     * [N, M] = [160, 128]: size = 1.25, Time = 1.957 msec, Performance = 1371.6681 GFlop/s
     * [N, M] = [160, 160]: size = 1.5625, Time = 3.179 msec, Performance = 1055.5027 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 2.172 msec, Performance = 1483.0687 GFlop/s
     * [N, M] = [192, 192]: size = 2.25, Time = 4.05 msec, Performance = 1193.0464 GFlop/s
     * [N, M] = [192, 224]: size = 2.625, Time = 4.256 msec, Performance = 1324.5171 GFlop/s
     * [N, M] = [224, 224]: size = 3.0625, Time = 5.288 msec, Performance = 1243.6967 GFlop/s
     * [N, M] = [128, 240]: size = 1.875, Time = 2.8 msec, Performance = 1438.0471 GFlop/s
     * [N, M] = [240, 240]: size = 3.515625, Time = 6.29 msec, Performance = 1200.2777 GFlop/s
     * [N, M] = [128, 252]: size = 1.96875, Time = 2.834 msec, Performance = 1491.8342 GFlop/s
     * [N, M] = [252, 252]: size = 3.8759766, Time = 5.51 msec, Performance = 1510.6345 GFlop/s
     * [N, M] = [127, 336]: size = 2.6044922, Time = 4.175 msec, Performance = 1339.6655 GFlop/s
     * [N, M] = [127, 340]: size = 2.635498, Time = 4.176 msec, Performance = 1355.2894 GFlop/s
     * [N, M] = [255, 340]: size = 5.291748, Time = 9.432 msec, Performance = 1204.8285 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param Batch
     * @param N
     * @param M
     * @param BK 
     * @param AK 
     */
    @Passed
    public static native void batchMatMul(long[] streamArray, int length,
            long dA_address, long dB_address,
            long dC_address, 
            int Batch, int N, int M, int BK, int AK);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMul_texture">
    /**
     * <pre>
     * (1) A belongs to Tensor[Batch,  N, AK]
     * (2) B belongs to Tensor[Batch, BK,  M]
     * (3) C belongs to Tensor[Batch,  N,  M]
     * (4) for i from 1 to Batch
     *      C[i] = A[i] * B[i]
     * (5) (Batch, M, AK)%4 == 0 && >= 4.
     * ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 16, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 3]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (82, 96), K = 20, Batch = [5 * 3]:  correct
     * 3D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [12]:  correct
     * 
     * for[Batch0, Batch1, K] = [32, 8, 256]:
     * [N, M] = [128, 128]: size = 1.0, Time = 1.448 msec, Performance = 1483.0688 GFlop/s
     * [N, M] = [132, 132]: size = 1.0634766, Time = 3.22 msec, Performance = 709.25415 GFlop/s
     * [N, M] = [140, 140]: size = 1.1962891, Time = 3.432 msec, Performance = 748.5464 GFlop/s
     * [N, M] = [144, 144]: size = 1.265625, Time = 3.394 msec, Performance = 800.79816 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 2.134 msec, Performance = 1509.4777 GFlop/s
     * [N, M] = [192, 192]: size = 2.25, Time = 4.034 msec, Performance = 1197.7784 GFlop/s
     * [N, M] = [144, 224]:size = 2.109375, Time = 4.7 msec, Performance = 963.7975 GFlop/s
     * [N, M] = [224, 192]: size = 2.625, Time = 4.566 msec, Performance = 1234.5914 GFlop/s
     * [N, M] = [224, 224]: size = 3.0625, Time = 5.31 msec, Performance = 1238.544 GFlop/s
     * [N, M] = [240, 240]: size = 3.515625, Time = 5.404 msec, Performance = 1397.0665 GFlop/s
     * [N, M] = [248, 248]: size = 3.7539062, Time = 5.476 msec, Performance = 1472.1425 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param Batch
     * @param N
     * @param M
     * @param BK 
     * @param AK 
     */
    @Passed
    public static native void batchMatMul_texture(long[] streamArray, int length,
            long dA_address, long dB_address,
            long dC_address, 
            int Batch, int N, int M, int BK, int AK);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT1">
    /**
     * <pre>
     * (1) A belongs to Tensor[Batch,  K, AN] logically transpose(1, 2)-> A^T[Batch, AN, K]
     * (2) B belongs to Tensor[Batch,  K,  M]
     * (3) C belongs to Tensor[Batch, CN, M]
     * (4) for i from 1 to Batch: C[i] = A^T[i] * B[i]
     * (5) (Batch, AN, M)%4 == 0 && >= 4.
     * ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 16, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (82, 96), K = 20, Batch = [5 * 3]:  correct
     * 3D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [12]:  correct
     * 
     * for[Batch0, Batch1, K] = [32, 8, 256]:
     * [N, M] = [128, 128]: size = 1.0, Time = 1.446 msec, Performance = 1485.1201 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 2.232 msec, Performance = 1443.2013 GFlop/s
     * [N, M] = [192, 192]: size = 2.25, Time = 3.228 msec, Performance = 1496.8519 GFlop/s
     * [N, M] = [192, 224]: size = 2.625, Time = 4.546 msec, Performance = 1240.023 GFlop/s
     * [N, M] = [224, 224]: size = 3.0625, Time = 6.116 msec, Performance = 1075.3219 GFlop/s
     * [N, M] = [240, 240]: size = 3.515625, Time = 6.19 msec, Performance = 1219.6683 GFlop/s
     * [N, M] = [248, 248]: size = 3.7539062, Time = 5.71 msec, Performance = 1411.813 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param Batch
     * @param CN
     * @param AN
     * @param M
     * @param K 
     */
     @Passed
    public static native void batchMatMulT1(long[] streamArray, int length,
            long dA_address, long dB_address,
            long dC_address, 
            int Batch, int CN, int AN, int M, int K);
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2">
    /**
     * <pre>
     * (1) A belongs to Tensor[Batch,  N,  K]
     * (2) B belongs to Tensor[Batch, BM,  K] logically transpose(1, 2)->Tensor[Batch, K, BM]
     * (3) C belongs to Tensor[Batch,  N, CM]
     * (4) for i from 1 to Batch: C[i] = A[i] * B^T[i]
     * (5) (Batch, BM, K)%4 == 0 && >= 4.
     * ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 16, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 3]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (82, 96), K = 20, Batch = [5 * 3]:  correct
     * 3D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [12]:  correct
     * 
     * for[Batch0, Batch1, K] = [32, 8, 256]:
     * [N, M] = [128, 128]: size = 1.0, Time = 1.558 msec, Performance = 1378.3591 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 2.754 msec, Performance = 1169.6533 GFlop/s
     * [N, M] = [128, 119]: size = 0.9296875, Time = 2.384 msec, Performance = 837.4533 GFlop/s
     * [N, M] = [192, 192]: size = 2.25, Time = 5.618 msec, Performance = 860.0637 GFlop/s
     * [N, M] = [192, 224]: size = 2.625, Time = 4.806 msec, Performance = 1172.9388 GFlop/s
     * [N, M] = [224, 224]: size = 3.0625, Time = 7.444 msec, Performance = 883.48584 GFlop/s
     * [N, M] = [240, 240]: size = 3.515625, Time = 7.89 msec, Performance = 956.8754 GFlop/s
     * [N, M] = [252, 252]: size = 3.8759766, Time = 8.434 msec, Performance = 986.9096 GFlop/s
     * [N, M] = [127, 336]: size = 2.6044922, Time = 6.254 msec, Performance = 894.3243 GFlop/s
     * [N, M] = [127, 340]: size = 2.635498, Time = 6.332 msec, Performance = 893.8233 GFlop/s
     * [N, M] = [255, 340]: size = 5.291748, Time = 14.554 msec, Performance = 780.8123 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param Batch
     * @param N
     * @param CM
     * @param BM
     * @param K 
     */
    @Passed
    public static native void batchMatMulT2(long[] streamArray, int length,
            long dA_address, long dB_address,
            long dC_address, 
            int Batch, int N, int CM, int BM, int K); 
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2_texture">
    /**
     * <pre>
     * (1) A belongs to Tensor[Batch,  N,  K]
     * (2) B belongs to Tensor[Batch, BM,  K] logically transpose(1, 2)->Tensor[Batch, K, BM]
     * (3) C belongs to Tensor[Batch,  N, CM]
     * (4) for i from 1 to Batch: C[i] = A[i] * B^T[i]
     * (5) (Batch, M, K)%4 == 0 && >= 4.
     * ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 16, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 8]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [4 * 3]:  correct
     * 4D Tensor: for(N, M) from (1, 1) to(+1, +1) to (82, 96), K = 20, Batch = [5 * 3]:  correct
     * 3D Tensor: for(N, M) from (1, 1) to(+1, +1) to (64, 64), K = 20, Batch = [12]:  correct
     * 
     * for[Batch0, Batch1, K] = [32, 8, 256]:
     * [N, M] = [128, 128]: Size = 1.0, Time = 1.824 msec, Performance = 1177.3484 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 3.404 msec, Performance = 946.3059 GFlop/s
     * [N, M] = [192, 128]: size = 1.5, Time = 3.454 msec, Performance = 932.60724 GFlop/s
     * [N, M] = [192, 192]: size = 2.25, Time = 4.564 msec, Performance = 1058.6848 GFlop/s
     * [N, M] = [224, 224]: size = 3.0625, Time = 6.948 msec, Performance = 946.55566 GFlop/s
     * [N, M] = [240, 240]: size = 3.515625, Time = 7.19 msec, Performance = 1050.0343 GFlop/s
     * [N, M] = [248, 248]: size = 3.7539062, Time = 7.28 msec, Performance = 1107.3423 GFlop/
     * [N, M] = [252, 252]: size = 3.8759766, Time = 7.496 msec, Performance = 1110.405 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param Batch
     * @param N
     * @param CM
     * @param BM
     * @param K 
     */
    @Passed
    public static native void batchMatMulT2_texture(long[] streamArray, int length,
            long dA_address, long dB_address,
            long dC_address, 
            int Batch, int N, int CM, int BM, int K); 
    //</editor-fold>
}

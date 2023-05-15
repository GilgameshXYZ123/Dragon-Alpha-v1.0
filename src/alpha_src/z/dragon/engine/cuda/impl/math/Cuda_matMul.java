/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * C = A * B.
 * @author Gilgamesh
 */
public final class Cuda_matMul 
{
    private Cuda_matMul(){}
    
    //<editor-fold defaultstate="collapsed" desc="streamSize && blockSize">
    public static int blockNum(int N, int M)//N%4, M%4 == 0
    {
        int bn = 0,bm = 0;
        for(;;) {
            if(N > 127) bn += N >> 7; N &= 127; if(N == 0) break;//2^7
            if(N >  63) bn += 1; N &= 63; if(N == 0) break;//2^6
            if(N >  31) bn += 1; N &= 31; if(N == 0) break;//2^5
            if(N >  15) bn += 1; N &= 15; if(N == 0) break;//2^4
            if(N >   7) bn += 1; N &=  7; if(N == 0) break;//2^3
            if(N >   3) bn += 1; break;
        }
      
        for(;;) {
            if(M > 127) bm += M >> 7; M &= 127; if(M == 0) break;//2^7
            if(M >  63) bm += 1; M &= 63; if(M == 0) break;//2^6, if M <= 63, M % 64 == M
            if(M >  31) bm += 1; M &= 31; if(M == 0) break;//2^5
            if(M >  15) bm += 1; M &= 15; if(M == 0) break;//2^4
            if(M >   7) bm += 1; M &=  7; if(M == 0) break;//2^3
            if(M >   3) bm += 1; break;
        }
        return bn * bm;
    }
    
    public static int streamSize(int N, int M)//N%4, M%4 == 0
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

        int size = sn * sm;
        return (size < 10? size : 10);//the max stream size is 10
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SKbuf_summary">
    public static native void SKbuf_summary(long stream_address,
            long dCbuf_address, long dC_address,
            int part, int sizeC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="matMul">
    /**
     * <pre>
     * (1) A belongs to Mat[N, K]
     * (2) B belongs to Mat[K, M]
     * (3) C belongs to Mat[N, M]
     * (4) C = A * B
     * (5) (N, M, K)%4 == 0 && >= 4.
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * for(N, M, K) from (1, 1, 1) to(+1, +1, +1) (64, 64, 32): correct
     * 
     * for(1024, 1024, 1024): Size = 1.000000, Time = 1.387000 msec, Performance = 1548.293945 GFlop/s
     * for(1088, 1088, 1024): Size = 1.128906, Time = 1.636000 msec, Performance = 1481.850586 GFlop/s
     * for(1120, 1120, 1024): Size = 1.196289, Time = 2.062000 msec, Performance = 1245.883179 GFlop/s
     * for(1136, 1136, 1024): Size = 1.230713, Time = 2.329000 msec, Performance = 1134.794189 GFlop/s
     * for(1144, 1144, 1024): Size = 1.248108, Time = 2.474000 msec, Performance = 1083.383667 GFlop/s
     * 
     * for(1024, 1088, 1024): Size = 1.062500, Time = 1.459000 msec, Performance = 1563.880371 GFlop/s
     * for(1024, 1120, 1024): Size = 1.093750, Time = 1.594000 msec, Performance = 1473.532104 GFlop/s
     * for(1024, 1136, 1024): Size = 1.109375, Time = 1.663000 msec, Performance = 1432.570435 GFlop/s
     * for(1024, 1144, 1024): Size = 1.117188, Time = 1.743000 msec, Performance = 1376.443970 GFlop/s
     * for(1024, 1148, 1024): Size = 1.121094, Time = 1.827000 msec, Performance = 1317.750610 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param N N = dA.height = dC.height
     * @param M M = dB.width = dC.width = dC.stride = dB.stride
     * @param K K = dA.width = dB.height = dA.stride
     */
    @Passed
    public static native void matMul(long[] streamArray, int length,
            long dA_address, long dB_address, 
            long dC_address,
            int N, int M, int K);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="matMul44T1">
    /**
     * <pre>
     * (1) A^T belongs to Mat[N, K], A belongs to Mat[K, N]
     * (2) B belongs to Mat[K, M]
     * (3) C belongs to Mat[N, M]
     * (4) C = (A^T) * B
     * (5) (N, M, K)%4 == 0 && >= 4.
     * 
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[sychronized]--------------
     * for(N, M, K) from (1, 1, 1) to(+1, +1, +1) (64, 64, 32): correct
     * for(1024, 1024, 1024): 1.54 msec, 1394.4698 GFlop/s
     * for(1088, 1088, 1024): 1.78 msec, 1361.9706 GFlop/s
     * for(1148, 1148, 1024): 2.45 msec, 1101.6602 GFlop/s
     * for(1151, 1151, 1024): 1.95 msec, 1391.3806 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * for(1024, 1024, 1024): 1.339 msec, 1603.7966 GFlop/s
     * for(1024, 1088, 1024): 1.391 msec, 1640.3317 GFlop/s
     * for(1088, 1088, 1024): 1.514 msec, 1601.26 GFlop/s
     * for(1120, 1120, 1024): 1.652 msec, 1555.0914 GFlop/s
     * for(1136, 1136, 1024): 1.846 msec, 1431.7095 GFlop/s
     * for(1148, 1148, 1024): 2.154 msec, 1253.0488 GFlop/s
     * for(1151, 1151, 1024): 1.686 msec, 1609.2482 GFlop/s
     * </pre>
     * @param streams
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param N N = A.width = C.height
     * @param M M = B.width = C.width
     * @param K K = A.height = B.height
     */
    @Passed
    public static native void matMulT1(long[] streams, int length,
            long dA_address,
            long dB_address, 
            long dC_address,
            int N, int M, int K);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMul44T1SK">
    /**
     * <pre>
     * (1) A^T belongs to Mat[N, K], A belongs to Mat[K, N]
     * (2) B belongs to Mat[K, M]
     * (3) C belongs to Mat[N, M]
     * (4) C = (A^T) * B
     * (5) (N, M, K)%4 == 0 && >= 4.
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * for(256, 256, 1024*16): 1.515 msec, 1417.481  GFlop/s
     * for(256, 320, 1024*16): 1.847 msec, 1453.3593 GFlop/s
     * for(256, 352, 1024*16): 2.105 msec, 1402.7506 GFlop/s
     * for(320, 256, 1024*16): 1.838 msec, 1460.4757 GFlop/s 
     * for(352, 256, 1024*16): 2.149 msec, 1374.0299 GFlop/s
     * for(320, 320, 1024*16): 2.298 msec, 1460.158  GFlop/s
     * for(320, 352, 1024*16): 2.626 msec, 1405.555  GFlop/s
     * for(352, 352, 1024*16): 3.043 msec, 1334.238  GFlop/s
     * for(352, 368, 1024*16): 3.337 msec, 1271.9916 GFlop/s
     * for(368, 368, 1024*16): 3.667 msec, 1210.1373 GFlop/s
     * for(368, 376, 1024*16): 4.073 msec, 1113.1948 GFlop/s
     * for(376, 376, 1024*16): 4.516 msec, 1025.8213 GFlop/s
     * for(376, 380, 1024*16): 5.143 msec,  910.34247 GFlop/s
     * for(380, 380, 1024*16): 5.906 msec,  801.1681 GFlop/s
     * </pre>
     * @param streams
     * @param length
     * @param GZ
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param dCBuf_address
     * @param N N = A.width = C.height
     * @param M M = B.width = C.width
     * @param K K = A.height = B.height
     */
    @Passed
    public static native void matMulT1SK(long[] streams, int length, int GZ,
            long dA_address,
            long dB_address, 
            long dC_address, long dCBuf_address,
            int N, int M, int K);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="matMul44T2">
    /**
     * <pre>
     * (1) A belongs to Mat[N, K]
     * (2) B belongs to Mat[M, K], B^T belongs to Mat[K, M]
     * (3) C belongs to Mat[N, M]
     * (4) C = A * B^T.
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * for(N, M, K) from (1, 1, 1) to(+1, +1, +1) (64, 64, 32): correct
     * 
     * for(1024, 1024, 1024): Size = 1.000000, Time = 1.419000 msec, Performance = 1513.378174 GFlop/s
     * for(1024, 1088, 1024): Size = 1.062500, Time = 1.550000 msec, Performance = 1472.065430 GFlop/s
     * for(1088, 1088, 1024): Size = 1.128906, Time = 1.696000 msec, Performance = 1429.426636 GFlop/s
     * for(1120, 1120, 1024): Size = 1.196289, Time = 2.039000 msec, Performance = 1259.936890 GFlop/s
     * for(1136, 1136, 1024): Size = 1.230713, Time = 2.462000 msec, Performance = 1073.491455 GFlop/s
     * for(1148, 1148, 1148): Size = 1.256851, Time = 3.258000 msec, Performance = 828.443054 GFlop/s
     * for(1149, 1149, 1149): Size = 1.259042, Time = 1.773000 msec, Performance = 1524.969849 GFlop/s
     * </pre>
     * @param streams
     * @param length
     * @param dA_address
     * @param dB_address
     * @param dC_address
     * @param N N = A.height = C.height
     * @param M M = C.width = B.height
     * @param K K = A.width = B.width 
     */
    @Passed
    public static native void matMulT2(long[] streams, int length,
            long dA_address,
            long dB_address, 
            long dC_address,
            int N, int M, int K);
    //</editor-fold>
}

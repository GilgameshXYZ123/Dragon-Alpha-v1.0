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
public final class Cuda_pool2D 
{
    private Cuda_pool2D() {}
    
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize(int OH, int OW, int N, int IC)
    {
        int GN = IC, GM = N * OH * OW;
        
        int sn = 0, sm = 0;
        if((GN & (GN - 1)) == 0) sn = 1;
        else for(;;) {
            if(GN > 127) sn++; GN &= 127; if(GN == 0) break;
            if(GN >  63) sn++; GN &=  63; if(GN == 0) break;
            if(GN >  31) sn++; GN &=  31; if(GN == 0) break;
            if(GN >  15) sn++; GN &=  15; if(GN == 0) break;
            if(GN >   7) sn++; GN &=   7; if(GN == 0) break;
            if(GN >   3) sn++; break;
        }
        
        if((GM & (GM - 1)) == 0) sm = 1;
        else for(;;) {
            if(GM > 127) sm++; GM &= 127; if(GM == 0) break;
            if(GM >  63) sm++; GM &=  63; if(GM == 0) break;
            if(GM >  31) sm++; GM &=  31; if(GM == 0) break;
            if(GM >  15) sm++; GM &=  15; if(GM == 0) break;
            if(GM >   7) sm++; GM &=   7; if(GM == 0) break;
            if(GM >   3) sm++; break;
        }
        
        int size = sn * sm;
        return (size <= 4? size : 4);//the max stream size is 4
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static int[] getOutputTensorShape(int IH, int IW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{N, IC, OH, OW};
    }
    
    public static int[] getInputTensorShape(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{N, IC, IH, IW};
    }
    
    public static int[] getImg2colMatSize(int FH, int FW, int OH, int OW, int N, int IC)
    {
        int GN = IC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        return new int[]{GN, GM, GK};
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pool2D max">
    /**
     * <pre>
     * Y[N, OH, OW, IC] = pool2D_max(X[N, IH, IW, IC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.260000 msec, Performance = 121.115562 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.380000 msec, Performance = 124.302818 GFlop/s
     * (3) IC = 248: Size = 0.056822, Time = 0.593000 msec, Performance = 102.886955 GFlop/s
     * (4) IC = 252: Size = 0.057738, Time = 0.680000 msec, Performance =  91.170624 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.330000 msec, Performance =  95.424370 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.458000 msec, Performance = 103.133339 GFlop/s
     * (3) IC = 224: Size = 0.051323, Time = 0.540000 msec, Performance = 102.051071 GFlop/s
     * (4) IC = 240: Size = 0.054989, Time = 0.590000 msec, Performance = 100.074303 GFlop/s
     * (5) IC = 248: Size = 0.055905, Time = 0.730000 msec, Performance =  82.229996 GFlop/s
     * (6) IC = 252: Size = 0.057738, Time = 0.760000 msec, Performance =  81.573723 GFlop/s
     * (7) IC = 255: Size = 0.058426, Time = 0.580000 msec, Performance = 108.162201 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address 
     * @param IH dim size on Y axis of X
     * @param IW dim size on X axis of X
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH dim size on Y axis of Y
     * @param OW dim size on Y axis of Y
     * @param N batch size 
     * @param IC input channel num
     * @param sh step on y axis
     * @param sw step on x axis
     * @param ph padding on y axis
     * @param pw padding on x axis
     */
    @Passed
    public static native void pool2D_max(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool2D max indexed">
    /**
     * <pre>
     * Y[N, OH, OW, IC] = pool2D_max(X[N, IH, IW, IC], [FH, FW])
     * Index: to save the relation X -> Y.
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.260000 msec, Performance = 121.115562 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.380000 msec, Performance = 124.302818 GFlop/s
     * (3) IC = 248: Size = 0.056822, Time = 0.593000 msec, Performance = 102.886955 GFlop/s
     * (4) IC = 252: Size = 0.057738, Time = 0.680000 msec, Performance =  91.170624 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.330000 msec, Performance =  95.424370 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.458000 msec, Performance = 103.133339 GFlop/s
     * (3) IC = 224: Size = 0.051323, Time = 0.540000 msec, Performance = 102.051071 GFlop/s
     * (4) IC = 240: Size = 0.054989, Time = 0.590000 msec, Performance = 100.074303 GFlop/s
     * (5) IC = 248: Size = 0.055905, Time = 0.730000 msec, Performance =  82.229996 GFlop/s
     * (6) IC = 252: Size = 0.057738, Time = 0.760000 msec, Performance =  81.573723 GFlop/s
     * (7) IC = 255: Size = 0.058426, Time = 0.580000 msec, Performance = 108.162201 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address 
     * @param IH dim size on Y axis of X
     * @param IW dim size on X axis of X
     * @param FH
     * @param FW
     * @param dY_address
     * @param dIndex_address
     * @param OH dim size on Y axis of Y
     * @param OW dim size on Y axis of Y
     * @param N batch size 
     * @param IC input channel num
     * @param sh step on y axis
     * @param sw step on x axis
     * @param ph padding on y axis
     * @param pw padding on x axis
     */
    @Passed
    public static native void pool2D_max_indexed(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            int FH, int FW,
            long dY_address, long dIndex_address, int OH, int OW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pool2D_avg">
    /**
     * <pre>
     * Y[N, OH, OW, IC] = pool2D_avg(X[N, IH, IW, IC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.270000 msec, Performance = 116.629791 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.390000 msec, Performance = 121.115570 GFlop/s
     * (3) IC = 224: Size = 0.051323, Time = 0.460000 msec, Performance = 119.799080 GFlop/s
     * (4) IC = 240: Size = 0.054989, Time = 0.510000 msec, Performance = 115.772224 GFlop/s
     * (5) IC = 248: Size = 0.056822, Time = 0.578000 msec, Performance = 105.557030 GFlop/s
     * (6) IC = 252: Size = 0.057738, Time = 0.670000 msec, Performance =  92.531387 GFlop/s
     * (7) IC = 255: Size = 0.058426, Time = 0.525000 msec, Performance = 119.493477 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * (1) IC = 128: Size = 0.029327, Time = 0.391000 msec, Performance = 80.537201 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.544000 msec, Performance = 86.829163 GFlop/s
     * (3) IC = 252: Size = 0.057738, Time = 0.893000 msec, Performance = 69.424446 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH dim size on Y axis of X
     * @param FH
     * @param FW
     * @param IW dim size on X axis of X
     * @param dY_address
     * @param OH dim size on Y axis of Y
     * @param OW dim size on Y axis of Y
     * @param N batch size 
     * @param IC input channel num
     * @param sh step on y axis
     * @param sw step on x axis
     * @param ph padding on y axis
     * @param pw padding on x axis
     */
    @Passed
    public static native void pool2D_avg(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="pool2D_avg_ignore_padding">
    /**
     * <pre>
     * Y[N, OH, OW, IC] = pool2D_avg(X[N, IH, IW, IC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 0.029327, Time = 0.270000 msec, Performance = 116.629791 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.390000 msec, Performance = 121.115570 GFlop/s
     * (3) IC = 224: Size = 0.051323, Time = 0.460000 msec, Performance = 119.799080 GFlop/s
     * (4) IC = 240: Size = 0.054989, Time = 0.510000 msec, Performance = 115.772224 GFlop/s
     * (5) IC = 248: Size = 0.056822, Time = 0.578000 msec, Performance = 105.557030 GFlop/s
     * (6) IC = 252: Size = 0.057738, Time = 0.670000 msec, Performance =  92.531387 GFlop/s
     * (7) IC = 255: Size = 0.058426, Time = 0.525000 msec, Performance = 119.493477 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 31, [FH, FW] = 3, [N] = 8, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 62, [FH, FW] = 4, [N] = 8, [sh, sw] = 2, [ph, pw] = 1
     * for ic from 1 to 128: correct
     * (1) IC = 128: Size = 0.029327, Time = 0.391000 msec, Performance = 80.537201 GFlop/s
     * (2) IC = 192: Size = 0.043991, Time = 0.544000 msec, Performance = 86.829163 GFlop/s
     * (3) IC = 252: Size = 0.057738, Time = 0.893000 msec, Performance = 69.424446 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH dim size on Y axis of X
     * @param FH
     * @param FW
     * @param IW dim size on X axis of X
     * @param dY_address
     * @param OH dim size on Y axis of Y
     * @param OW dim size on Y axis of Y
     * @param N batch size 
     * @param IC input channel num
     * @param sh step on y axis
     * @param sw step on x axis
     * @param ph padding on y axis
     * @param pw padding on x axis
     */
    @Passed
    public static native void pool2D_avg_ignore_padding(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Document">
    /**
     * <pre>
     * Pooling 2D. {@link z.jcuda.la.CudaLA_pool2D}
     * It is really OK to regrad pooling as a special type of Convolution
     *
     * (1) X is a 4D tensor: X[N, IC, IH, IW]
     * (2) Y is a 4D tensor: Y[N, IC, OH, OW]
     * (3) pooling kernel size: FH, FW
     * (4) pooling step: sh, sw
     * (5) padding: ph, pw
     *
     * -----------------------------------------------------------------------------
     * Forward Prop: Y = pooling2D(X)
     * (1) OH = (IH + 2*ph - FH)/sh + 1
     * (2) OW = (IW + 2*pw - FW)/sw + 1
     * (3) N: the batch size
     * (4) IC: channel num for Tensor X and Y
     *
     * For Back Prop: see {@link z.jcuda.la.CudaLA_unpool2D}
     *
     * -----------------------------------------------------------------------------
     * Use img2col to implements the 2D pooling, like a implicity Matrix multiply:
     *  X -> Tensor B[IC, FH*FW, N*OH*OW] -> A[GN, GK, GM]
     *  Y -> Matrix C[IC, N*OH*OW] -> B[GN, GM]
     *
     * As:<b>
     * (1) GN = IC
     * (2) GM = N*OH*OW
     * (3) GK = FH*FW, we can also use a double loop of(fh:FH, fw:FW), in instead of using
     * a single loop of (k:GK): fh = k/FW, fw=k%FW </b>
     *
     * -----------------------------------------------------------------------------
     * Make sure: <b>
     * (1) GN%4==0, GN>=4
     * (2) GM%4==0, GM>=4
     * (3) FH>=2, FW>=2
     * (4) oph = (FH - ph - 1 >= 0) -> (FH > ph)
     * (5) opw = (FW - pw - 1 >= 0) -> (FW > pw)
     * (6) to improve the performance: GN%8==0 || GM%8==0 </b>
     * </pre>
     */
    public static boolean LIKE_CONV3D = true;
    //</editor-fold>
}

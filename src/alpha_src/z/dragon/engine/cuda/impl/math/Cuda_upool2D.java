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
public final class Cuda_upool2D 
{
    private Cuda_upool2D () {}
    
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize(int IH, int IW, int N, int IC)
    {
        int GN = IC, GM = N * IH * IW;
        
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
    
    public static int[] getImg2colMatSize(int FH, int FW, int IH, int IW, int N, int IC)
    {
        int GN = IC;
        int GM = N * IH * IW;
        int GK = FH * FW;
        return new int[]{GN, GM, GK};
    }
    
    public static int[] getOutPaddingDim_deltaY(int FH, int FW, int ph, int pw) {
        int oph = FH - ph - 1;
        int opw = FH - pw - 1;
        return new int[]{oph, opw};
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="upool2D_max">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * (5) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 4, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 0.963000 msec, Performance = 1114.996582  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 1.471000 msec, Performance = 1094.910034  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 1.710000 msec, Performance = 1098.858521  GFlop/s
     * (4) IC = 240: Size = 0.937500, Time = 1.910000 msec, Performance = 1054.065796  GFlop/s
     * (5) IC = 248: Size = 0.968750, Time = 2.040000 msec, Performance = 1019.791443  GFlop/s
     * (6) IC = 252: Size = 0.984375, Time = 2.230000 msec, Performance =  947.950256  GFlop/s
     * (7) IC = 255: Size = 0.996094, Time = 1.900000 msec, Performance = 1125.839478  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 4, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.450000 msec, Performance = 740.511536  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.100000 msec, Performance = 766.958435  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 2.440000 msec, Performance = 770.101624  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 3.010000 msec, Performance = 702.302063  GFlop/s
    * 
    * </pre>
    * @param streamArray
    * @param length
    * @param d_deltaY_address
    * @param dY_address
    * @param OH
    * @param OW
    * @param FH
    * @param FW
    * @param d_deltaX_address
    * @param dX_address
    * @param IH
    * @param IW
    * @param N
    * @param IC
    * @param sh
    * @param sw
    * @param ph
    * @param pw 
    */
    @Passed
    public static native void upool2D_max(long[] streamArray, int length,
            long d_deltaY_address, long dY_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, long dX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="upool2D_max_Indexed">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * (5) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 4, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 0.963000 msec, Performance = 1114.996582  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 1.471000 msec, Performance = 1094.910034  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 1.710000 msec, Performance = 1098.858521  GFlop/s
     * (4) IC = 240: Size = 0.937500, Time = 1.910000 msec, Performance = 1054.065796  GFlop/s
     * (5) IC = 248: Size = 0.968750, Time = 2.040000 msec, Performance = 1019.791443  GFlop/s
     * (6) IC = 252: Size = 0.984375, Time = 2.230000 msec, Performance =  947.950256  GFlop/s
     * (7) IC = 255: Size = 0.996094, Time = 1.900000 msec, Performance = 1125.839478  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 4, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.450000 msec, Performance = 740.511536  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.100000 msec, Performance = 766.958435  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 2.440000 msec, Performance = 770.101624  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 3.010000 msec, Performance = 702.302063  GFlop/s
    * 
    * </pre>
    * @param streamArray
    * @param length
    * @param d_deltaY_address
    * @param dIndex_address
    * @param OH
    * @param OW
    * @param FH
    * @param FW
    * @param d_deltaX_address
    * @param IH
    * @param IW
    * @param N
    * @param IC
    * @param sh
    * @param sw
    * @param ph
    * @param pw 
    */
    @Passed
    public static native void upool2D_max_Indexed(long[] streamArray, int length,
            long d_deltaY_address, long dIndex_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="upool2D_avg">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * (5) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 0.942000 msec, Performance = 1139.853271  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 1.473000 msec, Performance = 1093.423218  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 1.715000 msec, Performance = 1095.654785  GFlop/s
     * (4) IC = 240: Size = 0.937500, Time = 1.871000 msec, Performance = 1076.037231  GFlop/s
     * (5) IC = 248: Size = 0.968750, Time = 1.936000 msec, Performance = 1074.573730  GFlop/s
     * (6) IC = 252: Size = 0.984375, Time = 1.999000 msec, Performance = 1057.493286  GFlop/s
     * (7) IC = 255: Size = 0.996094, Time = 1.925000 msec, Performance = 1111.218262  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.430000 msec, Performance = 750.868347  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.087000 msec, Performance = 771.735779  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 2.776000 msec, Performance = 761.501831  GFlop/s
    * 
    * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void upool2D_avg(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="upool2D_avg_tiled">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH = sh, FW = sw
     * (2) FH * FW >= 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4. GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 16, [FH, FW] = 4, [N] = 4, [sh, sw] = 4, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 3, [N] = 4, [sh, sw] = 3, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 2, [N] = 4, [sh, sw] = 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 4, [N] = 16, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.484497, Time = 1.550000 msec, Performance = 671.257751  GFlop/s
     * (2) IC = 192: Size = 0.726746, Time = 2.295000 msec, Performance = 680.032288  GFlop/s
     * (3) IC = 224: Size = 0.847870, Time = 2.981000 msec, Performance = 610.797241  GFlop/s
     * (4) IC = 240: Size = 0.908432, Time = 3.301000 msec, Performance = 590.985352  GFlop/s
     * (5) IC = 248: Size = 0.938713, Time = 3.717000 msec, Performance = 542.338135  GFlop/s
     * (6) IC = 252: Size = 0.953854, Time = 4.166000 msec, Performance = 491.691040  GFlop/s
     * (7) IC = 255: Size = 0.965209, Time = 3.022000 msec, Performance = 685.893616  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.430000 msec, Performance = 750.868347  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.087000 msec, Performance = 771.735779  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 2.776000 msec, Performance = 761.501831  GFlop/s
    * 
    * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void upool2D_avg_tiled(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="upool2D_avg_ignore_padding">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4. GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * (5) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 0.942000 msec, Performance = 1139.853271  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 1.473000 msec, Performance = 1093.423218  GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 1.715000 msec, Performance = 1095.654785  GFlop/s
     * (4) IC = 240: Size = 0.937500, Time = 1.871000 msec, Performance = 1076.037231  GFlop/s
     * (5) IC = 248: Size = 0.968750, Time = 1.936000 msec, Performance = 1074.573730  GFlop/s
     * (6) IC = 252: Size = 0.984375, Time = 1.999000 msec, Performance = 1057.493286  GFlop/s
     * (7) IC = 255: Size = 0.996094, Time = 1.925000 msec, Performance = 1111.218262  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.430000 msec, Performance = 750.868347  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.087000 msec, Performance = 771.735779  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 2.776000 msec, Performance = 761.501831  GFlop/s
    * 
    * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void upool2D_avg_ignore_padding(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="upool2D_avg_ignore_padding_tiled">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = upool2D(deltaY[N, OH, OW, OC], [FH, FW]).
     * (1) FH = sh, FW = sw
     * (2) FH * FW >= 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4. GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) sh * sw >= 2
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 16, [FH, FW] = 4, [N] = 4, [sh, sw] = 4, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 3, [N] = 4, [sh, sw] = 3, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 2, [N] = 4, [sh, sw] = 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 4, [N] = 16, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.484497, Time = 1.550000 msec, Performance = 671.257751  GFlop/s
     * (2) IC = 192: Size = 0.726746, Time = 2.295000 msec, Performance = 680.032288  GFlop/s
     * (3) IC = 224: Size = 0.847870, Time = 2.981000 msec, Performance = 610.797241  GFlop/s
     * (4) IC = 240: Size = 0.908432, Time = 3.301000 msec, Performance = 590.985352  GFlop/s
     * (5) IC = 248: Size = 0.938713, Time = 3.717000 msec, Performance = 542.338135  GFlop/s
     * (6) IC = 252: Size = 0.953854, Time = 4.166000 msec, Performance = 491.691040  GFlop/s
     * (7) IC = 255: Size = 0.965209, Time = 3.022000 msec, Performance = 685.893616  GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 3, [N] = 4, [sh, sw] = 1, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 8, [N] = 8, [sh, sw] = 4, [ph, pw] = 2
     * (1) IC = 128: Size = 0.500000, Time = 1.430000 msec, Performance = 750.868347  GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 2.087000 msec, Performance = 771.735779  GFlop/s
     * (3) IC = 252: Size = 0.984375, Time = 2.776000 msec, Performance = 761.501831  GFlop/s
    * 
    * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void upool2D_avg_ignore_padding_tiled(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW, 
            int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Document">
    /**
     * <pre>
     * Unpool2D. {@link z.jcuda.la.CudaLA_unpool2D}
     * It is really OK to regard unpooling as a special type of Deconvolution
     *
     * (1) X is a 4D tensor: X[N, IC, IH, IW]
     * (2) Y is a 4D tensor: Y[N, IC, OH, OW]
     * (3) pooling kernel size: FH, FW
     * (4) pooling step: sh, sw
     * (5) padding: ph, pw
     *
     * -----------------------------------------------------------------------------
     * Forward Prop: Y = pool2D(X), for detail see{@link  z.jcuda.la.CudaLA_pool2D}
     *
     * Back Prop: deltaX = unpool2D(deltaY)
     *                   = sepcial-deconv2d-for-each-channel(deltaY)
     * => average pooling: deltaX = sum-pool2D(deltaY_p)
     * => max pooling: delaX= sum-pooling2D({deltaY_p[index] * X[index].isMaxInForwardPropWithinAKernel? 0: 1})
     *
     * (1) IH = (OH - 1)*sh + FH - 2*ph
     * (2) IW = (OW - 1)*sw + FW - 2*pw
     * (3) deltaY_p[N, IC, OH_p, OW_p]
     *      => <b>OH_p = OH + iph = OH + (sh-1)*(OH-1)</b>
     *      => <b>OH_w = OW + ipw = OW + (sw-1)*(OW-1)</b>
     * (4) (OH_p + 2*oph - FH)/1 + 1 = IH
     *     (OW_p + 2*opw - FW)/1 + 1 = IW
     *      => <b>oph = FH - ph - 1</b>
     *      => <b>opw = FW - pw - 1</b>
     *
     * -----------------------------------------------------------------------------
     * Use Img2col to implements this algorithm:
     * (1) deltaY -> Tensor A[IC, FH*FW, N*IH*IW] = A[GN, GK, GM]
     * (2) deltaX -> Matrix B[IC, N*IH*IW]
     * As:<b>
     * (1) GN = IC
     * (2) GM = N*IH*IW
     * (3) GK = FH*FW</b>
     * -----------------------------------------------------------------------------
     * Make Sure:
     * (1) <b>GN%4==0, GN>=4</b>
     * (2) <b>GM%4==0, GM>=4</b>
     * (3) <b>FH>=2, FW>=2</b>
     * (4) <b>oph = (FH - ph - 1 >= 0) -> (FH > ph)</b>
     * (5) <b> opw = (FW - pw - 1 >= 0) -> (FW > pw)</b>
     *
     * </pre>
     */
    public static final boolean LIKE_CONV3D = true;
    //</editor-fold>
}
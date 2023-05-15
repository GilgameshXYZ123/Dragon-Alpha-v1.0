/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * Y[N, OH, OW, OC] = conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [ph, pw, sh, sw]).
 * @author Gilgamesh
 */
public final class Cuda_conv3D 
{
    private Cuda_conv3D() {}
    
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static int blockNum(int OH, int OW, int N, int OC) 
    {
        int GN = OC, GM = N * OH * OW;//get img2col size

        int bn = 0,bm = 0;
        for(;;) {
            if(GN > 127) bn += GN >> 7; GN &= 127; if(GN == 0) break;//2^7
            if(GN >  63) bn += 1; GN &=  63; if(GN == 0) break;//2^6
            if(GN >  31) bn += 1; GN &=  31; if(GN == 0) break;//2^5
            if(GN >  15) bn += 1; GN &=  15; if(GN == 0) break;//2^4
            if(GN >   7) bn += 1; GN &=   7; if(GN == 0) break;//2^3
            if(GN >   3) bn += 1; break;
        }
      
        for(;;) {
            if(GM > 127) bm += GM >> 7; GM &= 127; if(GM == 0) break;//2^7
            if(GM >  63) bm += 1; GM &=  63; if(GM == 0) break;//2^6
            if(GM >  31) bm += 1; GM &=  31; if(GM == 0) break;//2^5
            if(GM >  15) bm += 1; GM &=  15; if(GM == 0) break;//2^4
            if(GM >   7) bm += 1; GM &=   7; if(GM == 0) break;//2^3
            if(GM >   3) bm += 1; break;
        }
        return bn * bm;
    }
    
    public static int blockNumV2(int OH, int OW, int N, int OC) 
    {
        int GN = OC, GM = N;//get img2col size

        int bn = 0,bm = 0;
        for(;;) {
            if(GN > 127) bn += GN >> 7; GN &= 127; if(GN == 0) break;//2^7
            if(GN >  63) bn += 1; GN &=  63; if(GN == 0) break;//2^6
            if(GN >  31) bn += 1; GN &=  31; if(GN == 0) break;//2^5
            if(GN >  15) bn += 1; GN &=  15; if(GN == 0) break;//2^4
            if(GN >   7) bn += 1; GN &=   7; if(GN == 0) break;//2^3
            if(GN >   3) bn += 1; break;
        }
      
        for(;;) {
            if(GM > 127) bm += GM >> 7; GM &= 127; if(GM == 0) break;//2^7
            if(GM >  63) bm += 1; GM &=  63; if(GM == 0) break;//2^6
            if(GM >  31) bm += 1; GM &=  31; if(GM == 0) break;//2^5
            if(GM >  15) bm += 1; GM &=  15; if(GM == 0) break;//2^4
            if(GM >   7) bm += 1; GM &=   7; if(GM == 0) break;//2^3
            if(GM >   3) bm += 1; break;
        }
        return bn * bm * OH * OW;//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize(int OH, int OW, int N, int OC)
    {
        int GN = OC, GM = N * OH * OW;//get img2col size

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
        return (size < 10? size : 10);//the max stream size is 10
    }
    
    public static int streamSizeV2(int N, int OC)
    {
        int GN = OC, GM = N;//get img2col size
        
        int sn = 0, sm = 0;
        if((GN & (GN - 1)) ==0 ) sn = 1;
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
        return (size < 10? size : 10);//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static double paddingScaleUp(int IH, int IW, int OH, int OW,
            int FH, int FW, int sh, int sw)
    {
        int input_size = IH * IW;
        int padding_size = ((OH - 1)*sh + FH) * ((OW - 1)*sw + FW);
        return 1.0 * padding_size / input_size;
    }
    
    public static int[] getImg2colMatrixDim(
            int OH, int OW, 
            int FH, int FW, 
            int N, int IC, int OC) 
    {
        int GN = OC;
        int GM = N  * OH * OW;
        int GK = FH * FW * IC;
        return new int[]{GN, GM, GK};
    }
    
    public static int[] outputTensorDim(
            int IH, int IW, 
            int FH, int FW, 
            int N, int OC,
            int sh, int sw, int ph, int pw)
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{ N, OH, OW, OC };
    }
    
    public static int[] getInputTensorDim(
            int OH, int OW, 
            int FH, int FW, 
            int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        return new int[]{N, IH, IW, IC};
    }
    //</editor-fold>
 
    //<editor-fold defaultstate="collapsed" desc="conv3D">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.478000 msec, Performance = 1452.965942 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.297000 msec, Performance = 1402.362061 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 3.024000 msec, Performance = 1242.756714 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.783000 msec, Performance = 1064.375366 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.354000 msec, Performance =  955.615417 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.265000 msec, Performance =  803.012085 GFlop/s
     * 
     *  [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.912000 msec, Performance = 1324.517090 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.389000 msec, Performance = 1304.491943 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.812000 msec, Performance = 1166.627563 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.267000 msec, Performance =  999.084351 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.602000 msec, Performance =  899.470276 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.691000 msec, Performance = 1428.692505 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.925000 msec, Performance = 1115.575928 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 3.140000 msec, Performance = 1025.867920 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 4.220000 msec, Performance =  890.544189 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 4.930000 msec, Performance =  816.740784 GFlop/s
     * (5) OC = 244: Size = 1.906250, Time = 5.820000 msec, Performance =  703.374695 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.301000 msec, Performance = 1458.254272 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.668000 msec, Performance = 1372.182373 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.896000 msec, Performance = 1199.351685 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 6.301000 msec, Performance =  998.485291 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 7.447000 msec, Performance =  872.991943 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.456000 msec, Performance = 1506.033813 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_texture">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.476000 msec, Performance = 1454.934814 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.307000 msec, Performance = 1396.283325 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.999000 msec, Performance = 1253.116455 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.741000 msec, Performance = 1076.325073 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.314000 msec, Performance =  964.476013 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.257000 msec, Performance =  804.234070 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.903000 msec, Performance = 1337.718262 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.375000 msec, Performance = 1317.774048 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.791000 msec, Performance = 1180.306641 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.239000 msec, Performance = 1011.578430 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.598000 msec, Performance =  900.855103 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.678000 msec, Performance = 1439.761108 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.905000 msec, Performance = 1334.761963 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.388000 msec, Performance = 1305.431763 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.808000 msec, Performance = 1169.208618 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.239000 msec, Performance = 1011.578430 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.598000 msec, Performance =  900.855103 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.681000 msec, Performance = 1437.191650 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.270000 msec, Performance = 1478.168823 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.625000 msec, Performance = 1388.459229 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.766000 msec, Performance = 1232.065796 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 6.032000 msec, Performance = 1043.013306 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 7.162000 msec, Performance =  907.731201 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.477000 msec, Performance = 1498.969360 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_texture(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3DV2">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     *  [IH, IW] = 8, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 128: Size = 1.125000, Time = 1.655000 msec, Performance = 1459.769897 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.750000 msec, Performance = 1317.774048 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 3.727000 msec, Performance = 1134.386475 GFlop/s
     * (5) OC = 240: Size = 2.109375, Time = 4.608000 msec, Performance =  983.040039 GFlop/s
     * (8) OC = 256: Size = 2.250000, Time = 2.960000 msec, Performance = 1632.377808 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 1.125000, Time = 1.328000 msec, Performance = 1819.216309 GFlop/s
     * (2) OC = 576: Size = 1.265625, Time = 1.634000 msec, Performance = 1663.347046 GFlop/s
     * (3) OC = 608: Size = 1.335938, Time = 1.890000 msec, Performance = 1517.938599 GFlop/s
     * (4) OC = 624: Size = 1.371094, Time = 2.135000 msec, Performance = 1379.110718 GFlop/s
     * (7) OC = 512, IC = 256: Size = 2.250000, Time = 2.521000 msec, Performance = 1916.635620 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 2.000000, Time = 1.667000 msec, Performance = 2576.465088 GFlop/s
     * (2) OC = 576: Size = 2.250000, Time = 2.141000 msec, Performance = 2256.813721 GFlop/s
     * (3) OC = 608: Size = 2.375000, Time = 2.511000 msec, Performance = 2031.172363 GFlop/s
     * (4) OC = 624: Size = 2.437500, Time = 2.867000 msec, Performance = 1825.772949 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 3.125000, Time = 2.624000 msec, Performance = 2557.502441 GFlop/s
     * (2) OC = 576: Size = 3.515625, Time = 3.402000 msec, Performance = 2219.208496 GFlop/s
     * (3) OC = 608: Size = 3.710938, Time = 4.004000 msec, Performance = 1990.303955 GFlop/s
     * (4) OC = 624: Size = 3.808594, Time = 4.563000 msec, Performance = 1792.437500 GFlop/s
     * 
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3DV2(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_np">
    /**
     * <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * (5) ph = pw = 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 35, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.468000 msec, Performance = 1462.863525 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.263000 msec, Performance = 1423.431519 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 3.039000 msec, Performance = 1236.622681 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.877000 msec, Performance = 1038.568970 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.329000 msec, Performance =  961.134094 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 4.963000 msec, Performance =  851.875610 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     */
    @Passed
    public static native void conv3D_np(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_W1">
    /**
     *  <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (2) GM % 4 ==0, GM >= 4
     * (2) GN % 4 ==0, GN >= 4
     * (3) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 8, 128
     * (1) OC = 128: Size = 0.500000, Time = 0.902000 msec, Performance = 1190.401123 GFlop/s
     * (2) OC = 192: Size = 0.750000, Time = 1.448000 msec, Performance = 1112.301636 GFlop/s
     * (3) OC = 224: Size = 0.875000, Time = 1.816000 msec, Performance = 1034.718140 GFlop/s
     * (4) OC = 240: Size = 0.937500, Time = 2.301000 msec, Performance =  874.952576 GFlop/s
     * (5) OC = 248: Size = 0.968750, Time = 2.602000 msec, Performance =  799.529114 GFlop/s
     * (6) OC = 252: Size = 0.984375, Time = 2.946000 msec, Performance =  717.559082 GFlop/s
     * (7) OC = 255: Size = 0.996094, Time = 1.634000 msec, Performance = 1309.115723 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     *  [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * (1) OC = 128: Size = 0.250000, Time = 0.970000 msec, Performance = 553.475159 GFlop/s
     * (2) OC = 192: Size = 0.375000, Time = 1.430000 msec, Performance = 563.151306 GFlop/s
     * (3) OC = 252: Size = 0.492188, Time = 2.800000 msec, Performance = 377.487366 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dY_address
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed 
    public static native void conv3D_W1(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address,
            long dY_address,
            int N, int IC, int OC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="kernel_remode">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC].
     * </pre>
     * @param stream_address
     * @param dW_address
     * @param dCW_addres
     * @param FH
     * @param FW
     * @param OC
     * @param IC 
     */
    public static native void kernel_remode(long stream_address,
            long dW_address, long dCW_addres,
            int FH, int FW, int OC, int IC);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [8, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.462000 msec, Performance = 1468.867065 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.249000 msec, Performance = 1432.292236 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.834000 msec, Performance = 1326.074829 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.580000 msec, Performance = 1124.729614 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 4.170000 msec, Performance =  997.781677 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 5.093000 msec, Performance = 830.131226 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 0.562500, Time = 0.914000 msec, Performance = 1321.618774 GFlop/s
     * (2) OC = 192: Size = 0.843750, Time = 1.379000 msec, Performance = 1313.951660 GFlop/s
     * (3) OC = 224: Size = 0.984375, Time = 1.751000 msec, Performance = 1207.269653 GFlop/s
     * (4) OC = 240: Size = 1.054688, Time = 2.212000 msec, Performance = 1023.925964 GFlop/s
     * (5) OC = 248: Size = 1.089844, Time = 2.601000 msec, Performance =  899.816040 GFlop/s
     * (6) OC = 256: Size = 1.125000, Time = 1.679000 msec, Performance = 1438.903564 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.228000 msec, Performance = 1506.033813 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.496000 msec, Performance = 1439.692383 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.488000 msec, Performance = 1308.383667 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 5.754000 msec, Performance = 1093.405640 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 6.851000 msec, Performance =  948.937561 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.385000 msec, Performance = 1530.418701 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_GemmR(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmR_texture">
    /**
     * <pre>
     *  Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [64, 64]
     * (1) OC = 128: Size = 1.125000, Time = 1.603000 msec, Performance = 1507.123535 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.421000 msec, Performance = 1496.851929 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 2.996000 msec, Performance = 1411.167725 GFlop/s
     * (4) OC = 240: Size = 2.109375, Time = 3.662000 msec, Performance = 1236.987549 GFlop/s
     * (5) OC = 248: Size = 2.179688, Time = 4.187000 msec, Performance = 1117.946899 GFlop/s
     * (6) OC = 252: Size = 2.214844, Time = 4.750000 msec, Performance = 1001.334900 GFlop/s
     * (7) OC = 256: Size = 2.250000, Time = 3.051000 msec, Performance = 1583.689941 GFlop/s
     * => OC =  64: Size = 0.562500, Time = 1.035000 msec, Performance = 1167.110718 GFlop/s
     * => OC = 136: Size = 1.195313, Time = 2.186000 msec, Performance = 1174.251587 GFlop/s
     * => OC = 144: Size = 1.265625, Time = 2.225000 msec, Performance = 1221.532227 GFlop/s
     * => OC = 160: Size = 1.406250, Time = 2.242000 msec, Performance = 1346.966431 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.000000, Time = 1.459000 msec, Performance = 1471.887329 GFlop/s
     * (2) OC = 192: Size = 1.500000, Time = 2.214000 msec, Performance = 1454.934692 GFlop/s
     * (3) OC = 224: Size = 1.750000, Time = 2.795000 msec, Performance = 1344.578247 GFlop/s
     * (4) OC = 240: Size = 1.875000, Time = 3.274000 msec, Performance = 1229.850952 GFlop/s
     * (5) OC = 248: Size = 1.937500, Time = 3.774000 msec, Performance = 1102.477417 GFlop/s
     * (6) OC = 252: Size = 1.968750, Time = 4.275000 msec, Performance =  988.972717 GFlop/s
     * 
     * [IH, IW] = 32, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [32, 64]
     * (1) OC = 128: Size = 1.562500, Time = 2.226000 msec, Performance = 1507.386841 GFlop/s
     * (2) OC = 192: Size = 2.343750, Time = 3.303000 msec, Performance = 1523.816162 GFlop/s
     * (3) OC = 224: Size = 2.734375, Time = 4.160000 msec, Performance = 1411.544678 GFlop/s
     * (4) OC = 240: Size = 2.929688, Time = 5.049000 msec, Performance = 1246.079712 GFlop/s
     * (5) OC = 248: Size = 3.027344, Time = 5.759000 msec, Performance = 1128.871582 GFlop/s
     * (6) OC = 256: Size = 3.125000, Time = 4.230000 msec, Performance = 1586.497925 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_GemmR_texture(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_GemmV2R">
    /**
     * <pre>
     * Remode: W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], CW[FH, FW, IC, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GM % 4 ==0, GM >= 4
     * (3) GN % 4 ==0, GN >= 4
     * (4) GK % 4 ==0, GK >= 8
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [IH, IW] = 33, [FH, FW] = 4, [sh, sw] = 1, [ph, pw] = 1, [N, IC] = [4, 16]
     * [IH, IW] = 32, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [4, 16]
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 8, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 128: Size = 1.125000, Time = 1.548000 msec, Performance = 1560.671265 GFlop/s
     * (2) OC = 192: Size = 1.687500, Time = 2.397000 msec, Performance = 1511.839233 GFlop/s
     * (3) OC = 224: Size = 1.968750, Time = 3.294000 msec, Performance = 1283.502930 GFlop/s
     * (5) OC = 240: Size = 2.109375, Time = 4.249000 msec, Performance = 1066.097534 GFlop/s
     * (6) OC = 248: Size = 2.179688, Time = 5.380000 msec, Performance =  870.045227 GFlop/s
     * (7) OC = 252: Size = 2.214844, Time = 6.715000 msec, Performance =  708.315796 GFlop/s
     * (8) OC = 256: Size = 2.250000, Time = 2.687000 msec, Performance = 1798.227783 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 3, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 1.125000, Time = 1.334000 msec, Performance = 1811.033813 GFlop/s
     * (2) OC = 576: Size = 1.265625, Time = 1.551000 msec, Performance = 1752.359131 GFlop/s
     * (3) OC = 608: Size = 1.335938, Time = 1.826000 msec, Performance = 1571.141235 GFlop/s
     * (4) OC = 624: Size = 1.371094, Time = 2.066000 msec, Performance = 1425.170044 GFlop/s
     * (5) OC = 632: Size = 1.388672, Time = 2.292000 msec, Performance = 1301.112549 GFlop/s
     * (6) OC = 636: Size = 1.397461, Time = 2.612000 msec, Performance = 1148.937378 GFlop/s
     * (7) OC = 512, IC = 256: Size = 2.250000, Time = 2.521000 msec, Performance = 1916.635620 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 4, [sh, sw] = 2, [ph, pw] = 1, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 2.000000, Time = 1.878000 msec, Performance = 2286.989990 GFlop/s
     * (2) OC = 576: Size = 2.250000, Time = 2.277000 msec, Performance = 2122.019531 GFlop/s
     * (3) OC = 608: Size = 2.375000, Time = 2.576000 msec, Performance = 1979.919922 GFlop/s
     * (4) OC = 624: Size = 2.437500, Time = 2.886000 msec, Performance = 1813.753174 GFlop/s
     * (5) OC = 632: Size = 2.468750, Time = 3.270000 msec, Performance = 1621.284546 GFlop/s
     * (6) OC = 636: Size = 2.484375, Time = 3.800000 msec, Performance = 1403.988037 GFlop/s
     * 
     * [IH, IW] = 4, [FH, FW] = 5, [sh, sw] = 2, [ph, pw] = 2, [N, IC] = [512, 128]
     * (1) OC = 512: Size = 3.125000, Time = 2.724000 msec, Performance = 2463.614746 GFlop/s
     * (2) OC = 576: Size = 3.515625, Time = 3.384000 msec, Performance = 2231.012695 GFlop/s
     * (3) OC = 608: Size = 3.710938, Time = 3.867000 msec, Performance = 2060.816406 GFlop/s
     * (4) OC = 624: Size = 3.808594, Time = 4.397000 msec, Performance = 1860.107544 GFlop/s
     * (5) OC = 632: Size = 3.857422, Time = 4.953000 msec, Performance = 1672.471313 GFlop/s
     * (6) OC = 636: Size = 3.881836, Time = 5.837000 msec, Performance = 1428.161621 GFlop/s
     * 
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param FH
     * @param FW
     * @param dY_address
     * @param OH
     * @param OW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void conv3D_GemmV2R(boolean useTexture, long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address, int FH, int FW,
            long dY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="conv3D_W1">
    /**
     *  <pre>
     * Y[N, OH, OW, OC] = Conv3D(X[N, IH, IW, IC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (2) GM % 4 ==0, GM >= 4
     * (2) GN % 4 ==0, GN >= 4
     * (3) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 4, 16
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 64, [FH, FW] = 1, [sh, sw] = 1, [ph, pw] = 0, [N, IC] = 8, 128
     * (1) OC = 128: Size = 0.500000, Time = 0.871000 msec, Performance = 1232.769043 GFlop/s
     * (2) OC = 192: Size = 0.750000, Time = 1.202000 msec, Performance = 1339.943970 GFlop/s
     * (3) OC = 224: Size = 0.875000, Time = 1.580000 msec, Performance = 1189.270996 GFlop/s
     * (4) OC = 240: Size = 0.937500, Time = 2.059000 msec, Performance =  977.788208 GFlop/s
     * (5) OC = 248: Size = 0.968750, Time = 2.412000 msec, Performance =  862.510315 GFlop/s
     * (6) OC = 252: Size = 0.984375, Time = 2.683000 msec, Performance =  787.897583 GFlop/s
     * (7) OC = 255: Size = 0.996094, Time = 1.892000 msec, Performance = 1130.599976 GFlop/s
     * (8) OC = 256: Size = 1.000000, Time = 1.560000 msec, Performance = 1376.592163 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param dW_address
     * @param dCW_address
     * @param dY_address
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed 
    public static native void conv3D_GemmR_W1(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long dW_address, long dCW_address,
            long dY_address,
            int N, int IC, int OC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Document">
    /**
     * <pre>
     * 3D Convolution. 
     * X(N, IH, IW, IC): the input Tensor
     * Y(N, OH, OW, OC): the output tensor
     * W(OC, FH, FW, IC): a 4D covolution kernel
     * (1) N: the batch size, how many samples in this batch
     * (2) IC: input channel number of X.
     *  => each element of X is a 3D tensor(Z: IC, X: IH, Y: IW), like a colorful pitcure
     * (3) OC: output channel number of Y.
     *  => each element of Y is a 3D tensor(Z: OC, X: OH, Y: OW)
     * (4) each element of W is a 3D tensor(a 3D convlution kernel), with its dimension(Z: IC, Y: FH, X: FW)
     * (5) Padding:
     *  => ph: the padding on Y axis
     *  => pw: the padding on X axis
     * (5) Stride: to move the kernel on each element of X
     *  => sh: the step on Y axis
     *  => sw: the step on X axis
     * (6) OH = (IH + 2*ph - FH)/sh + 1
     *     OW = (IW + 2*pw - FW)/sw + 1
     *  kindly let: (IH + 2*ph - FH)%sh==0, (IW + 2*pw - FW)%sw == 0,
     *  if not the result is also correct, but may has some bad effect on backprop of CNN.
     *
     * Use "Img2col Algorithm" to implement the 3D convolution as an implicit Matrix Multiply:
     * (1) GN = OC;
     * (2) GM = N * OH * OW;
     * (3) GK = IC * FH * FW;
     * W -> Matrix A[GN, GK] = [OC, IC*FH*FW(the size of each patch on X)]
     * X -> Matrix B[GK, GM] = [IC*FH*FW, N*OH*OW(the times of patchs on X)]
     * Y -> Matrix C[GN, GM] = [OC, N*OH*OW]
     * Y = conv3D(X, W) <-> C = A*B
     *
     * The img2co algorithm js just implemented logically, by a specific way to fetch  memory, not physically.
     * Make Sure: <b>
     * (1) FH>=2, FW>=2
     * (2) GN%4==0, GN>4
     * (3) GM%4==0, GM>4
     * (4) GK>=4</b>
     * </pre>
     */
    public static final boolean IM2COL_GEMM = true;
    //</editor-fold>
}

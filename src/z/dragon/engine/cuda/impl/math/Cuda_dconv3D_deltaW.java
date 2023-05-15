/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
 * @author Gilgamesh
 */
public final class Cuda_dconv3D_deltaW 
{
    private Cuda_dconv3D_deltaW() {}
    
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static int blockNum(int FH, int FW, int OC, int IC)
    {
        int GN = OC, GM = FH * FW * IC;//get Img2col size
        
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
    
    public static int blockNumV2(int FH, int FW, int OC, int IC)
    {
        int GN = OC, GM = IC;//get Img2col size
        
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
        
        return bn * bm * FH * FW;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize(int FH, int FW, int OC, int IC)
    {
        int GN = OC, GM = FH * FW * IC;//get Img2col size
        
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
        return (size < 12 ? size : 12);//the max stream size is 12
    }
    
    public static int streamSizeV2(int OC, int IC)
    {
        int GN = OC, GM = IC;//get Img2col size
        
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
        return (size < 12 ? size : 12);//the max stream size is 12
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
    
    public static int[] getOutputTensorDim(
            int IH, int IW, 
            int FH, int FW, 
            int N, int OC,
            int sh, int sw, int ph, int pw)
    {
        int OH = (IH + (ph << 1) - FH) / sh + 1;
        int OW = (IW + (pw << 1) - FW) / sw + 1;
        return new int[]{N, OH, OW, OC};
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
    
    public static int[] getImg2colMatrixDim(
            int FH, int FW, 
            int OH, int OW,
            int N, int OC, int IC) 
    {
        int GN = OC;
        int GM = FH * FW * IC;
        int GK = N  * OH * OW;
        return new int[]{GN, GM, GK};
    }
    
    public static int[] getImg2colMatrixDim_Logically(
            int OH, int OW, 
            int FH, int FW,
            int N, int OC, int IC, 
            int sh, int sw) 
    {
        int OHp = OH + (OH - 1) * (sh - 1);
        int OWp = OW + (OW - 1) * (sw - 1);
        int GN = OC;
        int GM = IC * FH * FW;
        int GK0 = N * OHp * OWp;
        return new int[]{GN, GM, GK0};
    }
    
    public static int[] getOutPaddingShape_X(int ph, int pw) {
        int oph = ph;
        int opw = pw;
        return new int[]{oph, opw};
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_Gemm">
    /**
     * <pre>
     * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4, GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW] = 4, [N, IC] = 4, 32, [sh, sw] = 2, [ph, pw] = 1
     * for OC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 4, [N, IC] = 4, 128, [sh, sw] = 2, [ph, pw] = 1
     * for OC from 1 to 64: correct
     * (1) OC = 128: Time = 1.634000, Size = 1.000000, Performance = 1314.249512 GFlop/s, Size0 = 3.875977, Performance0 = 5094.000000 GFlop/s
     * (2) OC = 192: Time = 2.366000, Size = 1.500000, Performance = 1361.464600 GFlop/s, Size0 = 5.813965, Performance0 = 5277.004883 GFlop/s
     * (3) OC = 224: Time = 2.696000, Size = 1.750000, Performance = 1393.952515 GFlop/s, Size0 = 6.782959, Performance0 = 5402.927246 GFlop/s
     * (4) OC = 240: Time = 3.718000, Size = 1.875000, Performance = 1082.983154 GFlop/s, Size0 = 7.267456, Performance0 = 4197.617676 GFlop/s
     * (5) OC = 248: Time = 3.938000, Size = 1.937500, Performance = 1056.564087 GFlop/s, Size0 = 7.509705, Performance0 = 4095.217285 GFlop/s
     * (6) OC = 252: Time = 4.780000, Size = 1.968750, Performance =  884.489136 GFlop/s, Size0 = 7.630829, Performance0 = 3428.258789 GFlop/s
     * (7) OC = 255: Time = 2.794000, Size = 1.992188, Performance = 1531.206177 GFlop/s, Size0 = 7.721672, Performance0 = 5934.919434 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW] = 4, [N, OC] = 4, 128, [sh, sw] = 2, [ph, pw] = 1
     * for OC from 1 to 128: correct
     * 
     * [OH, OW] = 32, [FH, FW] = 4, [N, OC] = 4, 128, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 128: Time = 1.150000, Size = 0.500000, Performance = 933.688477 GFlop/s, Size0 = 1.937988, Performance0 = 3618.954590 GFlop/s
     * (2) OC = 252: Time = 3.250000, Size = 0.984375, Performance = 650.439758 GFlop/s, Size0 = 3.815414, Performance0 = 2521.089111 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param d_deltaW_address
     * @param FH
     * @param FW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public native static void dconv3D_deltaW(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long d_deltaY_address, int OH, int OW,
            long d_deltaW_address, int FH, int FW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_W1">
    /**
     * <pre>
     * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (3) GM % 4 ==0, GM >= 4
     * (4) GN % 4 ==0, GN >= 4
     * (5) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 15, [FH, FW = 1, [N, IC] = 2, 128, [sh, sw] = 1, [ph, pw] = 0
     *  for OC from 1 to 128: correct
     * 
     * [OH, OW] = 63, [FH, FW = 1, [N, IC] = 4, 128, [sh, sw] = 1, [ph, pw] = 0
     * (1) OC = 128: Time = 0.630000, Size = 0.242249, Performance = 825.753540 GFlop/s
     * (2) OC = 192: Time = 0.890000, Size = 0.363373, Performance = 876.783325 GFlop/s
     * (3) OC = 224: Time = 0.910000, Size = 0.363373, Performance = 857.513306 GFlop/s
     * (4) OC = 240: Time = 1.190000, Size = 0.454216, Performance = 819.681824 GFlop/s
     * (5) OC = 248: Time = 1.380000, Size = 0.469357, Performance = 730.387939 GFlop/s
     * (6) OC = 252: Time = 1.540000, Size = 0.476927, Performance = 665.059998 GFlop/s
     * (7) OC = 255: Time = 1.180000, Size = 0.482605, Performance = 878.292603 GFlop/s
     * [OH, OW] = 62, [FH, FW = 1, [N, IC] = 4, 128, [sh, sw] = 1, [ph, pw] = 1
     * (1) OC = 128: Time = 0.585000, Size = 0.234619, Performance = 861.266174 GFlop/s
     * (2) OC = 192: Time = 0.864000, Size = 0.351929, Performance = 874.723511 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 15, [FH, FW = 1, [N, IC] = 2, 128, [sh, sw] = 1, [ph, pw] = 0
     * for OC from 1 to 128: correct
     * 
     * [OH, OW] = 62, [FH, FW = 1, [N, IC] = 4, 128, [sh, sw] = 1, [ph, pw] = 0
     * (1) OC = 128: Time = 1.144000, Size = 0.234619, Performance = 440.420197 GFlop/s
     * (2) OC = 192: Time = 1.651000, Size = 0.351929, Performance = 457.759583 GFlop/
     * (3) OC = 252: Time = 2.558000, Size = 0.461906, Performance = 387.778137 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param d_deltaW_address
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed
    public native static void dconv3D_deltaW_W1(long[] streamArray, int length,
            long dX_address, int IH, int IW,
            long d_deltaY_address, 
            long d_deltaW_address,
            int N, int IC, int OC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="GemmSK-Reducer">
    //part = GridZ - 1
    public static native void buf_summary(long stream_address,
            long d_deltaW_buf_address,
            long d_deltaW_address,
            int part, int sizeW);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_GemmSK">
    /**
     * <pre>
     * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4, GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 14, [OH, OW] =  7, [FH, FW] = 4, [N, IC] = 16, 32, [sh, sw] = 2, [ph, pw] = 1
     * [IH, IW] = 13, [OH, OW] =  7, [FH, FW] = 3, [N, IC] = 16, 32, [sh, sw] = 2, [ph, pw] = 1
     * [IH, IW] = 32, [OH, OW] = 16, [FH, FW] = 4, [N, OC] = 16, 32, [sh, sw] = 1, [ph, pw] = 1
     * [IH, IW] = 31, [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 16, 32, [sh, sw] = 1, [ph, pw] = 1
     * for OC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [FH, FW] = 4, [N, IC] = 32, 64, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 128: Time = 1.620000, Size = 1.000000, Performance = 1325.607056 GFlop/s
     * (2) OC = 192: Time = 2.308000, Size = 1.500000, Performance = 1395.678101 GFlop/s
     * (3) OC = 224: Time = 2.776000, Size = 1.750000, Performance = 1353.781006 GFlop/s
     * (4) OC = 240: Time = 3.188000, Size = 1.875000, Performance = 1263.027466 GFlop/s
     * (5) OC = 248: Time = 3.574000, Size = 1.937500, Performance = 1164.171631 GFlop/s
     * (6) OC = 252: Time = 3.974000, Size = 1.968750, Performance = 1063.879883 GFlop/s
     * (7) OC = 256: Time = 2.918000, Size = 2.000000, Performance = 1471.887207 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param GridZ
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param d_deltaW_address
     * @param d_deltaW_buf_address
     * @param FH
     * @param FW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public native static void dconv3D_deltaW_GemmSK(long[] streamArray, int length, int GridZ,
            long dX_address, int IH, int IW,
            long d_deltaY_address, int OH, int OW,
            long d_deltaW_address, 
            long d_deltaW_buf_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_GemmV2SK">
    /**
     * <pre>
     * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) GN >= 4, GN % 4 == 0
     * (3) GM >= 4, GM % 4 == 0
     * (4) GK >= 8, GK % 4 == 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [IH, IW] = 14, [OH, OW] = 7, [FH, FW] = 4, [N, IC, OC] = 16, 32, 32:
     * [IH, IW] = 10, [OH, OW] = 5, [FH, FW] = 4, [N, IC, OC] = 32, 32, 32:
     * for ic from 1 to 128: correct
     * for oc from 1 to 128: correct
     * 
     * [IH, IW] =  4, [FH, FW] = 3, [N, IC] = 512, 256, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 512: Time = 2.438000, Size = 2.250000, Performance = 1981.885864 GFlop/s
     * (2) OC = 576: Time = 2.728000, Size = 2.531250, Performance = 1992.601807 GFlop/s
     * (3) OC = 608: Time = 2.896000, Size = 2.671875, Performance = 1981.287231 GFlop/s
     * (4) OC = 624：Time = 3.072000, Size = 2.742188, Performance = 1916.927856 GFlop/s
     * (5) OC = 632: Time = 3.332000, Size = 2.777344, Performance = 1790.005981 GFlop/s
     * (6) OC = 636: Time = 3.514000, Size = 2.794922, Performance = 1708.038818 GFlop/s
     * 
     * [IH, IW] =  8, [FH, FW] = 3, [N, OC] = 128, 256, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 512: Time = 2.786000, Size = 2.250000, Performance = 1734.328003 GFlop/s
     * (2) OC = 576: Time = 3.172000, Size = 2.531250, Performance = 1713.687744 GFlop/s
     * (3) OC = 608: Time = 3.416000, Size = 2.671875, Performance = 1679.686035 GFlop/s
     * (4) OC = 624: Time = 3.610000, Size = 2.742188, Performance = 1631.247192 GFlop/s
     * (5) OC = 632: Time = 3.836000, Size = 2.777344, Performance = 1554.822754 GFlop/s
     * (6) OC = 636: Time = 4.090000, Size = 2.794922, Performance = 1467.493530 GFlop/s
     * 
     * [IH, IW] = 16, [FH, FW] = 3, [N, OC] = 32, 128, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 512: Time = 3.010000, Size = 2.250000, Performance = 1605.261841 GFlop/s
     * (2) OC = 576: Time = 3.438000, Size = 2.531250, Performance = 1581.098755 GFlop/s
     * (3) OC = 608: Time = 3.688000, Size = 2.671875, Performance = 1555.804688 GFlop/s
     * (4) OC = 624: Time = 3.908000, Size = 2.742188, Performance = 1506.858398 GFlop/s
     * (5) OC = 632: Time = 4.180000, Size = 2.777344, Performance = 1426.865967 GFlop/s
     * (6) OC = 636: Time = 4.348000, Size = 2.794922, Performance = 1380.416016 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param GridZ
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param d_deltaW_address
     * @param d_deltaW_buf_address
     * @param FH
     * @param FW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public native static void dconv3D_deltaW_GemmV2SK(long[] streamArray, int length, int GridZ,
            long dX_address, int IH, int IW,
            long d_deltaY_address, int OH, int OW,
            long d_deltaW_address, 
            long d_deltaW_buf_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaW_GemmSK_W1">
    /**
     * <pre>
     * deltaW[OC, FH, FW, IC] = dconv3D(X[N, IH, IW, IC], deltaY[N, OH, OW, OC], [sh, sw], [ph, pw]).
     * (1) FH = FW = 1
     * (2) ph = pw = 0, sh = sw = 1
     * (3) GM % 4 ==0, GM >= 4
     * (4) GN % 4 ==0, GN >= 4
     * (5) GK % 4 ==0, GK >= 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 16, [FH, FW] = 1, [N, IC] = 32, 64, [sh, sw] = 1, [ph, pw] = 0
     * for OC from 1 to 128: correct
     * 
     * [OH, OW] = 16, [FH, FW] = 1, [N, IC] = 64, 128, [sh, sw] = 1, [ph, pw] = 1
     * (1) OC = 128: Time = 0.548000, Size = 0.250000, Performance =  979.691467 GFlop/s
     * (2) OC = 192: Time = 0.812000, Size = 0.375000, Performance =  991.756531 GFlop/s
     * (3) OC = 224: Time = 0.894000, Size = 0.437500, Performance = 1050.921753 GFlop/s
     * (4) OC = 240: Time = 1.024000, Size = 0.468750, Performance =  983.039856 GFlop/s
     * (5) OC = 248: Time = 1.146000, Size = 0.484375, Performance =  907.667786 GFlop/s
     * (6) OC = 252: Time = 1.186000, Size = 0.492188, Performance =  891.201172 GFlop/s
     * (7) OC = 255: Time = 0.960000, Size = 0.500000, Performance = 1118.480957 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param GridZ
     * @param dX_address
     * @param IH
     * @param IW
     * @param d_deltaY_address
     * @param d_deltaW_address
     * @param d_deltaW_address_buf
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed
    public native static void dconv3D_deltaW_GemmSK_W1(long[] streamArray, int length, int GridZ,
            long dX_address, int IH, int IW,
            long d_deltaY_address, 
            long d_deltaW_address,
            long d_deltaW_address_buf,
            int N, int IC, int OC);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Documents">
    /**
     * <pre>
     * Deconvolution 3D for deltaW.
     * (1) Y belongs to Tensor[N , OC, OH, OW]
     * (2) X belongs to Tensor[N , IC, IH, IW]
     * (3) W belongs to Tenosr[OC, IC, FH, FW]
     * (4) IH = (OH - 1)*sh + FH - 2*ph
     * (5) IW = (OW - 1)*sw + FW - 2*pw
     *
     * -----------------------------------------------------------------------------
     * Forward Prop: Y = conv3D(X, W)
     * (1) step: sh(on Y axis), sw(on X axis)
     * (2) padding: ph(on Y axis), pw(on X axis)
     * (3) W[OC, IC, FH, FW]: a 4D convolution kernel, each element of W is a
     * 3D convolution kernel——W[i][IC, FH, FW]
     * (4) use Img2col algorithm to implement 3D the convolution in forward
     * propation, for detail, see {@link z.jcuda.la.CudaLA_conv3D}
     *
     * -----------------------------------------------------------------------------
     * Back Prop: deltaW  = deconv3D(X, deltaY)
     *            deltaW_e = conv3D(X_e, deltaY_pe)| step=1
     * for deltaX, see{@link z.jcuda.la.CudaLA_deconv3D_deltaX}
     *
     * (1) deltaY_pe[OC, N, OH_p, OW_p]: is the 4D convolution kernel
     * [1.1] each element of deltaY_pe is a 3D convolution kernel: deltaY_pe[i][N, OH_p, OW_p]
     * [1.2] inner padding on XoY plane: deltaY - > deltaY_p
     *      => <b>OH_p = OH + (sh-1)*(OH-1)</b>
     *      => <b>OW_p = OW + (sw-1)*(OW-1)</b>
     * [1.3] exchange N and OC axis of deltaY_p: deltaY_p -> deltaY_pe
     *      => <b> deltaY_pe[oc, n, oh, ow] = deltaY_p[n, oc, oh, ow]</b>
     * [1.4] <b> deltaY_pe[oc, n, oh*sh, ow*sw] = deltaY[n, oc, oh, ow]</b>
     *
     * (2) X_e[IC, N, IH, IW]
     * [2.1] exchange N and OC axis of X: X -> X_e
     *      => <b> X_e[ic, n, ih, iw] = X_e[n, ic, ih, iw]</b>
     *
     * (3) implements the inner-padding and the axis-exchange logically
     * [3.1] <b>get(X_e, ic, n, ih, ow) = get(X, n, ic, ih, iw)</b>
     * [3.2] <b>get(deltaY_pe, oc, n, oh*sh, ow*sw) = get(deltaY, n, oc, oh, ow)</b>
     *      => as: deltaY_pe[oc, n, oh2, ow2]: if oh2%sh!=0 || ow2%sw!=0: deltaY_pe[oc, n, oh2, ow2]=0
     *
     * (4) oph, opw: outer padding of X:
     *      => (IH + 2oph - OH_p)/1 + 1 = FH => <b>oph = ph, obviously, oph>=0</b>
     *      => (IW + 2opw - OW_p)/1 + 1 = FW => <b>opw = pw, obviously, opw>=0</b>
     *
     * (5) deltaW_e: exchange IC and OC axis of deltaW:
     *      => deltaW_e[ic][oc][fh][fw] = deltaW[oc][ic][fh][fw]
     * -----------------------------------------------------------------------------
     * use img2col method to implement the deconvolution: deltaW
     * (1) deltaY -> deltaY_p -> deltaY_pe -> Matrix A[OC, N*OH_p*OW_p] -> A[GN, GK]
     * (2) X -> X_e -> Matrix B[N*OH_p*OW_p, IC*FH*FW] -> B[GK, GM]
     * (3) W -> Matrix C[OC, IC*FH*FW] -> C[GN, GM]
     * (4) GN = OC
     * (5) GM = IC*FH*FW
     * (6) GK0 = N*OH_p*OW_p(logically)
     *     GK  = N*OH*OW
     *
     * as we "implements deltaY_pe logically", so:
     * (1) deltaY -> Matrix A[OC, N*OH*OW], to skip 0 elements in deltaY
     * (2) X -> B[N*OH_p*OW_p, IC*FH*FW]
     * when we know: deltaY[oc, n, oh, ow], we know deltaY_pe[n, oc, oh*sh, ow*sw]
     * when we know oh*sh, ow*sw, we know:
     *      => ih = fh*1 - oph + (oh*sh)
     *      => iw = fw*1 - opw + (ow*sw)
     *      we can find: X[n, ic, ih, iw]: v+=X[n, ic, ih, iw]*deltaY[n, oc, oh, ow]
     * -----------------------------------------------------------------------------
     * <b>Make sure:
     * (1) GN%4==0, GN>=4
     * (2) GM%4==0, GM>=4
     * (3) GK>=4
     * (4) FH>=2, FW>=2
     * (5) oph = (FH - ph - 1 >= 0) -> (FH > ph)
     * (6) opw = (FW - pw - 1 >= 0) -> (FW > pw)
     * </b>
     * </pre>
     */
    public static final boolean IM2COL_GEMM = true;
    //</editor-fold>
}

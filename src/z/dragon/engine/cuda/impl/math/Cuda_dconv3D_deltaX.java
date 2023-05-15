/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
 * @author Gilgamesh
 */
public final class Cuda_dconv3D_deltaX 
{
    private Cuda_dconv3D_deltaX() {}
    
    //<editor-fold defaultstate="collapsed" desc="blockNum">
    public static int blockNum(int IH, int IW, int N, int IC)
    {
        int GN = IC, GM = N * IH * IW;//get img2col size
        
        int bn = 0,bm = 0;
        for(;;) {
            if(GN > 127) bn += GN >> 7; GN &= 127; if(GN == 0) break;//2^7
            if(GN >  63) bn += 1; GN &= 63; if(GN == 0) break;//2^6
            if(GN >  31) bn += 1; GN &= 31; if(GN == 0) break;//2^5
            if(GN >  15) bn += 1; GN &= 15; if(GN == 0) break;//2^4
            if(GN >   7) bn += 1; GN &=  7; if(GN == 0) break;//2^3
            if(GN >   3) bn += 1; break;
        }
      
        for(;;) {
            if(GM > 127) bm += GM >> 7; GM &= 127; if(GM == 0) break;//2^7
            if(GM >  63) bm += 1; GM &= 63; if(GM == 0) break;//2^6
            if(GM >  31) bm += 1; GM &= 31; if(GM == 0) break;//2^5
            if(GM >  15) bm += 1; GM &= 15; if(GM == 0) break;//2^4
            if(GM >   7) bm += 1; GM &=  7; if(GM == 0) break;//2^3
            if(GM >   3) bm += 1; break;
        }
        return bn * bm;
    }
     
    public static int blockNumV2(int IH, int IW, int N, int IC)
    {
        int GN = IC, GM = N;//get img2col size
        
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
        return bn * bm * IH * IW;//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="streamSize">
    public static int streamSize_ZeroPadding_dense(int IH, int IW, int N, int IC)
    {
        int GN = IC, GM = N * IH * IW;//get img2col size
        
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
    
    public static int streamSizeV2(int N, int IC)
    {
        int GN = IC, GM = N;//get img2col size
        
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
        return (size < 10 ? size : 10);//the max stream size is 10
    }
        
    public static int streamSize_KernelSplit(int IH, int IW, int N, int IC, int sh, int sw)
    {
        int GN = IC;//get img2col size
        int GM = N * ((IH + sh - 1) / sh) * ((IW + sw - 1) / sw);
        
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
    
    public static int streamSize_CrossAdd(int OH, int OW, int N, int OC)
    {
        int GN = OC, GM = N * OH *OW;//get img2col size
        
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
        return (size < 7? size : 7);//the max stream size is 10
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Common">
    public static double Ims2_paddingScaleUp(int IH, int IW, int FH, int FW, int OH, int OW) {
        int OH0 = (IH >> 1) + ((FH + 1) >> 1) - 1;
	int OW0 = (IW >> 1) + ((FW + 1) >> 1) - 1;
	int OH1 = (IH >> 1) + (FH >> 1) - 1;
	int OW1 = (IW >> 1) + (FW >> 1) - 1;
	return 0.25 * ((OH0 + OH1) * (OW0 + OW1)) / (OH * OW);
    }
    
    public static double s1_paddingScaleUp(int IH, int IW, int FH, int FW, int OH, int OW) {
        return  (1.0 * ((IH - 1 + FH) * (IW - 1 + FW)) / (OH * OW));
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
    
    public static int[] getImg2colMatDim(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC) 
    {
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        return new int[]{GN, GM, GK};
    }
     
    public static int[] getOutPaddingDim_deltaY(int FH, int FW, int ph, int pw) {
        int oph = FH - ph - 1;
        int opw = FH - pw - 1;
        return new int[]{oph, opw};
    }
    //</editor-fold>
    
    //kernel dense: for sh = sw = 1
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaX_s1">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Dense Kernel: 4> sh*sw >0
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) sh = sw = 1
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 31, [FH, FW] = 3, [N, OC] = 4,  8, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 5, [N, OC] = 4, 16, [sh, sw] = 1, [ph, pw] = 2
     * [OH, OW] = 16, [FH, FW] = 4, [N, OC] = 4, 16, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 4, 16, [sh, sw] = 1, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 31, [FH, FW] = 4, [N, OC] = 16, 32, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.000000, Time = 1.502000 msec, Performance = 1429.749512 GFlop/s
     * (2) IC = 192: Size = 1.500000, Time = 2.284000 msec, Performance = 1410.343994 GFlop/s
     * (3) IC = 224: Size = 1.750000, Time = 2.772000 msec, Performance = 1355.734619 GFlop/s
     * (4) IC = 240: Size = 1.875000, Time = 3.138000 msec, Performance = 1283.152222 GFlop/s
     * (5) IC = 248: Size = 1.937500, Time = 3.930000 msec, Performance = 1058.714844 GFlop/s
     * (6) IC = 252: Size = 1.968750, Time = 5.276000 msec, Performance =  801.337830 GFlop/s
     * (7) IC = 255: Size = 1.992188, Time = 2.836000 msec, Performance = 1508.529663 GFlop/s
     * 
     * [OH, OW] = 32, [FH, FW] = 3, [N, OC] = 16, 64, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 1.622000 msec, Performance = 1489.469238 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 2.434000 msec, Performance = 1488.857300 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 2.966000 msec, Performance = 1425.441162 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 3.694000 msec, Performance = 1226.271851 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 4.514000 msec, Performance = 1036.961304 GFlop/s
     * (6) IC = 256: Size = 2.250000, Time = 3.092000 msec, Performance = 1562.690186 GFlop/s
     * 
     * [OH, OW] = 32, [FH, FW] = 5, [N, OC] = 8, 64, [sh, sw] = 1, [ph, pw] = 2
     * (1) IC = 128: Size = 1.562500, Time = 2.216000 msec, Performance = 1514.189087 GFlop/s
     * (2) IC = 192: Size = 2.343750, Time = 3.272000 msec, Performance = 1538.253296 GFlop/s
     * (3) IC = 224: Size = 2.734375, Time = 4.234000 msec, Performance = 1386.874146 GFlop/s
     * (4) IC = 240: Size = 2.929688, Time = 4.888000 msec, Performance = 1287.122803 GFlop/s
     * (5) IC = 248: Size = 3.027344, Time = 6.074000 msec, Performance = 1070.327881 GFlop/s
     * (6) IC = 256: Size = 3.125000, Time = 4.104000 msec, Performance = 1635.206177 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 31, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 1, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 31, [FH, FW] = 4, [N, OC] = 4, 64, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.000000, Time = 2.096000 msec, Performance = 1024.562866 GFlop/s
     * (2) IC = 192: Size = 1.500000, Time = 3.204000 msec, Performance = 1005.376221 GFlop/s
     * (3) IC = 252: Size = 1.968750, Time = 7.228000 msec, Performance =  584.927856 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_s1(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaX_s1_texture">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Dense Kernel: 4> sh*sw >0
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) sh = sw = 1
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 31, [FH, FW] = 3, [N, OC] = 2,  8, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 15, [FH, FW] = 3, [N, OC] = 4, 16, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 16, [FH, FW] = 4, [N, OC] = 4, 16, [sh, sw] = 1, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 31, [FH, FW] = 4, [N, OC] = 4, 64, [sh, sw] = 1, [ph, pw] = 0
     * (1) IC = 128: Size = 1.000000, Time = 1.522000 msec, Performance = 1410.961670 GFlop/s
     * (2) IC = 192: Size = 1.500000, Time = 2.272000 msec, Performance = 1417.792847 GFlop/s
     * (3) IC = 224: Size = 1.750000, Time = 2.744000 msec, Performance = 1369.568726 GFlop/s
     * (4) IC = 240: Size = 1.875000, Time = 3.114000 msec, Performance = 1293.041626 GFlop/s
     * (5) IC = 248: Size = 1.937500, Time = 3.902000 msec, Performance = 1066.312012 GFlop/s
     * (6) IC = 252: Size = 1.968750, Time = 5.220000 msec, Performance =  809.934570 GFlop/s
     * (7) IC = 255: Size = 1.992188, Time = 2.844000 msec, Performance = 1504.286133 GFlop/s
     * 
     * [OH, OW] = 32, [FH, FW] = 3, [N, OC] = 16, 64, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 1.640000 msec, Performance = 1473.121460 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 2.442000 msec, Performance = 1483.979858 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 2.966000 msec, Performance = 1425.441162 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 3.500000 msec, Performance = 1294.242432 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 4.440000 msec, Performance = 1054.244019 GFlop/s
     * (6) IC = 256: Size = 2.250000, Time = 3.054000 msec, Performance = 1582.134399 GFlop/s
     * (7) IC = 144: Size = 1.265625, Time = 2.126000 msec, Performance = 1278.414429 GFlop/s
     * 
     * [OH, OW] = 32, [FH, FW] = 5, [N, OC] = 8, 64, [sh, sw] = 1, [ph, pw] = 2
     * (1) IC = 128: Size = 1.562500, Time = 2.214000 msec, Performance = 1515.557007 GFlop/s
     * (2) IC = 192: Size = 2.343750, Time = 3.280000 msec, Performance = 1534.501465 GFlop/s
     * (3) IC = 224: Size = 2.734375, Time = 4.058000 msec, Performance = 1447.024536 GFlop/s
     * (4) IC = 240: Size = 2.929688, Time = 4.804000 msec, Performance = 1309.628662 GFlop/s
     * (5) IC = 248: Size = 3.027344, Time = 6.098000 msec, Performance = 1066.115356 GFlop/s
     * (6) IC = 256: Size = 3.125000, Time = 4.134000 msec, Performance = 1623.339844 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 31, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 1, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 31, [FH, FW] = 4, [N, OC] = 4, 64, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.000000, Time = 2.096000 msec, Performance = 1024.562866 GFlop/s
     * (2) IC = 192: Size = 1.500000, Time = 3.204000 msec, Performance = 1005.376221 GFlop/s
     * (3) IC = 252: Size = 1.968750, Time = 7.228000 msec, Performance =  584.927856 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_s1_texture(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaX_V2_s1">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Dense Kernel: 4> sh*sw >0
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) sh = sw = 1
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 4, [FH, FW] = 3, [N, OC] = 64, 16, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 4, [FH, FW] = 4, [N, OC] = 64, 16, [sh, sw] = 1, [ph, pw] = 1
     * [OH, OW] = 4, [FH, FW] = 5, [N, OC] = 64, 16, [sh, sw] = 1, [ph, pw] = 2
     * for IC from 1 to 128: correct
     * 
     * ======[FH = FW = 3]======================================================
     * [OH, OW] = 4, [FH, FW] = 3, [N, OC] = 256, 256, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 1.164000 msec, Performance = 2075.531738 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 1.774000 msec, Performance = 2042.772583 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 2.438000 msec, Performance = 1734.150269 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 3.148000 msec, Performance = 1438.960693 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 4.002000 msec, Performance = 1169.626099 GFlop/s
     * (6) IC = 256: Size = 2.250000, Time = 2.138000 msec, Performance = 2259.980469 GFlop/s
     * (7) IC =  64: Size = 0.562500, Time = 0.784000 msec, Performance = 1540.764771 GFlop/s
     * 
     * [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 256, 64, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 1.342000 msec, Performance = 1800.237793 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 2.040000 msec, Performance = 1776.411133 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 2.598000 msec, Performance = 1627.351196 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 3.174000 msec, Performance = 1427.173340 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 4.068000 msec, Performance = 1150.649780 GFlop/s
     * (7) IC = 256: Size = 2.250000, Time = 2.540000 msec, Performance = 1902.298584 GFlop/s
     * 
     * [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 128, 32, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 1.496000 msec, Performance = 1614.919189 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 2.404000 msec, Performance = 1507.437012 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 2.932000 msec, Performance = 1441.970825 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 3.370000 msec, Performance = 1344.168701 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 4.284000 msec, Performance = 1092.633789 GFlop/s
     * (6) IC = 256: Size = 2.250000, Time = 2.930000 msec, Performance = 1649.091553 GFlop/s
     * 
     *  ======[FH = FW = 5]======================================================
     * [OH, OW] = 4, [FH, FW] = 3, [N, OC] = 256, 128, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.562500, Time = 1.168000 msec, Performance = 2872.811035 GFlop/s
     * (2) IC = 192: Size = 2.343750, Time = 1.736000 msec, Performance = 2899.288574 GFlop/s
     * (3) IC = 224: Size = 2.734375, Time = 2.294000 msec, Performance = 2559.732178 GFlop/s
     * (4) IC = 248: Size = 3.027344, Time = 4.154000 msec, Performance = 1565.038818 GFlop/s
     * (5) IC = 256: Size = 3.125000, Time = 2.104000 msec, Performance = 3189.584717 GFlop/s
     * 
     * [OH, OW] = 8, [FH, FW] = 5, [N, OC] = 256, 32, [sh, sw] = 1, [ph, pw] = 2
     * (1) IC = 128: Size = 1.562500, Time = 1.606000 msec, Performance = 2089.317139 GFlop/s
     * (2) IC = 192: Size = 2.343750, Time = 2.392000 msec, Performance = 2104.166016 GFlop/s
     * (3) IC = 224: Size = 2.734375, Time = 3.044000 msec, Performance = 1929.049194 GFlop/s
     * (5) IC = 248: Size = 3.027344, Time = 4.876000 msec, Performance = 1333.300049 GFlop/s
     * (7) IC = 256: Size = 3.125000, Time = 3.028000 msec, Performance = 2216.276855 GFlop/s
     * 
     * [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 128, 16, [sh, sw] = 1, [ph, pw] = 1
     * (1) IC = 128: Size = 1.562500, Time = 1.906000 msec, Performance = 1760.463379 GFlop/s
     * (2) IC = 192: Size = 1.562500, Time = 1.906000 msec, Performance = 1760.463379 GFlop/s
     * (3) IC = 224: Size = 2.734375, Time = 3.528000 msec, Performance = 1664.406250 GFlop/s
     * 
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_V2_s1(boolean useTexture, long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_deltaX_W1">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH = FW =1
     * (2) Dense Kernel: 4> sh*sw >0
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 4, GK % 4 == 0
     * (6) sh = sw = 1, ph = pw = 0
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]------------
     * [OH, OW] = 31, [FH, FW] = 1, [N, OC] = 4, 32, [sh, sw] = 1, [ph, pw] = 0
     * [OH, OW] = 15, [FH, FW] = 1, [N, OC] = 4, 32, [sh, sw] = 1, [ph, pw] = 0
     * [OH, OW] =  7, [FH, FW] = 1, [N, OC] = 4, 32, [sh, sw] = 1, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 64, [FH, FW] = 1, [N, OC] = 8, 128, [sh, sw] = 1, [ph, pw] = 0
     * (1) IC = 128: Size = 0.500000, Time = 0.936000 msec, Performance = 1147.160034 GFlop/s
     * (2) IC = 192: Size = 0.750000, Time = 1.282000 msec, Performance = 1256.328247 GFlop/s
     * (3) IC = 224: Size = 0.875000, Time = 1.608000 msec, Performance = 1168.562256 GFlop/s
     * (4) IC = 240: Size = 0.937500, Time = 2.106000 msec, Performance =  955.966736 GFlop/s
     * (5) IC = 248: Size = 0.968750, Time = 2.404000 msec, Performance =  865.380493 GFlop/s
     * (6) IC = 252: Size = 0.984375, Time = 2.720000 msec, Performance =  777.179871 GFlop/s
     * (7) IC = 255: Size = 0.996094, Time = 1.620000 msec, Performance = 1320.429077 GFlop/s
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * [OH, OW] = 31, [FH, FW] = 1, [N, OC] = 2, 16, [sh, sw] = 1, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * (1) IC = 128: Size = 0.250000, Time = 0.552000 msec, Performance = 972.592285 GFlop/s
     * (2) IC = 192: Size = 0.375000, Time = 0.856000 msec, Performance = 940.778442 GFlop/s
     * (3) IC = 252: Size = 0.492188, Time = 1.586000 msec, Performance = 666.434204 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param dW_address
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC 
     */
    @Passed
    public static native void dconv3D_deltaX_W1(long[] streamArray, int length,
            long d_deltaY_address, 
            long dW_address, 
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC);
    //</editor-fold>
    
    //kernel sparse kernel: for sh * sw >= 2
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_remode">
    /**
     * <pre>
     * W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC].
     * (1) lengthv = OC * CFH * CFW * IC
     * (2) fh = y + (CFH - 1 - fhr)*sh
     * (3) fw = x + (CFW - 1 - fwr)*sw
     * </pre>
     * @param stream_address
     * @param dW_address
     * @param FH
     * @param FW
     * @param dCW_address
     * @param CFH
     * @param CFW
     * @param OC
     * @param IC
     * @param sh
     * @param sw 
     */
    @Passed
    public static native void ks_remode(long stream_address,
            long dW_address, int FH, int FW,
            long dCW_address, int CFH, int CFW,
            int OC, int IC, int sh, int sw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_remodev2">
    /**
     * <pre>
     * W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC].
     * (1) lengthv = OC * CFH * CFW * IC
     * (2) fh = y + (CFH - 1 - fhr)*sh
     * (3) fw = x + (CFW - 1 - fwr)*sw
     * </pre>
     * @param stream_address
     * @param dW_address
     * @param FH
     * @param FW
     * @param dCW_address
     * @param CFH
     * @param CFW
     * @param OC
     * @param IC
     * @param sh
     * @param sw 
     */
    @Passed
    public static native void ks_remodev2(long stream_address,
            long dW_address, int FH, int FW,
            long dCW_address, int CFH, int CFW,
            int OC, int IC, int sh, int sw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Sparse Kernel: sh*sw>=4
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]-----------
     * [IH, IW] = 15, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 3, 3, [ph, pw] = 0
     * [IH, IW] =  7, [OH, OW] = 4, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 3, 3, [ph, pw] = 0
     * [IH, IW] = 11, [OH, OW] = 6, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 3, 3, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 31, [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 32, 32, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 1.055786, Time = 0.794000 msec, Performance = 2855.520508 GFlop/s
     * (2) IC = 192: Size = 1.583679, Time = 1.042000 msec, Performance = 3263.843506 GFlop/s
     * (3) IC = 224: Size = 1.847626, Time = 1.258000 msec, Performance = 3154.011230 GFlop/s
     * (4) IC = 240: Size = 1.979599, Time = 1.676000 msec, Performance = 2536.489502 GFlop/s
     * (5) IC = 248: Size = 2.045586, Time = 1.952000 msec, Performance = 2250.441406 GFlop/s
     * (6) IC = 252: Size = 2.078579, Time = 2.358000 msec, Performance = 1893.008545 GFlop/s
     * (7) IC = 255: Size = 2.103324, Time = 1.592000 msec, Performance = 2837.219727 GFlop/s
     * (8) IC = 256: Size = 2.111572, Time = 1.282000 msec, Performance = 3537.103760 GFlop/s
     * 
     * [OH, OW] = 16, [FH, FW] = 4, [N, OC] = 16, 16, [sh, sw] = 4, [ph, pw] = 1
     * (1) IC = 128: Size = 1.876953, Time = 0.788000 msec, Performance = 5115.134766 GFlop/s
     * (2) IC = 192: Size = 2.815430, Time = 1.026000 msec, Performance = 5892.874512 GFlop/s
     * (3) IC = 224: Size = 3.284668, Time = 1.162000 msec, Performance = 6070.371094 GFlop/s
     * (4) IC = 256: Size = 3.753906, Time = 1.272000 msec, Performance = 6337.620117 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_kernelSplit(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_ImsR">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) (IH, IW) % (sh, sw) = 0
     * (7) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 32, 32, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 0.744000 msec, Performance = 3247.203125 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 0.986000 msec, Performance = 3675.333252 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 1.198000 msec, Performance = 3529.097412 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 1.426000 msec, Performance = 3176.611816 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 1.678000 msec, Performance = 2789.537109 GFlop/s
     * (6) IC = 252: Size = 2.214844, Time = 2.028000 msec, Performance = 2345.335449 GFlop/s
     * (7) IC = 255: Size = 2.241211, Time = 1.584000 msec, Performance = 3038.487305 GFlop/s
     * (8) IC = 256: Size = 2.250000, Time = 1.244000 msec, Performance = 3884.114502 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_ksImsR(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_ImsR_texture">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) (IH, IW) % (sh, sw) = 0
     * (7) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 16, [FH, FW] = 4, [N, OC] = 16, 16, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 1.125000, Time = 0.748000 msec, Performance = 3229.838379 GFlop/s
     * (2) IC = 192: Size = 1.687500, Time = 1.000000 msec, Performance = 3623.878662 GFlop/s
     * (3) IC = 224: Size = 1.968750, Time = 1.196000 msec, Performance = 3534.998779 GFlop/s
     * (4) IC = 240: Size = 2.109375, Time = 1.448000 msec, Performance = 3128.348389 GFlop/s
     * (5) IC = 248: Size = 2.179688, Time = 1.720000 msec, Performance = 2721.420410 GFlop/s
     * (6) IC = 252: Size = 2.214844, Time = 2.038000 msec, Performance = 2333.827637 GFlop/s
     * (7) IC = 255: Size = 2.241211, Time = 1.592000 msec, Performance = 3023.218506 GFlop/s
     * (8) IC = 256: Size = 2.250000, Time = 1.192000 msec, Performance = 4053.555420 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_ksImsR_texture(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_Ims2R">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Sparse Kernel: sh == sw == 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) (IH, IW) % (sh, sw) = 0
     * (7) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 16, 128, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 2.250000, Time = 0.986000 msec, Performance = 4900.444336 GFlop/s
     * (2) IC = 192: Size = 3.375000, Time = 1.348000 msec, Performance = 5376.674316 GFlop/s
     * (3) IC = 224: Size = 3.937500, Time = 1.714000 msec, Performance = 4933.323730 GFlop/s
     * (4) IC = 240: Size = 4.218750, Time = 2.102000 msec, Performance = 4310.036621 GFlop/s
     * (5) IC = 248: Size = 4.359375, Time = 2.486000 msec, Performance = 3765.762695 GFlop/s
     * (6) IC = 252: Size = 4.429688, Time = 3.136000 msec, Performance = 3033.380615 GFlop/s
     * (7) IC = 255: Size = 4.482422, Time = 1.916000 msec, Performance = 5023.970703 GFlop/s
     * (8) IC = 256: Size = 4.500000, Time = 1.688000 msec, Performance = 5724.926758 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_ksIms2R(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplit_Ims2R_texture">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Sparse Kernel: sh == sw == 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) (IH, IW) % (sh, sw) = 0
     * (7) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 32, [OH, OW] = 16, [FH, FW] = 3, [N, OC] = 16, 64, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 2.250000, Time = 0.990000 msec, Performance = 4880.644531 GFlop/s
     * (2) IC = 192: Size = 3.375000, Time = 1.346000 msec, Performance = 5384.664063 GFlop/s
     * (3) IC = 224: Size = 3.937500, Time = 1.722000 msec, Performance = 4910.404785 GFlop/s
     * (4) IC = 240: Size = 4.218750, Time = 2.120000 msec, Performance = 4273.441895 GFlop/s
     * (5) IC = 248: Size = 4.359375, Time = 2.486000 msec, Performance = 3765.762695 GFlop/s
     * (6) IC = 252: Size = 4.429688, Time = 3.136000 msec, Performance = 3033.380615 GFlop/s
     * (7) IC = 255: Size = 4.482422, Time = 1.916000 msec, Performance = 5023.970703 GFlop/s
     * (8) IC = 256: Size = 4.500000, Time = 1.692000 msec, Performance = 5711.392578 GFlop/s
     * 
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_ksIms2R_texture(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="dconv3D_kernelSplitV2_Ims2R">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Sparse Kernel: sh == sw == 2
     * (3) GN >= 4, GN % 4 == 0
     * (4) GM >= 4, GM % 4 == 0
     * (5) GK >= 8, GK % 4 == 0
     * (6) (IH, IW) % (sh, sw) = 0
     * (7) W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]------------
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 12, [OH, OW] = 6, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 3, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * [IH, IW] = 16, [OH, OW] = 8, [FH, FW] = 4, [N, OC] = 4, 32, [sh, sw] = 2, 2, [ph, pw] = 1
     * for IC from 1 to 128: correct
     * 
     * [IH, IW] = 4, [OH, OW] = 2, [FH, FW] = 3, [N, OC] = 256, 512, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 2.250000, Time = 0.716000 msec, Performance = 6748.376953 GFlop/s
     * (2) IC = 192: Size = 3.375000, Time = 1.078000 msec, Performance = 6723.337402 GFlop/s
     * (3) IC = 224: Size = 3.937500, Time = 1.400000 msec, Performance = 6039.797852 GFlop/s
     * (4) IC = 240: Size = 4.218750, Time = 1.850000 msec, Performance = 4897.133301 GFlop/s
     * (5) IC = 248: Size = 4.359375, Time = 2.284000 msec, Performance = 4098.812012 GFlop/s
     * (6) IC = 252: Size = 4.429688, Time = 2.952000 msec, Performance = 3222.453125 GFlop/s
     * (7) IC = 255: Size = 4.482422, Time = 1.358000 msec, Performance = 7088.312012 GFlop/s
     * (8) IC = 256: Size = 4.500000, Time = 1.254000 msec, Performance = 7706.281250 GFlop/s
     * 
     * [IH, IW] = 8, [OH, OW] = 4, [FH, FW] = 3, [N, OC] = 32, 64, [sh, sw] = 2, [ph, pw] = 1
     * (1) IC = 128: Size = 2.250000, Time = 0.814000 msec, Performance = 5935.918945 GFlop/s
     * (2) IC = 192: Size = 3.375000, Time = 1.198000 msec, Performance = 6049.881348 GFlop/s
     * (3) IC = 224: Size = 3.937500, Time = 1.588000 msec, Performance = 5324.758301 GFlop/s
     * (4) IC = 240: Size = 4.218750, Time = 1.922000 msec, Performance = 4713.681641 GFlop/s
     * (5) IC = 248: Size = 4.359375, Time = 2.378000 msec, Performance = 3936.790039 GFlop/s
     * (6) IC = 256: Size = 4.500000, Time = 1.406000 msec, Performance = 6873.169434 GFlop/s
     * </pre>
     * @param useTexture
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dCW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_ksV2_Ims2R(boolean useTexture, long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dCW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="dconv3D_crossAdd">
    /**
     * <pre>
     * deltaX[N, IH, IW, IC] = dconv3D(deltaY[N, OH, OW, OC], W[OC, FH, FW, IC], [sh, sw], [ph, pw]).
     * (1) FH * FW >= 2
     * (2) Sparse Kernel: sh*sw>=4
     * (3) GN >= 4, GN % 4 == 0, GN = 0C
     * (4) GM >= 4, GM % 4 == 0, GM = N * OH * OW
     * (5) GK >= 8, GK % 4 == 0, GK = FH * FW * IC
     * only used for: 16 >= IC(sh*sw >= 4), 8 >= IC
     * 
     * -----Performace on CudaFloat32 Engine(GTX 1050)[asynchronized]-----------
     * [OH, OW] = 15, [FH, FW] = 3, [N, OC] = 2, 8, [sh, sw] = 3, 3, [ph, pw] = 0
     * for IC from 1 to 128: correct
     * 
     * [OH, OW] = 16, [FH, FW] = 4, [N, IC] = 16, 16, [sh, sw] = 2, [ph, pw] = 1
     * (1) OC = 128: Size = 0.500000, Time = 1.202000 msec, Performance =  893.296021 GFlop/s
     * (2) OC = 192: Size = 0.750000, Time = 1.914000 msec, Performance =  841.490417 GFlop/s
     * (3) OC = 224: Size = 0.875000, Time = 2.470000 msec, Performance =  760.748230 GFlop/s
     * (4) OC = 240: Size = 0.937500, Time = 3.036000 msec, Performance =  663.131042 GFlop/s
     * (5) OC = 248: Size = 0.968750, Time = 3.600000 msec, Performance =  577.881897 GFlop/s
     * (6) OC = 252: Size = 0.984375, Time = 4.346000 msec, Performance =  486.407990 GFlop/s
     * (7) OC = 255: Size = 0.996094, Time = 1.720000 msec, Performance = 1243.659912 GFlop/s
     * </pre>
     * @param streamArray
     * @param length
     * @param d_deltaY_address
     * @param OH
     * @param OW
     * @param dW_address
     * @param FH
     * @param FW
     * @param d_deltaX_address
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    @Passed
    public static native void dconv3D_deltaX_crossAdd(long[] streamArray, int length,
            long d_deltaY_address, int OH, int OW,
            long dW_address, int FH, int FW,
            long d_deltaX_address, int IH, int IW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Document">
    /**
     * <pre>
     * Deconvolution 3D for deltaX.
     * (1) Y belongs to Tensor[N , OC, OH, OW]
     * (2) X belongs to Tensor[N , IC, IH, IW]
     * (3) W belongs to Tensor[OC, IC, FH, FW]
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
     * Back Prop: deltaX = deconv3D(deltaY, W)
     *                   = conv3D(deltaY_p, W_r)| step=1
     * for deltaW, see{@link  z.jcuda.la.CudaLA_deconv3D_deltaW}
     *
     * (1) Wr[IC, OC, FH, FW] is the 4D convolution kernel
     * [1.1] each element of Wr is 3D convolution kernel: W_r[i][OC, FH, FW]
     * [1.2] W_r[ic, oc, (FH - 1 - fh), (FW - 1 - fw)] = W[oc, ic, fh, fw]
     *  => In plane Xoy, rotate W 180 degress to W_r
     *  => exchange coordinates of IC and OC axiss of W to get W_r
     * [1.3] We get W_r logically, using a specific way to fetch memory, As:
     * => W_r[ic, oc, fh, fw]
     *    = W[oc, ic, (FH - 1 - fh), (FW - 1 - fw)]
     *    = get4d(W, oc, ic, (FH - 1- fh), (Fw - 1 - fw), IC, FH, FW)
     * (2) deltaY[N, OC, OH, OW]
     * (3) deltaY_p[N, OC, OH_p, OW_p]
     *       = innerPadding(deltaY, sh-1, sw-1), as sh>=1, sw>=1
     * [1.1] implements the innerPadding of deltaY logically.
     * [1.2] inner padding on XoY plane: deltaY
     *      => <b>OH_p = OH + (sh-1)*(OH-1)</b>
     *      => <b>OW_p = OW + (sw-1)*(OW-1)</b>
     * [1.3] get4(deltaY_p, n, oc, oh, ow):
     *  <b>if(0>oh||oh>=OH_p||0>ow||ow>=OW_p) return 0;
     *     if( oh%sh!=0 && ow%sw!=0 ) return 0;
     *     return get(deltaY, n, oc, oh/sh, ow/sw);</b>
     *
     * (4) inner padding of delataY_p: iph = sh-1, ipw = sw-1, like a0,....(s-1) 0....,a1
     * (5) outter padding of deltaY_p: oph, ohw, the same padding of conv3D in Forward Prop
     * (6) step of the deconv3D: dsh = dsw = 1
     * (7) (OH_p + 2*oph - FH)/1 + 1 = IH
     *     (OW_p + 2*opw - FW)/1 + 1 = IW
     *    => <b>oph = FH - ph - 1 >= 0, obviously, FH >= ph-1</b>
     *    => <b>opw = FW - pw - 1 >= 0, obvioysly, FW >= pw-1</b>
     *
     * -----------------------------------------------------------------------------
     * use img2col method to implemnt the deconvolution:  deltaX = conv3D(deltaY_p, W_r)
     * (1) W -> W_r -> Matrix A[IC, OC*FH*FW] -> A[GN, GK]
     * (2) deltaY -> deltaY_p -> Matrix B[OC*FH*FW, N*IH*IW] -> B[GK, GM]
     * (3) deltaX -> Matrix C[IC, N*IH*IW] -> C[GN, GM]
     * (4) C = A*B
     * (5) GN = IC
     * (6) GM = N*IH*IW
     * (7) GK = OC*FH*FW
     * -----------------------------------------------------------------------------
     * <b>Make Sure:
     *  (1) GN%4==0, GN>=4
     *  (2) GM%4==0, GM>=4
     *  (4) FH>=1, FW>=1, don't use 1*x convolution kernel, it may work slow
     *  (5) oph = (FH - ph - 1 >= 0) -> (FH > ph)
     *  (6) opw = (FW - pw - 1 >= 0) -> (FW > pw)</b>
     * </pre>
     */
    public static final boolean IM2COL_GEMM = true;
    //</editor-fold>
}

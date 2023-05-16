package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Tense;
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
public class Cuda_upool2D_avg_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        
        int GN = IC;
        int GM = N * IH * IW;
        int GK = FH * FW;
        
        System.out.println("Test Correct:");
        System.out.format("IH, IW = %d, %d\n", IH, IW);
        System.out.format("FH, FW = %d, %d\n", FH, FW);
        System.out.format("OH, OW = %d, %d\n", OH, OW);
        System.out.format("N, IC = %d, %d\n", N, IC);
        System.out.format("sh, sw, ph, pw = %d, %d, %d, %d\n", sh, sw, ph, pw);
        System.out.format("GN, GM, GK = %d, %d, %d\n", GN, GM, GK);
        
        int sizeX = N * IH * IW * IC;
        int sizeY = N * OH * OW * IC;
        
        float[] deltaY = Vector.randomFloatVector(sizeY, 1, 10);
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, IC);
                
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaY = Tense.vectorToTensor_4D(deltaY, N, OH, OW, IC);
        float[][][][] cdeltaX = new float[N][IH][IW][IC];
        
//        Tense.unpool2D_avgerage_img2col_plus2(
//                cdeltaY, OH, OW, 
//                FH, FW,
//                cdeltaX, IH, IW,
//                N, IC, 
//                sh, sw, ph, pw);
        
        Tense.unpool2D_avgerage_img2col_plus2_ignore_padding(
                cdeltaY, OH, OW,
                FH, FW,
                cdeltaX, IH, IW,
                N, IC, 
                sh, sw, ph, pw);
       
        float[] deltaX1 = Tense.tensor_4DToVector(cdeltaX, sizeX);

        //GPU------------------------------------------------------------------
//        Tensor tdeltaX1 = eg.upool2D_avg(tdeltaY, FH, FW, sh, sw, ph, pw).c();
//        Tensor tdeltaX2 = eg.upool2D_avg(eg.empty(N, IH, IW, IC), tdeltaY, FH, FW, sh, sw).c();

        Tensor tdeltaX1 = eg.upool2D_avg_ignore_padding(tdeltaY, FH, FW, sh, sw, ph, pw).c();
        Tensor tdeltaX2 = eg.upool2D_avg_ignore_padding(eg.empty(N, IH, IW, IC).c(), tdeltaY, FH, FW, sh, sw).c();
        
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        
        //compare---------------------------------------------------------------
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);
        System.out.print("GPU1: "); Vector.println(deltaX2, 0, 10);
        System.out.print("GPU2: "); Vector.println(deltaX3, 0, 10);
        
        float sp1 = Vector.samePercentAbsolute(deltaX1, deltaX2); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercentAbsolute(deltaX1, deltaX3); System.out.println("sp3: "+sp2);
        float zp0 = Vector.zeroPercent(deltaX1); System.out.println("zp0: " + zp0);
        float zp1 = Vector.zeroPercent(deltaX2); System.out.println("zp1: " + zp1);
        
        eg.delete(tdeltaY, tdeltaX1, tdeltaX2);
        
        if(sp1!=1) throw new RuntimeException();
    }
    
     public static void testSpeed(int OH, int OW, int FH, int FW, int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false);
        eg.sync(false);
        int IH = (OH - 1)*sh + FH - 2*ph;
        int IW = (OW - 1)*sw + FW - 2*pw;
        
        int GN = IC;
        int GM = N * IH * IW;
        int GK = FH * FW;
        
        System.out.format("IH, IW = %d, %d\n", IH, IW);
        System.out.format("FH, FW = %d, %d\n", FH, FW);
        System.out.format("OH, OW = %d, %d\n", OH, OW);
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N * IH * IW * IC;
        int sizeY = N * OH * OW * IC;
        
        float[] deltaX = Vector.randomFloatVector(sizeX);
        float[] X = Vector.randomFloatVector(sizeX);
        float[] Y = Vector.randomFloatVector(sizeY);
        
        Tensor tdeltaY = eg.empty(N, OH, OW, IC);
        Tensor tdeltaX = eg.empty(N, IH, IW, IC).c();
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0;i<nIter;i++)
        {
//            tdeltaX = eg.upool2D_avg(tdeltaY, FH, FW, sh, sw, ph, pw).sync(); eg.delete(tdeltaX);
            tdeltaX = eg.upool2D_avg(tdeltaX, tdeltaY, FH, FW, sh, sw);
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        long div=timer.timeStampDifMills();
        float time = 1.0f*div / nIter;
	float size = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
	float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
	float performance = total / (time*1e-3f);
        System.out.format("Size = %f, Time = %f msec, Performance = %f  GFlop/s\n", 
                size, time, performance);
        
        eg.delete(tdeltaY, tdeltaX);
    }
    public static void main(String[] args)
    {
//        int OH = 32, OW = 32;
//	int N = 2, IC = 255;
//	int FH = 8, FW = 8;
//	int sh = 4, sw = 4, ph = 2, pw = 2;
        
        
        int OH = 32, OW = 32;
	int N = 4, IC = 255;
	int FH = 4, FW = 4;
	int sh = 4, sw = 4, ph = 1, pw = 1;
        for(int ic = 1; ic<=128; ic++) 
            testCorrect(OH, OW, FH, FW, N, ic, sh, sw, ph, pw);
        
//        testCorrect(OH, OW, FH, FW, N, IC, sh, sw, ph, pw);
//        testSpeed(OH, OW, FH, FW, N*2, IC, sh, sw, ph, pw);
    }
}

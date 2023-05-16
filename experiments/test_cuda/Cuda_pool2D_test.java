package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaException;
import z.dragon.engine.cuda.impl.Cuda_expk2;
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
public class Cuda_pool2D_test
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
     
    public static void testCorrect(
            int IH, int IW,
            int FH, int FW,
            int N, int IC, 
            int sh, int sw, int ph ,int pw)
    {
        int OH = (IH+ph*2 - FH)/sh+1;
        int OW = (IW+pw*2 - FW)/sw+1;
        
        int GN = IC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        
        System.out.println("TestCorrect:");
        System.out.format("\t(IH, IW) = (%d, %d)\n", IH, IW);
        System.out.format("\t(FH, FW) = (%d, %d)\n", FH, FW);
        System.out.format("\t(OH, OW) = (%d, %d)\n", OH, OW);
        System.out.format("\t(N, IC) = (%d, %d)\n", N, IC);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N * IH * IW * IC;
        int sizeY = N * OH * OW * IC;
        
        float[] X = Vector.randomFloatVector(sizeX, -19, 19);
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cY = new float[N][OH][OW][IC];
        
//        ----------------------------------------------------------------------
//        Tense.pool2D_max_img2col(
//                cX, IH, IW,
//                FH, FW,
//                cY, OH, OW,
//                N, IC, 
//                sh, sw, ph, pw);//A
//
        Tense.pool2D_avg_naive_ignore_padding(cX, IH, IW, FH, FW, cY, OH, OW, N, IC, sh, sw, ph, pw);
//        Tense.pool2D_avg_naive(cX, IH, IW, FH, FW, cY, OH, OW, N, IC, sh, sw, ph, pw);
        
        //----------------------------------------------------------------------
        float[] Y1 = Tense.tensor_4DToVector(cY, sizeY); 
        
        
        //GPU-------------------------------------------------------------------
//        Tensor tY1 = eg.pool2D_max(tX, FH, FW, sh, sw, ph, pw).c();
//        Tensor tY2 = eg.pool2D_max(eg.tensor(N, OH, OW, IC).c(), tX, FH, FW, sh, sw).c();

//        Tensor[] R = eg.pool2D_max_indexed(tX, FH, FW, sh, sw, ph, pw);
//        Tensor tY1 = R[0].c(), Index = R[1];
//        Cuda_expk2.checkMemAlign(Index);
//        Tensor tY2 = eg.pool2D_max_indexed(eg.empty(N, OH, OW, IC).c(), Index, tX, FH, FW, sh, sw).c();
        
//        Tensor tY1 = eg.pool2D_avg(tX, FH, FW, sh, sw, ph, pw).c();
//        Tensor tY2 = eg.pool2D_avg(eg.empty(N, OH, OW, IC).c(), tX, FH, FW, sh, sw).c();
        
        Tensor tY1 = eg.pool2D_avg_ignore_padding(tX, FH, FW, sh, sw, ph, pw).c();
        Tensor tY2 = eg.pool2D_avg_ignore_padding(eg.empty(N, OH, OW, IC).c(), tX, FH, FW, sh, sw).c();
        
        float[] Y2 = eg.valueOf(tY1);
        float[] Y3 = eg.valueOf(tY2);

        //----------------------------------------------------------------------
        float zp0 = Vector.zeroPercent(Y1); System.out.println("zp0: " + zp0);
        float zp1 = Vector.zeroPercent(Y2); System.out.println("zp1 = " + zp1);
        float zp2 = Vector.zeroPercent(Y3); System.out.println("zp2 = " + zp2);
        
        System.out.print("CPU: "); Vector.println(Y1, 0, 10);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentAbsolute(Y1, Y2); System.out.println("sp: "+sp1);
        float sp2 = Vector.samePercentAbsolute(Y1, Y3); System.out.println("sp: "+sp2);
        
        if(sp1 != 1) System.out.println("12312313123");
        if(sp1 != 1) throw new RuntimeException("IC = " + IC);
        
        eg.delete(tX, tY1, tY2);
    }
    
    public static void testSpeed(
            int IH, int IW,
            int FH, int FW,
            int N, int IC, 
            int sh, int sw, int ph ,int pw)
    {
        eg.sync(false);
        eg.check(false);
        int OH = (IH + ph*2  - FH)/sh+1;
        int OW = (IW + pw*2 - FW)/sw+1;
        
        int GN = IC;
        int GM = N * OH * OW;
        int GK = FH * FW;
        
        System.out.println("TestCorrect:");
        System.out.format("\t(IH, IW) = (%d, %d)\n", IH, IW);
        System.out.format("\t(FH, FW) = (%d, %d)\n", FH, FW);
        System.out.format("\t(OH, OW) = (%d, %d)\n", OH, OW);
        System.out.format("\t(N, IC) = (%d, %d)\n", N, IC);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N * IH * IW * IC;
        int sizeY = N * OH * OW * IC;
        
        float[] X=Vector.randomFloatVector(sizeX, -19, 19);
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tY = eg.empty(N, OH, OW, IC);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0; i< nIter; i++)
        {
//            tY = eg.pool2D_max(tX, FH, FW, sh, sw, ph, pw).c();eg.delete(tY);
//            tY = eg.pool2D_max(tY, tX, FH, FW, sh, sw);
//            tY = eg.pool2D_avg(tX, FH, FW, sh, sw, ph, pw).c(); eg.delete(tY);
//            tY = eg.pool2D_avg(tY, tX, FH, FW, sh, sw);
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        long div=timer.timeStampDifMills();
	float time = 1.0f*div / nIter;
	float size = 2.0f* GN / 1024 * GM / 1024 * GK / 1024;
	float performance = (1024 * 1024 * size*1.0e-9f * 1024) / (time*1e-3f);

	System.out.printf("Size = %f, Time = %f msec, Performance = %f GFlop/s\n",
		size, time, performance);
        
        eg.delete(tY, tX);
    }
    
    public static void main(String[] args)
    {
//        int IH = 62, IW = 62;
//	int N = 4, IC = 192;
//	int FH = 4, FW = 4;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
        int IH = 16, IW = 16;
	int N = 4, IC = 128;
	int FH = 4, FW = 4;
	int sh = 2, sw = 2, ph = 1, pw = 1;
        try
        {
            for(int ic = 1; ic <= 128; ic++)
                testCorrect(IH, IW, FH, FW, N, ic, sh, sw, ph, pw);
            
//            testCorrect(31, 31, 3, 3, 4, 72, 1, 2, 1, 1);
//            testCorrect(IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
//            testSpeed(IH, IW, FH, FW, N*2, IC, sh, sw, ph, pw);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        CudaException ex = CudaException.lastException();
        System.out.println(ex);
    }
}

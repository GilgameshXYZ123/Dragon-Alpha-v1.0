package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
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
public class Cuda_dconv3D_dW_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
    }
    
    public static void testCorrect(
            int IH, int IW,
            int OH, int OW, 
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        
        int OHp = OH + (OH-1) * (sh-1), OWp = OW + (OW-1) * (sw-1);
        int oph = ph, opw = pw;
        
        System.out.println("\nTest correct:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = OC;
        int GM = IC * FH * FW;
        int GK0 = N * OHp * OWp;
        System.out.format("(GN, GM, GK0) = (%d, %d, %d)\n", GN, GM, GK0);
        
        int sizeX = N * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;//sizeW_e = IC*OC(
        int sizeY = N * OH * OW * OC;
        
        float[] deltaY = Vector.randomFloatVector(sizeY);
        float[] X = Vector.randomFloatVector(sizeX);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cY = Tense.vectorToTensor_4D(deltaY, N, OH, OW, OC);
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cdeltaW = new float[OC][FH][FW][IC];
        
        Tense.deconv3d_deltaW_naive(cX, IH, IW, 
                cY, OH, OW,
                cdeltaW, FH, FW,
                N, IC, OC,
                sh, sw, ph, pw);
//        Tense.deconv3D_deltaW_img2col2(X1, IH, IW, 
//                deltaY1, OH, OW,
//                t_deltaW1, FH, FW,
//                N, IC, OC,
//                sh, sw, ph, pw);
                
        float[] deltaW1 = Tense.tensor_4DToVector(cdeltaW, sizeW); 
        System.out.print("CPU1: ");Vector.println(deltaW1, 0, 10);
        float zp0 = Vector.zeroPercent(deltaW1); 
        
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tdeltaW1 = eg.dconv3D_deltaW(tX, tdeltaY, FH, FW, sh, sw, ph, pw).c();
        Tensor tdeltaW2 = eg.dconv3D_deltaW(eg.empty(OC, FH, FW, IC).c(), tX, tdeltaY, sh, sw).c();
        
        float[] deltaW2 = eg.valueOf(tdeltaW1);
        System.out.print("GPU1: "); Vector.println(deltaW2, 0, 10);
        float[] deltaW3 = eg.valueOf(tdeltaW2);
        System.out.print("GPU2: "); Vector.println(deltaW3, 0, 10);
        
        float zp3 = Vector.zeroPercent(deltaW2);
        
        //compare---------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(deltaW1, deltaW2, 1e-5f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercentRelative(deltaW1, deltaW3, 1e-5f); System.out.println("sp2: "+sp2);
        
        System.out.println("zp0: " + zp0);
        System.out.println("zp3: "+zp3);
        
        if(sp1 != 1) {throw new RuntimeException();}
        if(sp2 != 1) {throw new RuntimeException();}
        
        eg.delete(tX, tdeltaY, tdeltaW1, tdeltaW2);
    }
    
    public static void testSpeed(
            int IH, int IW,
            int OH, int OW, 
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false).sync(false);
        
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;

        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = ph, opw = pw;
        
        System.out.println("Test speed:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = OC;
        int GM = IC * FH * FW;
        int GK = N * OH * OW;
        int GK0 = N * OHp * OWp;
        System.out.format("(GN, GM, GK) = (%d, %d, %d, %d)\n", GN, GM, GK, GK0);
        
        int sizeX = N*IC*IH*IW;
        int sizeY = N*OC*OH*OW;
        
        float[] deltaY = Vector.randomFloatVector(sizeY);
        float[] X = Vector.randomFloatVector(sizeX);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tdeltaW = eg.empty(OC, FH, FW, IC).c();
        
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 500;
        for(int i = 0;i < nIter;i++)
        {
            tdeltaW = eg.dconv3D_deltaW(tX, tdeltaY, FH, FW, sh, sw, ph, pw); eg.delete(tdeltaW.c());
//            tdeltaW = eg.dconv3D_deltaW(tdeltaW, tX, tdeltaY, sh, sw);
        }
        
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStampDifMills();
        float time = 1.0f*div / nIter;
        float size  = 1.0f * GN / 1024 * GM / 1024 * GK / 1024;
        float total = 2 * 1024 * size * 1e-9f * 1024 * 1024;
        float performance = total / (time*1e-3f);
        
        float size0 = 1.0f * GN / 1024 * GM / 1024 * GK0 / 1024;
        float total0 = 2 * 1024 * size0 * 1e-9f * 1024 * 1024;
	float performance0 = total0 / (time*1e-3f);
        
        System.out.format("Time = %f, Size = %f, Performance = %f GFlop/s, Size0 = %f, Performance0 = %f GFlop/s\n", 
                time, size, performance, size0, performance0);
        
        eg.delete(tX, tdeltaW, tdeltaY);
    }
    
    public static void main(String[] args)
    {
//        Vector.PRINT_DIFFERENT = true;

        //test W11--------------------------------------------------------------
//        int OH = 15, OW = 15;
//	int FH = 1, FW = 1;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 128, OC = 252;//9*4=36 
//        for(int oc = 1; oc <= 128; oc++)
//            testCorrect(OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw);

//        int OH = 63, OW = 63;
//	int FH = 1, FW = 1;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 128, OC = 128;//9*4=36 

//        testCorrect(OH, OW, FH, FW, N/2, IC, OC, sh, sw, ph, pw);
//        testSpeed(OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        
        //test GemmSK W11-------------------------------------------------------
//        int IH = -1, IW = -1;
//        int OH = 15, OW = 15;
//	int FH = 1, FW = 1;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 32;
//	int IC = 128, OC = 128;//9*4=36 
        
//        for(int oc = 1; oc <= 128; oc++)
//            testCorrect(IH, IW, OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw);
         
//        for(int ic = 1; ic <= 128; ic++)
//            testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);

//        int IH = -1, IW = -1;
//        int OH = 16, OW = 16;
//	int FH = 1, FW = 1;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 32, IC = 128;
//	int OC = 256;//9*4=36 
//
//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);

        //test gemm-------------------------------------------------------------
//        int IH = -1, IW = -1;
//        int OH = 15, OW = 15;
//	int FH = 4, FW = 4;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 4;
//	int IC = 32, OC = 252;//9*4=36 
//        for(int oc = 1; oc <= 128; oc++)
//            testCorrect(IH, IW, OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw);
        
//        int IH = -1, IW = -1;
//        int OH = 32, OW = 32;
//	int FH = 4, FW = 4;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 2, IC = 128;
//	int OC = 160;//9*4=36 
//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test GemmSK-----------------------------------------------------------
//        int IH = -1, IW = -1;
//        int OH = 16, OW = 16;
//        int OH = 7, OW = 7;
//        int OH = 5, OW = 5;
//	int FH = 4, FW = 4;
//        int FH = 3, FW/ = 3;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 32, IC = 32, OC = 32;//9*4=36  
        
//        for(int oc = 1; oc <=128; oc++) 
//            testCorrect(IH, IW, OH, OW, FH, FW, N, IC, oc, sh, sw, ph, pw);
        
//         for(int ic = 1; ic <=128; ic++) 
//            testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        
        //61, 62
        
//        int IH = -1, IW = -1;
//        int OH = 16, OW = 16;
//	int FH = 4, FW = 4;
//	int sh = 1, sw = 1, ph = 1, pw = 1;//N * OH * OW
//	int N = 64, IC = 32;
//	int OC = 128;//9*4=36 

//        int IH = 32, IW = 32;
//        int OH = 16, OW = 16;
//	int FH = 4, FW = 4;
//	int sh = 2, sw = 2, ph = 1, pw = 1;//N * OH * OW
//	int N = 32, IC = 32;
//	int OC = 256;//9*4=36 
        
//        int IH = 4, IW = 4, OH = 2, OW = 2, N = 512;
//	int IH = 8, IW = 8, OH = 4, OW = 4, N = 128;
//        int IH = 14, IW = 14, OH = 7, OW = 7, N = 32;
	int IH = 16, IW = 16, OH = 8, OW = 8, N = 32;
//	int IH = 32, IW = 32, OH = 16, OW = 16, N = 8;
	int FH = 3, FW = 3, ph = 1, pw = 1;//N * OH * OW
	int sh = 2, sw = 2;
	int IC = 128, OC = 128;//9*4=36 128 -> 20
//        
//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, OH, OW, FH, FW, N, IC*2, OC, sh, sw, ph, pw);

//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC - 256, sh, sw, ph, pw);
        testSpeed(IH, IW, OH, OW, FH, FW, N, IC*2, OC, sh, sw, ph, pw);
//        
//        System.out.println(CudaException.lastException());
        
        //for input feature-----------------------------------------------------
//        int IH = 32, IW = 32;
//        int OH = 32, OW = 32;
//        int FH = 3, FW = 3;
//        int N = 128, IC = 3, OC = 64;
//        int sh = 1, sw = 1, ph = 1, pw = 1;
//        
//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
    }
 
}

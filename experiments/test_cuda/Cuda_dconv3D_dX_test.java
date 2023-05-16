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
public class Cuda_dconv3D_dX_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.dconv3D_deltaX_s1_useTexture(false);
        cu32.dconv3D_deltaX_ks_useTexture(true);
    }
    
    public static void testCorrect(int IH, int IW,
            int OH, int OW,
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;

        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH - ph - 1, opw = FW - pw - 1;
        
        System.out.println("Test correct:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N  * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;
	int sizeY = N  * OH * OW * OC;
        
        float[] W = Vector.randomFloatVector(sizeW);
        float[] deltaY = Vector.randomFloatVector(sizeY);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cdeltaY = Tense.vectorToTensor_4D(deltaY, N, OH, OW, OC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cdeltaX = new float[N][IH][IW][IC];
        
        Tense.deconv3D_deltaX_img2col(
                cdeltaY, OH, OW, 
                cW, FH, FW,
                cdeltaX, IH, IW,
                N, IC, OC,
                sh, sw, ph, pw);
        
        float[] deltaX1 = Tense.tensor_4DToVector(cdeltaX, sizeX); 
        System.out.print("CPU1: "); Vector.println(deltaX1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tdeltaX1 = eg.dconv3D_deltaX(tdeltaY, tW, IH, IW, sh, sw, ph, pw).c();
        Tensor tdeltaX2 = eg.dconv3D_deltaX(eg.empty(N, IH, IW, IC), tdeltaY, tW, sh, sw).c();
        
        //----------------------------------------------------------------------
        float[] deltaX2 = eg.valueOf(tdeltaX1);
        System.out.print("GPU : "); Vector.println(deltaX2, 0, 10);
        
        float[] deltaX3 = eg.valueOf(tdeltaX2);
        System.out.print("GPU : "); Vector.println(deltaX3, 0, 10);
        
        float sp1 = Vector.samePercentRelative(deltaX1, deltaX2, 1e-4f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercentRelative(deltaX1, deltaX3, 1e-4f); System.out.println("sp2: "+sp2);
        float zp1 = Vector.zeroPercent(deltaX2); System.out.println("zp1: " + zp1);
        float zp2 = Vector.zeroPercent(deltaX3); System.out.println("zp2: " + zp2);
        
        eg.delete(tdeltaY, tW, tdeltaX1);
        if(sp1 < 0.999f) throw new RuntimeException();
        if(sp2 < 0.999f) throw new RuntimeException();
    }
    
    public static void testSpeed(int IH, int IW,
            int OH, int OW,
            int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false).sync(false);
        
        if(IH == -1) IH = (OH - 1)*sh + FH - 2*ph;
        if(IW == -1) IW = (OW - 1)*sw + FW - 2*pw;
        
        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH - ph - 1, opw = FW - pw - 1;
        
        System.out.println("Test Speed:");
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        System.out.format("\t(OH_p, OW_p) = (%d, %d)\n", OHp, OWp);
        System.out.format("\t(oph, opw) = (%d, %d)\n", oph, opw);
        
        int GN = IC;
        int GM = N  * IH * IW;
        int GK = OC * FH * FW;
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N  * IH * IW * IC;
        int sizeW = OC * FH * FW * IC;
	int sizeY = N  * OH * OW * OC;
        
        float[] W = Vector.randomFloatVector(sizeW);
        float[] deltaY = Vector.randomFloatVector(sizeY);
        
        Tensor tdeltaY = eg.tensor(deltaY, N, OH, OW, OC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tdeltaX = eg.empty(N, IH, IW, IC);
                
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 500;
        for(int i=0; i<nIter; i++)
        {
            tdeltaX = eg.dconv3D_deltaX(tdeltaY, tW, IH, IW, sh, sw, ph, pw).c(); eg.delete(tdeltaX);//sync
//            tdeltaX = eg.dconv3D_deltaX(tdeltaX, tdeltaY, tW, sh, sw);//async
        }
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStampDifMills();
        float time = 1.0f*div/nIter;
        double sizeV = 1.0* GN/1024*GM/1024*GK/1024;
        float performance = (float)((sizeV*1024*1024 * 1e-9 * 2 * 1024)/(time*1e-3));
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
        
        eg.delete(tdeltaX, tdeltaY, tW);
    }
    
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        //test for kernel============
//	int OH = 15, OW = 15;
//	int FH = 8, FW = 8;
//	int N = 4;
//	int IC = 128, OC = 17;//9*4=36 
//	int sh = 4, sw = 4, ph = 1, pw = 1;
     
        //test for W1===========================================================
//        int OH = 31, OW = 31;
//        int OH = 15, OW = 15;
//        int OH = 7, OW = 7;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 192, OC = 32;//128+64+32+16+8+4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//        for(int ic=1; ic<=128; ic++) 
//            testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
//        
//        int OH = 64, OW = 64;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 255, OC = 128;//128+64+32+16+8+4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//        testCorrect(-1, -1, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(-1, -1, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test for s1===========================================================
//        int OH = 4, OW = 4, N = 64;
//        int OH = 16, OW = 16, N = 4;
	
//        int FH = 5, FW = 5, ph = 2, pw = 2;
//        int FH = 4, FW = 4, ph = 1, pw = 1;
//        int FH = 3, FW = 3, ph = 1, pw = 1;
//	int IC = 128, OC = 16;//9*4=36 
//	int sh = 1, sw = 1;
        
//        for(int ic = 1; ic <= 128; ic++) 
//            testCorrect(-1, -1, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
        
//        int IH = -1, IW = -1;
//        int OH = 31, OW = 31;
//	int FH = 4, FW = 4;
//	int N = 4;
//	int IC = 255, OC = 32;//128+64+32+16+8+4
//	int sh = 1, sw = 1, ph = 1, pw = 1;
        
//        int IH = -1, IW = -1;
//        int OH = 32, OW = 32;
//	int FH = 3, FW = 3, ph = 1, pw = 1;
////        int FH = 5, FW = 5, ph = 2, pw = 2;
//	int N = 8;
//	int IC = 128, OC = 64;//128+64+32+16+8+4
//	int sh = 1, sw = 1;
        
//        int FH = 3, FW = 3, OC = 256, ph = 1, pw = 1;//4*4*8 = 32*4 = 128
//        int FH = 5, FW = 5, OC = 128, ph = 2, pw = 2;//4*4*8 = 32*4 = 128

        //int IH = 2, IW = 2, OH = 2, OW = 2, N = 512;
//	int IH = 4, IW = 4, OH = 4, OW = 4, N = 128;
//	int IH = 8, IW = 8, OH = 8, OW = 8, N = 128; OC /= 4;
//	int IH = 16, IW = 16, OH = 16, OW = 16, N = 64; OC /= 8;
	//int IH = 32, IW = 32, OH = 32, OW = 32, N = 64; OC /= 16; 
        
//        int IC = 224;//9*4=36 3*3*3 = 9*3 = 27;
//	int sh = 1, sw = 1;
      
//        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, OH, OW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test ImsR-------------------------------------------------------------
//        int IH = 11, IW = 11;
//	int OH = 6, OW = 6;
//        int IH = 7, IW = 7;
//        int OH = 4, OW = 4;
//        int IH = 15, IW = 15;
//	int OH = 8, OW = 8;

//        int IH = 12, IW = 12, OH = 6, OW = 6;
////        int IH = 16, IW = 16, OH = 8, OW = 8;
//	int FH = 3, FW = 3;//4*4*8 = 32*4 = 128//3*3
//	int N = 4;
//	int IC = 128, OC = 32;//9*4=36 3*3*3 = 9*3 = 27;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//        for(int ic = 1; ic <= 128; ic++) {
//            int size = Cuda_dconv3D_deltaX.streamSize_KernelSplit(IH, IW, N, ic, sh, sw);
//            System.out.format("IC, N, IH, IW = [%d, %d, %d, %d]\n", ic, N, IH, IW);
//            System.out.println("size = " + size);
//            
////            testCorrect(IH, IW, OH, OW, FH, FW, N, ic, OC, sh, sw, ph, pw);
//        }
        
//        int IH = 31, IW = 31;
//	int OH = 16, OW = 16;
//	int FH = 3, FW = 3;//4*4*8 = 32*4 = 128
//	int N = 8;
//	int IC = 256, OC = 32;//9*4=36 3*3*3 = 9*3 = 27;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
        
//        int IH = 32, IW = 32;
//	int OH = 16, OW = 16;
//	int FH = 3, FW = 3;//4*4*8 = 32*4 = 128
//	int N = 8, OC = 64;
//	int IC = 256;//9*4=36 3*3*3 = 9*3 = 27;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
        
//        int IH = 32, IW = 32, OH = 16, OW = 16, OC = 16;
	//int IH = 16, IW = 16, OH = 8, OW = 8, OC = 16;
	int IH = 8, IW = 8, OH = 4, OW = 4, OC = 64;
//	int IH = 4, IW = 4, OH = 2, OW = 2, OC = 256;
	int FH = 3, FW = 3, ph = 1, pw = 1;
	//int FH = 5, FW = 5, ph = 2, pw = 2; OC /= 2;
//	
        int N = 128, sh = 2, sw = 2;
	int IC = 248;//9*4=36 3*3*3 = 9*3 = 27;
        
        testCorrect(IH, IW, OH, OW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        testSpeed(IH, IW, OH, OW, FH, FW, N*2, IC, OC*2, sh, sw, ph, pw);
    }
}

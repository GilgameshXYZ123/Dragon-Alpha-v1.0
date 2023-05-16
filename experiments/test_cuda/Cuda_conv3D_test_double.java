package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.dragon.engine.memp.Mempool;
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
public class Cuda_conv3D_test_double 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Mempool memp = alpha.engine.memp1();
    static Engine eg = alpha.engine.cuda_float32(0, memp);
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
        //cu32.setConv3D_useTexture(true);
    }
     
     public static void testCorrect(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.sync(false);
        int[] ODim = Cuda_conv3D.outputTensorDim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int[] Gsize = Cuda_conv3D.getImg2colMatrixDim(OH, OW, FH, FW, N, IC, OC);
        int GN=Gsize[0], GM=Gsize[1], GK=Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.randomFloatVector(sizeX);
        float[] W = Vector.randomFloatVector(sizeW);
        
        //CPU-------------------------------------------------------------------
        double[][][][] cX = Vector.toTense4D_double(X, N, IH, IW, IC);
        double[][][][] cW = Vector.toTense4D_double(W, OC, FH, FW, IC);
        double[][][][] cY = new double[N][OH][OW][OC];
        
        Tense.conv3D_naive_double(
                cX, IH, IW,
                cW, FH, FW,
                cY, OH, OW,
                N, IC, OC, 
                sh, sw, ph, pw);
        
        double[] Y1 = Vector.toVector_double(cY);
        System.out.print("CPU1: "); 
        for(int i=0; i<10; i++) System.out.print(Y1[i] + ", ");
        System.out.println();
        
        //GPU-------------------------------------------------------------------
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tY1 = eg.conv3D(tX, tW, sh, sw, ph, pw).c();
        Tensor tY2 = eg.conv3D(eg.empty(N, OH, OW, OC), tX, tW, sh, sw).c();
        
        float[] Y2 = eg.valueOf(tY1); System.out.print("GPU1 : "); Vector.println(Y2, 0, 10);
        float[] Y3 = eg.valueOf(tY2); System.out.print("GPU2 : "); Vector.println(Y3, 0, 10);
        float sp1 = Vector.samePercentRelative(Y1, Y2, 1e-6f); System.out.println("sp1: "+sp1);
        float sp2 = Vector.samePercentRelative(Y1, Y3, 1e-6f); System.out.println("sp2: "+sp2);
        
        eg.delete(tX, tW, tY1, tY2);
        
        if(sp1 != 1) throw new RuntimeException();
        if(sp2 != 1) throw new RuntimeException();
    }
    public static void testSpeed(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false);
        eg.sync(false);
        int[] ODim = Cuda_conv3D.outputTensorDim(IH, IW, FH, FW, N, OC, sh, sw, ph, pw);
        int OH = ODim[1], OW = ODim[2];
        
        System.out.println("Test correct:");
        System.out.format("\tXsize( N, IC, IH, IW) = (%d, %d, %d, %d)\n", N, IC, IH, IW);
        System.out.format("\tWsize(OC, IC, FH, FW) = (%d, %d, %d, %d)\n", OC, IC, FH, FW);
        System.out.format("\tYsize( N, OC, OH, OW) = (%d, %d, %d, %d)\n", N, OC, OH, OW);
        
        int[] Gsize = Cuda_conv3D.getImg2colMatrixDim(OH, OW, FH, FW, N, IC, OC);
        int GN=Gsize[0], GM=Gsize[1], GK=Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;//256*256
	int sizeW = OC*IC*FH*FW;
        
        float[] X=Vector.randomFloatVector(sizeX);
        float[] W=Vector.randomFloatVector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        Tensor tY = eg.empty(N, OH, OW, OC);
        
        SimpleTimer timer=new SimpleTimer();
        timer.record();
        int nIter = 1000;
        for(int i=0;i<nIter;i++)
        {
            //sync--------------------------------------------------------------
//            tY = eg.conv3D(tX, tW, sh, sw, ph, pw); eg.delete(tY.c());//sync
            tY = eg.conv3D(tY, tX, tW, sh, sw);//async
        }
        Cuda.deviceSynchronize();
        timer.record();
        
        long div=timer.timeStampDifMills();
        float time = 1.0f*div/nIter;
        double sizeV = 1.0 * GN/1024*GM/1024*GK/1024;
        float performance = (float) ((sizeV*1024*1024 * 1e-9 * 2 * 1024)/(time * 1e-3));
        
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
    }
    
    public static void main(String[] args)
    {
        //for conv3D
//        int IH = 33, IW = 33;
//	int FH = 4, FW = 4;
//	int sh = 1, sw = 1, ph = 1, pw = 1;
//	int N = 2;
//	int IC = 16, OC = 128;//9*4=36 
        
        //for conv3D_W1
//        int IH = 64, IW = 64;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 128, OC = 255;//128+64+32+16+8+4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
        
        //(33 - 4 + 2) + 1 = 33 - 2 + 1 = 32
        int IH = 33, IW = 33;
	int FH = 4, FW = 4;
	int sh = 1, sw = 1, ph = 1, pw = 1;
	int N = 4;
	int IC = 16, OC = 128;//9*4=36 
        
//        int IH = 32, IW = 32;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 128, OC = 16;//128+64+32+16+8+4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
        
        //(N, IC, OC) = (4, 128, 12)
//        testCorrect(IH, IW, FH, FW, N, IC, 12, sh, sw, ph, pw);
        
//        for(int oc = 64; oc <= 128; oc++) Cuda_conv3D_test.testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//        Cuda_conv3D_test.testCorrect(IH, IW, FH, FW, N, IC, 93, sh, sw, ph, pw);
        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
//        System.out.println(CudaException.lastException());
    }
}

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
public class Cuda_conv3D_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Mempool memp = alpha.engine.memp1();
    static Engine eg = alpha.engine.cuda_float32(0, memp);
     
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
        int GN = Gsize[0], GM = Gsize[1], GK = Gsize[2];
        System.out.format("(GN, GM, GK) = (%d, %d, %d)\n", GN, GM, GK);
        
        int sizeX = N*IC*IH*IW;
	int sizeW = OC*IC*FH*FW;
	int sizeY = N*OC*OH*OW;
        
        float[] X = Vector.randomFloatVector(sizeX);
        float[] W = Vector.randomFloatVector(sizeW);
        
        Tensor tX = eg.tensor(X, N, IH, IW, IC);
        Tensor tW = eg.tensor(W, OC, FH, FW, IC);
        
        //CPU-------------------------------------------------------------------
        float[][][][] cX = Tense.vectorToTensor_4D(X, N, IH, IW, IC);
        float[][][][] cW = Tense.vectorToTensor_4D(W, OC, FH, FW, IC);
        float[][][][] cY = new float[N][OH][OW][OC];
        
        Tense.conv3D_naive(
                cX, IH, IW,
                cW, FH, FW,
                cY, OH, OW,
                N, IC, OC, 
                sh, sw, ph, pw);
        
        float[] Y1 = Tense.tensor_4DToVector(cY, sizeY);
        
        //GPU-------------------------------------------------------------------
        Tensor tY1 = eg.conv3D(tX, tW, sh, sw, ph, pw);
        Tensor tY2 = eg.conv3D(eg.empty(N, OH, OW, OC).c(), tX, tW, sh, sw);

        float[] Y2 = eg.valueOf(tY1);
        float[] Y3 = eg.valueOf(tY2);
        
        System.out.println(tY1);
        System.out.println(tY2);
        
        //compare----------------------------------------------------------------
        float sp1 = Vector.samePercentRelative(Y1, Y2, 1e-5f); 
        float sp2 = Vector.samePercentRelative(Y1, Y3, 1e-5f);
      
        System.out.print("CPU : "); Vector.println(Y1, 0, 10);
        System.out.print("GPU1: "); Vector.println(Y2, 0, 10);
        System.out.print("GPU2: "); Vector.println(Y3, 0, 10);
        System.out.println("sp1: " + sp1);
        System.out.println("sp2: " + sp2);
         
        eg.delete(tX, tW, tY1, tY2);
        
        if(sp1 <0.999f || sp2 < 0.999f) throw new RuntimeException();
    }
     
    public static void testSpeed(
            int IH, int IW, 
            int FH, int FW, 
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        eg.check(false).sync(false);
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
        Tensor tY = eg.empty(N, OH, OW, OC).c();
        
        SimpleTimer timer = new SimpleTimer().record();
        int nIter = 1000;
        for(int i=0;i<nIter;i++) {
            tY = eg.conv3D(tX, tW, sh, sw, ph, pw); eg.delete(tY.c());//sync
        }
        Cuda.deviceSynchronize();
        
        long div = timer.record().timeStampDifMills();
        float time = 1.0f*div/nIter;
        double sizeV = 1.0 * GN/1024 * GM/1024 * GK/1024;
        float performance = (float) ((sizeV*1024*1024 * 1e-9 * 2 * 1024)/(time * 1e-3));
        
        System.out.format("Size = %f, Time = %f msec, Performance = %f GFlop/s\n", 
                (float)sizeV, time, performance);
    }
    
    public static void main(String[] args)
    {
        //Vector.PRINT_DIFFERENT = true;
        
        //test W11==============================================================
//        int IH = 32, IW = 32;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 128, OC = 16;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//        for(int oc = 1; oc <= 128; oc++) 
//            testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//        
//        int IH = 64, IW = 64;
//	int FH = 1, FW = 1;
//	int N = 4;
//	int IC = 128, OC = 255;
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test np===============================================================
//        int IH = 33, IW = 33;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 16, OC = 128;//9*4=36 
//        for(int oc = 1; oc <= 128; oc++) 
//            testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);

//        int IH = 35, IW = 35;//(33 - 4 + 2)/1 + 1
//	int FH = 4, FW = 4;//FH = 4, FW = 4
//	int sh = 1, sw = 1, ph = 0, pw = 0;
//	int N = 4;
//	int IC = 64, OC = 192;//9*4=36 
//        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
//        testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
        
        //test common===========================================================
//        int IH = 33, IW = 33;//(33 - 4 + 2)/1 + 1
//	int FH = 4, FW = 4;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 4;
//	int IC = 16, OC = 128;//9*4=36 
//
//        int IH = 16, IW = 16;//(33 - 4 + 2)/1 + 1
////	int FH = 3, FW = 3;
//        int FH = 4, FW = 4;
////        int FH = 5, FW = 5;
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 4;
//	int IC = 16, OC = 128;//9*4=36 
//        for(int oc = 1; oc <= 128; oc++) {
//            int OH = (IH - FH + 2*ph)/sh + 1;
//            int OW = (IW - FW + 2*pw)/sw + 1;
//            
//            System.out.format("OH, OW, OC = [%d, %d, %d]\n", OH, OW, oc);
//            
////            int size = Cuda_conv3D.streamSize(OH, OW, N, oc);
//            int size = Cuda_conv3D.streamSizeV2(N, oc);
//            System.out.println("size = " + size);
//            
////            testCorrect(IH, IW, FH, FW, N, IC, oc, sh, sw, ph, pw);
//        }
        
//        
        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
	int FH = 4, FW = 4;//FH = 4, FW = 4
	int sh = 2, sw = 2, ph = 1, pw = 1;
	int N = 16;
        int IC = 64, OC = 252;//9*4=36 

//        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
//	int FH = 3, FW = 3;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 1, pw = 1;
//	int N = 32;
//        int IC = 64, OC = 252;//9*4=36 

//        int IH = 32, IW = 32;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 2, pw = 2;
//	int N = 16;
//        int IC = 64, OC = 248;//9*4=36 

//        int IH = 4, IW = 4;//(33 - 4 + 2)/1 + 1
//	int FH = 5, FW = 5;//FH = 4, FW = 4
//	int sh = 2, sw = 2, ph = 2, pw = 2;
//	int N = 129;
//        int IC = 128, OC = 512;//9*4=36 

        testCorrect(IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        testSpeed(IH, IW, FH, FW, N*2, IC, OC, sh, sw, ph, pw);
    }
}

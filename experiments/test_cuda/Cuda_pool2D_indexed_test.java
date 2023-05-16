/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_pool2D_indexed_test 
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
        
        System.out.println("TestCorrect:");
        System.out.format("\t(IH, IW) = (%d, %d)\n", IH, IW);
        System.out.format("\t(FH, FW) = (%d, %d)\n", FH, FW);
        System.out.format("\t(OH, OW) = (%d, %d)\n", OH, OW);
        System.out.format("\t(N, IC) = (%d, %d)\n", N, IC);
        System.out.format("\t(sh, sw, ph, pw) = (%d, %d, %d, %d)\n", sh, sw, ph, pw);
        
        int sizeX = N * IH * IW * IC;
        int sizeY = N * OH * OW * IC;
        
        //prepared--------------------------------------------------------------
        Tensor X = eg.gaussian(0, 4, N, IH, IW, IC).c();
        Tensor deltaY = eg.gaussian(0, 4, N, OH, OW, IC).c();
        
        Tensor[] R = eg.pool2D_max_indexed(X, FH, FW, sh, sw, ph, pw);
        Tensor Y = R[0].c(), Index = R[1];
        
        //GPU0------------------------------------------------------------------
        Tensor deltaX1 = eg.upool2D_max(deltaY, Y, X, FH, FW, sh, sw, ph, pw).c();
        Cuda_expk2.checkMemAlign(deltaX1);
        
        Tensor A1 = eg.reshape(false, deltaX1, N*IH*IW, IC).c();
        Tensor B1 = eg.matMulT2(A1, A1).c();
        
        //GPU1------------------------------------------------------------------
        Tensor deltaX2 = eg.upool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, sh, sw, ph, pw).c();
        Cuda_expk2.checkMemAlign(deltaX2);
        Tensor A2 = eg.reshape(false, deltaX2, N*IH*IW, IC).c();
        Tensor B2 = eg.matMulT2(A2, A2).c();
       
        //compare---------------------------------------------------------------
        float[] dX1 = deltaX1.value(); 
        float[] dX2 = deltaX2.value();
        
        System.out.println("dX1 = "); Vector.println(dX1, 0, 10);
        System.out.println("dX2 = "); Vector.println(dX2, 0, 10);
        
        float zp0 = Vector.zeroPercent(dX1); System.out.println("zp0 = " + zp0);
        float zp1 = Vector.zeroPercent(dX2); System.out.println("zp1 = " + zp1);
        
        float sp = Vector.samePercentAbsolute(dX1, dX2);
        System.out.println("sp = " + sp);
        
        //compare---------------------------------------------------------------
        float[] V1 = B1.value();
        float[] V2 = B2.value();
        
        float sp2 = Vector.samePercentAbsolute(V1, V2);
        System.out.println("sp2 = " + sp2);
       
        if(sp  < 0.99f) throw new RuntimeException();
        if(sp2 < 0.99f) throw new RuntimeException();
        
        eg.delete(A2, B2, A1, B1, deltaX1, deltaX2);
    }
 
    
    public static void main(String[] args) 
    {
        int IH = 32, IW = 32;
        int N = 4, IC = 128;
	int FH = 4, FW = 4;
	int sh = 2, sw = 2, ph = 1, pw = 1;
        try
        {
//            Vector.PRINT_DIFFERENT = true;
            
            for(int ic = 1; ic <= 128; ic++)
                testCorrect(IH, IW, FH, FW, N, ic, sh, sw, ph, pw);
            
//            testCorrect(31, 31, 3, 3, 4, 72, 1, 2, 1, 1);
//            testCorrect(IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
//            testSpeed(IH, IW, FH, FW, N*2, IC, sh, sw, ph, pw);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}

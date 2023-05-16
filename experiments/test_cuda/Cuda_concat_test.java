/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static test.cuda.Cuda_function_test.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.memp.Mempool;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_concat_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void testCorrect(int dimIdx, int[]... dims) 
    {
        System.out.println("testCorrect: dimIdx = " + dimIdx);
        for(int i=0; i<dims.length; i++) {
            System.out.print("dim[" + i + "] = "); Vector.println(dims[i]);
        }
        
        float[][] A = new float[dims.length][];
        Tensor[] tA = new Tensor[dims.length];
        for(int i=0; i<A.length; i++) {
            A[i] = Vector.randomFloatVector(Vector.mul(dims[i]));
            tA[i] = eg.tensor(A[i], dims[i]).c();
        }
        
        //CPU-------------------------------------------------------------------
        float[][][][][] cA = new float[dims.length][][][][];
        for(int i=0; i<cA.length; i++) {
            int[] dim = dims[i];
            cA[i] = Vector.to4D(A[i], dim[0], dim[1], dim[2], dim[3]);
        }
        float[][][][] cY = Vector.concat4D(dimIdx, cA);
        
        float[] Y1 = Vector.toVector(cY);
        Vector.println("CPU: ", Y1, 0, 10);
        
        //GPU-------------------------------------------------------------------
        Tensor tY = eg.concat(dimIdx, tA).c();
        System.out.println(tY);
        float[] Y2 = tY.value();
        Vector.println("GPU: ", Y2, 0, 10);
        
        
        //compare---------------------------------------------------------------
        float zp1 = Vector.zeroPercent(Y1); System.out.println("zp1 = " + zp1);
        float zp2 = Vector.zeroPercent(Y2); System.out.println("zp2 = " + zp2);
        float sp = Vector.samePercentAbsolute(Y1, Y2); System.out.println("sp = " + sp);
        
        eg.delete(tA);
        eg.delete(tY);
        if(sp != 1) throw new RuntimeException();
    }
    
    public static void testSpeed(int dimIdx, int[]... dims) 
    {
        System.out.println("testSpeed: dimIdx = " + dimIdx);
        for(int i=0; i<dims.length; i++) {
            System.out.print("dim[" + i + "] = "); Vector.println(dims[i]);
        }
        
        
        float[][] A = new float[dims.length][];
        Tensor[] tA = new Tensor[dims.length];
        for(int i=0; i<A.length; i++) {
            A[i] = Vector.randomFloatVector(Vector.mul(dims[i]));
            tA[i] = eg.tensor(A[i], dims[i]).c();
        }
        
        //GPU-------------------------------------------------------------------
        int nIter = 1000;
        SimpleTimer timer = new SimpleTimer().record();
        int length = 0;
        for(int i=0; i<nIter; i++)
        {
            Tensor tY = eg.concat(dimIdx, tA).c();
            length = tY.length();
            eg.delete(tY);
        }
        long div = timer.record().timeStampDifMills();
        float time = 1.0f * div / nIter;
        int dataSize = length * 4 * 2;
        float speed =  ((float)dataSize) / (1 << 30) / (time * 1e-3f);
        System.out.format("Time = %f, Speed = %f GB/s\n", time, speed);
        eg.delete(tA);
    }
    
    public static void main(String[] args) 
    {
        {
            int d0 = 32, d1 = 16, d2 = 16, d3 = 32;
            int[] dim0 = new int[]{d0, d1, d2, d3};
            int[] dim1 = new int[]{d0, d1, d2, d3};
            int[] dim2 = new int[]{d0, d1, d2, d3};
            int[] dim3 = new int[]{d0, d1, d2, d3};
            testCorrect(3, dim0, dim1, dim2, dim3);
            testSpeed(3, dim0, dim1, dim2, dim3);
        }
//      
//        
        for(int d0 = 1; d0 <= 32; d0++) 
        for(int d1 = 1; d1 <= 8; d1++)  
        for(int d2 = 1; d2 <= 16; d2++)
        for(int d3 = 1; d3 <= 8; d3++)
        {
            int[] dim0 = new int[]{d0, d1, d2, d3};
            int[] dim1 = new int[]{d0, d1, d2, d3};
            int[] dim2 = new int[]{d0, d1, d2, d3};
            testCorrect(-1, dim0, dim1, dim2);
        }
    }
}

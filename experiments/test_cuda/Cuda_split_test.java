/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_split_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) eg.engineBase();
        //cu32.setConv3D_useTexture(true);
    }
    
    public static void testCorrect(int[] dim, int dimIdx, int[] section) 
    {
        System.out.println("TestCorrect: dimIdx = " + dimIdx);
        System.out.print("dim = "); Vector.println(dim);
        System.out.print("section = "); Vector.println(dim);
        
        int size = Vector.mul(dim);
        float[] A = Vector.randomFloatVector(size);
        Tensor tA = eg.tensor(A, dim).c();
        
        //CPU-------------------------------------------------------------------
        float[][][][] cA = Vector.to4D(A, dim[0], dim[1], dim[2], dim[3]);
        float[][][][][] cY = Vector.split4D(cA, dimIdx, section);
        
        float[][] Y1 = new float[section.length][];
        for(int i=0; i<section.length; i++)  Y1[i] = Vector.toVector(cY[i]);
        
        //GPU-------------------------------------------------------------------
        Tensor[] tY = eg.split(tA, dimIdx, section);
        
        float[][] Y2 = new float[section.length][];
        for(int i=0; i<section.length; i++) Y2[i] = tY[i].value();
        
        //compare---------------------------------------------------------------
        for(int i=0; i<section.length; i++) {
            System.out.print("CPU: "); Vector.println(Y1[i], 0, 10);
            System.out.print("GPU: "); Vector.println(Y2[i], 0, 10);
            float sp = Vector.samePercentAbsolute(Y1[i], Y2[i]);
            float zp0 = Vector.zeroPercent(Y1[i]);
            float zp1 = Vector.zeroPercent(Y2[i]);
            System.out.println("[sp, zp0, zp1] = [" + sp + ", " + zp0 + ", " + zp1 + ']');
            
            if(sp != 1) throw new RuntimeException();
        }
        
        eg.delete(tY);
        eg.delete(tA);
    }
    
    public static void main(String[] args)
    {
        {
            int[] dim = new int[]{128, 32, 32, 4};
            int[] section = new int[]{32, 32, 32, 32};
            testCorrect(dim, 0, section);
        }
        
//        int[] section = new int[]{32, 32, 32, 32};
//        int[] section = new int[]{4, 12, 8, 8};
        int[] section = new int[]{3, 3, 3, 3};
        
        for(int d3=1; d3<=32; d3++){
            int[] dim = new int[]{128, 32, d3, 12};
            testCorrect(dim, 3, section);
        }
    }
    
    
}

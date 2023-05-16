package test.cuda;


import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.cuda.impl.CudaDevice;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Gilgamesh
 */
public class CudaDevice_test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    public static void main(String[] args)
    {
//        CudaDevice dev = new CudaDevice(0);
//        System.out.println(dev.getId());
//        System.out.println(dev.getL2CacheSize());
        
//        int x = (1<<30) - 3;
//        System.out.println((int)(float)x);
//        System.out.println(x);
        
        CudaDevice dev = new CudaDevice(0);
        System.out.println(dev.getRegsPerBlock());
        System.out.println(dev);
    }
}

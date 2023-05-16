package test.cuda;

import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_deltaX_test2 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
    
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * stride;
        int length = height * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, -5, 5);
        Tensor tX = eg.tensor(X, height, width); Vector.println("X = ", X, 0, 10);
        
        float[] deltaY = Vector.randomFloatVector(length, 0, 1); Vector.println("deltaY = ", deltaY, 0, 10);
        Tensor tdeltaY = eg.tensor(deltaY, height, width);
        float alpha = exr.nextFloat(0, 0.8f); System.out.println("alpha = " + alpha);
        float k = exr.nextFloat(0, 0.5f); System.out.println("k = " + k);
        
        //GPU-------------------------------------------------------------------
//        Tensor tY = eg.leakyRelu(false, tX, k).c();
//        Tensor tdeltaX1 = eg.leakyRelu_deltaX_v1(false, tdeltaY, tY, k).c();
//        Tensor tdeltaX2 = eg.leakyRelu_deltaX_v2(false, tdeltaY, tX, k).c();

//        Tensor tY = eg.relu(false, tX).c();
//        Tensor tdeltaX1 = eg.relu_deltaX_v1(false, tdeltaY, tY).c();
//        Tensor tdeltaX2 = eg.relu_deltaX_v2(false, tdeltaY, tX).c();

//        Tensor tY = eg.elu(false, tX, alpha, k).c();
//        Tensor tdeltaX1 = eg.elu_deltaX_v1(false, tdeltaY, tY, alpha, k).c();
//        Tensor tdeltaX2 = eg.elu_deltaX_v2(false, tdeltaY, tX, alpha, k).c();

//        Tensor tY = eg.softplus(false, tX).c();
//        Tensor tdeltaX1 = eg.softplus_deltaX_v1(false, tdeltaY, tY).c();
//        Tensor tdeltaX2 = eg.softplus_deltaX_v2(false, tdeltaY, tX).c();
            
//        Tensor tY = eg.tanh(false, tX).c();
//        Tensor tdeltaX1 = eg.tanh_deltaX_v1(false, tdeltaY, tY).c();
//        Tensor tdeltaX2 = eg.tanh_deltaX_v2(false, tdeltaY, tX).c();

        Tensor tY = eg.sigmoid(false, tX).c();
        Tensor tdeltaX1 = eg.sigmoid_deltaX_v1(false, tdeltaY, tY).c();
        Tensor tdeltaX2 = eg.sigmoid_deltaX_v2(false, tdeltaY, tX).c();
        
        //compare---------------------------------------------------------------
        float[] tX1 = tdeltaX1.value();
        float[] tX2 = tdeltaX2.value();
        
        Vector.println("deltaX1: ", tX1, 0, 10);
        Vector.println("deltaX2: ", tX2, 0, 10);
        float sp1 = Vector.samePercentRelative(tX1, tX2, 1e-5f); System.out.println("sp1:" + sp1);
       
        //delete----------------------------------------------------------------
        eg.delete(tY, tdeltaX2, tdeltaX1);
        
        if(sp1 < 0.95) throw new RuntimeException();
    }
    
  
    public static void main(String[] args)
    {
        Vector.PRINT_DIFFERENT = true;
        
        for(int h=1; h<=10; h++)
            for(int w=1; w<=256; w++) testCorrect(h, w);
        
        for(int h=100; h<=210; h++)
            for(int w=100; w<=125; w++) testCorrect(h, w);
        
        testCorrect(1024, 1024);
    }

}

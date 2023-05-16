package test.cuda;


import z.dragon.engine.Engine;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.impl.CudaException;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_reduce_row_test3 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static final ExRandom exr = new ExRandom();
   
    public static void testCorrect(int height, int width)
    {
        int stride = (width + 3) >> 2 << 2;
	int lengthv = height * 3 * 2 * stride;
        int length  = height * 3 * 2 * width;
        
        System.out.println("\nTest Corret:");
        System.out.format("(height, width, stride) = (%d, %d, %d)\n", height, width, stride);
        System.out.format("(lengthv, length) = (%d, %d)\n", lengthv, length);
        
        float[] X = Vector.randomFloatVector(length, 3f, 6f);
        Tensor tX = eg.tensor(X, height, 3, 2, width);
        
        //CPU-------------------------------------------------------------------
        float[][] mX = Matrix.toMatrix(X, 3 * 2 * width);
        
        float[] mean1 = Matrix.row_mean(mX);
        float[] smean1 = Matrix.row_squareMean(mX);
        
        //GPU-------------------------------------------------------------------
        Tensor[] result = eg.row_mean_sqmean(tX, 2 * 3 * width);
        float[] mean2  = result[0].value();
        float[] smean2 = result[1].value();
        
        //compare---------------------------------------------------------------
        System.out.print("mean1: "); Vector.println(mean1, 0, 10);
        System.out.print("mean2: "); Vector.println(mean2, 0, 10);
        System.out.print("smean1: "); Vector.println(mean1, 0, 10);
        System.out.print("smean2: "); Vector.println(mean2, 0, 10);
        
        float sp1 = Vector.samePercentRelative(mean1, mean2);
        float sp2 = Vector.samePercentRelative(smean1, smean2);
        System.out.println("sp1 = " + sp1);
        System.out.println("sp2 = " + sp2);
        
        if(sp1 < 0.99 && sp2 < 0.99) throw new RuntimeException();
    }
    
    public static void main(String[] args)
    {
        try
        {
            Vector.PRINT_DIFFERENT = true;
//            (3, 1), (3, 2,
            for(int h = 1; h <= 20; h++)
                for(int w = 1; w <= 256; w++) testCorrect(h, w);
//            
            for(int h=100; h<=105; h++)
                for(int w= 40; w<=64; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w=7; w<=12; w++) testCorrect(h, w);
            
            for(int h=300; h<=305; h++)
                for(int w= 140; w<=164; w++) testCorrect(h, w);
            
            testCorrect(1024, 1024);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        System.out.println(CudaException.lastException());
    }
}

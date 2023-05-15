/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 * @author Gilgamesh
 */
public final class Cuda_random 
{
    private Cuda_random() {}
    
    //<editor-fold defaultstate="collapsed" desc="Bernouli Distribution">
    /**
     * <pre>
     * Random Vector: Bernouli Distribution.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) p belongs to (0, 1): the probability of the first value(v1)
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024: 0.085000, Speed = 45.955879 GB/s
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param seed
     * @param p
     * @param v1
     * @param v2
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void bernouli2D(long cudaStream_address,
            long dX_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride);
    /**
     * <pre>
     * Random Vector: 
     * [1] R = bernouli(p, v1, v2)
     * [2] elemntwise_mul: Y = R * X.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) p belongs to (0, 1): the probability of the first value(v1)
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024: 0.085000, Speed = 45.955879 GB/s
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param dR_address
     * @param dY_address
     * @param seed
     * @param p
     * @param v1
     * @param v2
     * @param lengthv
     * @param width
     * @param stride 
     */
    public static native void bernouli_mul2D(long cudaStream_address,
            long dX_address, 
            long dR_address, long dY_address,
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Uniform Distribution">
    /**
     * <pre>
     * Uniform Distribution.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024: Time = 0.090000, Speed = 43.402775 GB/s 
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param seed
     * @param vmin
     * @param vmax
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void uniform2D(long cudaStream_address,
            long dX_address, 
            int seed,
            float vmin, float vmax,
            int lengthv, int width, int stride);
    
    /**
     * <pre>
     * Sparse Uniform Distribution.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) p belongs to [0, 1], controling the posibility of 1 
     * (5) X[i] = uniform(min, max) * bernouli(1, 0, p)
     * (6) seed1 != seed2
     * 
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024: Time = 0.082000, Speed = 47.637192 GB/s
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param seed1
     * @param seed2
     * @param p the posibility of success
     * @param vmin
     * @param vmax
     * @param lengthv
     * @param width 
     * @param stride 
     */
    public static native void sparse_uniform2D(long cudaStream_address,
            long dX_address, 
            int seed1, int seed2, 
            float p, float vmin, float vmax,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Gaussian Distribution">
    /**
     * <pre>
     * Gaussian Distribution.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) sigma > 0
     * (5) seed1 != seed2
     * 
     * ----Use Box - Muller method(Inverse Transforming Sampling)---------------
     * (1) v1 = next_float(seed1), v1 belongs to (0, 1), v1 obeys uniform(0, 1)
     * (2) v2 = next_float(seed2), v2 belongs to [0, 1], v2 obeys uniform(0, 2)
     * (3) v1 and v2 is indepent
     * 
     * (4) u = sqrt(-2 * log(v1))
     * (5) a1 = u*sin(2pi * v2)
     * (6) a2 = u*cos(2pi * v2)
     * (7) a1, and a2 obeys N(0, 1)
     * 
     * (8) x1 = sigma * a1 + mu
     * (9) x2 = sigma * a2 + mu
     * (10) x1, x2 obeys N(mu, sigma^2)
     * (1) X belongs to Vector[length]
     * (2) X[i] obeys N(mu, sigma^2)
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024: Time = 0.130000, Speed = 30.048077 GB/s
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param seed1
     * @param seed2
     * @param mu
     * @param sigma
     * @param lengthv
     * @param width 
     * @param stride 
     */
    @Passed
    public static native void gaussian2D(long cudaStream_address,
            long dX_address,
            int seed1, int seed2, 
            float mu, float sigma,
            int lengthv, int width, int stride);
    
    /**
     * <pre>
     * Gaussian Distribution.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) sigma > 0
     * (5) seed1 != seed2 != seed3
     * (6) p control the posibility of the first value of the corresponding 
     * bernouli distribution.
     * (7) X[i] = uniform(mu, sigma) * bernouli(1, 0, p)
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [length] from [1] to(+1) [2048]: correct
     * for [length] = 1024*1024:  Time = 0.130000, Speed = 30.048077 GB/s
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param seed1
     * @param seed2
     * @param seed3
     * @param p
     * @param mu
     * @param sigma
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void sparse_gaussian2D(long cudaStream_address,
            long dX_address,
            int seed1, int seed2, int seed3, 
            float p, float mu, float sigma,
            int lengthv, int width, int stride);
    //</editor-fold>
}

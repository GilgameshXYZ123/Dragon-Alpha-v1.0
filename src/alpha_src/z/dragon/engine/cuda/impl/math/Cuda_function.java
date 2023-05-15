/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
public final class Cuda_function 
{
    private Cuda_function() {}

    //<editor-fold defaultstate="collapsed" desc="greater, equal, linear, quadratic, rpl, div, add_div"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs2D">
    /**
     * <pre>
     *  max >= |X1 - X2| >= min.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param dX2_address
     * @param min
     * @param max
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void equal_abs2D(long stream_address,
            long dX1_address, long dX2_address,
            float min, float max,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     *  max >= |X1 - X2| >= min.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param dX2_address
     * @param min
     * @param max
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void equal_abs2D_char(long stream_address,
            long dX1_address, long dX2_address,
            byte min, byte max,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     *  max >= |X1 - X2| >= min.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param dX2_address
     * @param min
     * @param max
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void equal_abs2D_int(long stream_address,
            long dX1_address, long dX2_address,
            int min, int max,
            long dY_address,
            int lengthv, int width, int stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater2D, linear_greater_dual2D">
    /**
     * <pre>
     *  alpha*X + beta > 0.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param stream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void linear_greater2D(long stream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     *  alpha*X1 + beta*X2 + gamma > 0.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * </pre>
     *
     * @param stream_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void linear_greater_dual2D(long stream_address,
            long dX1_address, long dX2_address,
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int width, int stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="linear2D">
    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
   
    /**
     * <pre>
     * Linear Transformation: 
     * [1] Y1 = alpha1 * X + beta1
     * [2] Y2 = alpha2s * X + beta2.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha1
     * @param beta1
     * @param dY1_address
     * @param lengthv
     * @param alpha2
     * @param dY2_address
     * @param beta2
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear_dual_out2D(long cudaStream_address,
            long dX_address, 
            float alpha1, float beta1,
            float alpha2, float beta2,
            long dY1_address, long dY2_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2D: char2float">
    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear2D_char2float(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear2D_float2char(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2D: int2float">
    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear2D_int2float(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear2D_float2int(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_dual2D: element/row_field">
    /**
     * <pre>
     * Linear Transformation: Y = alpha*X1 + beta*X2 + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear_dual2D(long cudaStream_address,
            long dX1_address,
            long dX2_address,
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] = alpha*X1[i] + beta*X2 + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha
     * @param dY_address
     * @param beta
     * @param lengthv
     * @param gamma
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void linear_dual2D_row(long cudaStream_address,
            long dX1_address,
            long dX2_address, int row_lengthv,//row_lengthv = X2_lengthv
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[X2.length, length/X2.length]
     * (6) X1[i], Y[i] is the ith filed vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] =alpha*X1[i] + beta*X2 + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.089000, Speed = 131.671341 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param alpha
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void linear_dual2D_field(long cudaStream_address,
            long dX1_address,
            long dX2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int width, int stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2D">
    /**
     * <pre>
     * Y = alpha*X^2 + beta*X + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.137000, Speed = 57.025551 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void quadratic2D(long cudaStream_address,
            long dX_address,
            float alpha, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Y = alpha*X^2 + beta*X + gamma
     * (2) Y'= 2*alpha*X + beta
     * (3) deltaX = deltaY <*> Y'.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.275000, Speed = 71.022720 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void quadratic2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address, float alpha, float beta,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic_dual2D">
    /**
     * <pre>
     * Y = k11*X1^2 + X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void quadratic_dual2D(long cudaStream_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = k11*X1^2 + X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C
     * (1) dY / dX1 = 2*k11*X1 + k12*X2 + k1
     * (2) dY / dX2 = 2*k22*X2 + k12*X1 + k2
     * (3) deltaX1 = (dY / dX1) * deltaY
     * (4) deltaX2 = (dY / dX2) * deltaY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX1_address
     * @param d_deltaX2_address
     * @param d_deltaY_address
     * @param dX1_address
     * @param dX2_address
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void quadratic_dual2D_deltaX(long cudaStream_address,
            long d_deltaX1_address, long d_deltaX2_address,
            long d_deltaY_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_dual2D_row / field">
    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[length/X2.length, X2.length]
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] = k11*X1[i]*X1[i] + k12*X1[i]*X2 + k22*X2*X2 + k1*X1[i] + k2*X2 + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 106.534081 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param X2_lengthv
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void quadratic_dual2D_row(long cudaStream_address,
            long dX1_address,
            long dX2_address, int X2_lengthv,//row_lengthv = X2_lengthv
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[X2.length, length/X2.length]
     * (6) X1[i], Y[i] is the ith filed vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] = k11*X1[i]*X1[i] + k12*X1[i]*X2 + k22*X2*X2 + k1*X1[i] + k2*X2 + C.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.089000, Speed = 131.671341 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param row_lengthv
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void quadratic_dual2D_field(long cudaStream_address,
            long dX1_address,
            long dX2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            long dY_address,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="variance2D_f64">
    /**
     * <pre>
     * Linear Transformation: Y = alpha * X + beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param stream_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void variance2D_f64(long stream_address,
            long dX_mean_address,
            long dX_sqmean_address,
            long dX_var_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    

    //<editor-fold defaultstate="collapsed" desc="BP: rpl2D">
    /**
     * <pre>
     * Reciprocal: Y = alpha / (X + beta) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) rpl(1/beta + gamma) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.119000, Speed = 65.651253 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void rpl2D(long cudaStream_address,
            float alpha, long dX_address, float beta, float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Reciprocal: Y = alpha / (X + beta) + gamma
     * Y' = - alpha / (X + beta)^2
     * As: alpha / (X + beta) + gamma = Y
     * we have: 1/(X + beta) = (1 / alpha) * (Y - gamma)
     * So: Y' = -(1 / alpha) * (Y - gamma)^2
     * deltaX = deltaY <*> Y' =  deltaX <*> (1 - Y^2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.172000, Speed = 68.132263 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param gamma
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void rpl2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha, float gamma,
            int lengthv, int width, int stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div2D, (alpha*X1 + beta1) / (alpha2*X2 + beta2)">
    /**
     * <pre>
     * Y = (alpha1*X1 + beta1)/(alpha2*X2 + beta2) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha2 || beta2 !=0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.112000, Speed = 69.754456 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha1
     * @param dX1_address
     * @param beta1
     * @param alpha2
     * @param dX2_address
     * @param beta2
     * @param gamma
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void div2D(long cudaStream_address,
            float alpha1, long dX1_address, float beta1,
            float alpha2, long dX2_address, float beta2,
            float gamma,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Y = (alpha1*X1 + beta1)/(alpha2*X2 + beta2) + gamma,
     * (2) dY / dX1 = a1 / (a2*X2 + b2)
     * (3) dY / dX2 = -a2 * (a1*X1 + b1)/{(a2*X2 + b2)^2}
     * (4) deltaX1 = (dY / dX1) * deltaY
     * (5) deltaX2 = (dY / dX2) * deltaY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha2 || beta2 !=0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.280000, Speed = 69.754463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX1_address
     * @param d_deltaX2_address
     * @param d_deltaY_address
     * @param dX1_address
     * @param alpha1
     * @param beta1
     * @param dX2_address
     * @param alpha2
     * @param beta2
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void div2D_deltaX(long cudaStream_address,
            long d_deltaX1_address, long d_deltaX2_address,
            long d_deltaY_address,
            long dX1_address, float alpha1, float beta1,
            long dX2_address, float alpha2, float beta2,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D: row/field">
    /**
     * <pre>
     * (1) X1.width = X2.width = Y.height
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) reshape: X1, Y -> 2D Tensor[height, row_lengthv]
     * (4) reshape: X2 -> 1D Tensor[height];
     * (6) X1[i], Y[i] is the ith row vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] = (alpha1*X1[i] + beta1) / (alpha2*X2 + beta2) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha2 || beta2 !=0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.101000, Speed = 116.027222 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha1
     * @param dX1_address
     * @param beta1
     * @param alpha2
     * @param dX2_address
     * @param beta2
     * @param gamma
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void div2D_row(long cudaStream_address,
            float alpha1, long dX1_address, float beta1,
            float alpha2, long dX2_address, float beta2,
            float gamma, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) X1.width = X2.width = Y.width
     * (2) X1.[length, lengthv] = Y.[length, lengthv] = [length, lengthv]
     * (3) [length, lengthv] % X2.[length, lengthv] == 0
     * (4) reshape: Xrow2 -> Xrow2[Xrow2.length]
     * (5) reshape: X1, Y -> (X1, Y)[X2.length, length/X2.length]
     * (6) X1[i], Y[i] is the ith field vector of X1, Y:
     *      for i from 1 to X2_length:
     *          Y[i] = (alpha1*X1[i] + beta1) / (alpha2*X2 + beta2) + gamma.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha2 || beta2 !=0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.101000, Speed = 116.027222 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha1
     * @param dX1_address
     * @param beta1
     * @param alpha2
     * @param dX2_address
     * @param beta2
     * @param gamma
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void div2D_field(long cudaStream_address,
            float alpha1, long dX1_address, float beta1,
            float alpha2, long dX2_address, float beta2,
            float gamma, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="add_div2D_row\field">
    /**
     * <pre>
     * (1) reshape: Y, X1  -> 2D Tensor[height, row_lengthv]
     * (2) reshape: X2, X3 -> 2D Tensor[row_lengthv]
     * (3) for each row vector of Y:
     *      Y[i] = (alpha*X1[i] + beta*X2[i] + gamma) / (X3[i] + delta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.101000, Speed = 116.027222 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param dX3_address
     * @param gamma
     * @param row_lengthv
     * @param alpha
     * @param dY_address
     * @param beta
     * @param delta
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void add_div2D_row(long cudaStream_address,
            long dX1_address,
            long dX2_address,
            long dX3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) reshape: Y, X1  -> 2D Tensor[height, row_lengthv]
     * (2) reshape: X2, X3 -> 2D Tensor[height = field_length]
     * (3) for each field vector of Y:
     *      Y[i] = (alpha*X1[i] + beta*X2[i] + gamma) / (X3[i] + delta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.101000, Speed = 116.027222 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param dX3_address
     * @param gamma
     * @param row_lengthv
     * @param alpha
     * @param dY_address
     * @param beta
     * @param delta
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void add_div2D_field(long cudaStream_address,
            long dX1_address,
            long dX2_address,
            long dX3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, zero_nan, sqrt"> 
    /**
     * <pre>
     * Y = sign(alpha*X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) sign(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.111000, Speed = 70.382881 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sign2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * ceiling, up to the whole:
     *      Y = ceiling(alpha*X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) ceiling(beta) != 0
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void ceil2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * floor, down to the whole:
     *      Y = floor(alpha*X + beta), floor(beta) != 0.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) floor(beta) != 0
     *
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.110000, Speed = 71.022720 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void floor2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    //<editor-fold defaultstate="collapsed" desc="BP: abs">
    /**
     * <pre>
     * Y = abs(alpha * X + beta) = |alpha * X + beta|.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) abs(beta) != 0
     *
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void abs2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = abs(alpha * X + beta) = |alpha * X + beta|
     * Y'= alpha * sign(alpha*X + beta)
     * deltaX = deltaY <*> Y' =  deltaX <*> (1 - Y^2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.171000, Speed = 68.530701 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void abs2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address, float alpha, float beta,
            int lengthv, int width, int stride);
    //</editor-fold>

    /**
     * <pre>
     * Y = X * !isNan(X).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) abs(beta) != 0
     *
     *  ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.175000, Speed = 66.964279 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void zero_nan2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = sqrt(alpha*X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) sign(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.111000, Speed = 70.382881 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqrt2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = sqrt(k11*X1^2 + X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX1_address
     * @param dX2_address
     * @param k11
     * @param k12
     * @param k22
     * @param k1
     * @param k2
     * @param C
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqrt_quadratic_dual2D(long cudaStream_address,
            long dX1_address, long dX2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="min, max, clip"> 
    //<editor-fold defaultstate="collapsed" desc="min, min_dual">
    /**
     * <pre>
     * Y = min(alpha*X + beta, vmin).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) min(beta, vmin) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param vmin
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void min2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            float vmin,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = min(alpha1*X1 + beta1, alpha2*X2 + beta2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) min(beta, vmin) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha1
     * @param dX1_address
     * @param beta1
     * @param alpha2
     * @param dX2_address
     * @param beta2
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void min_dual2D(long cudaStream_address,
            float alpha1, long dX1_address, float beta1,
            float alpha2, long dX2_address, float beta2,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max, max_dual">
    /**
     * <pre>
     * Y = max(alpha*X + beta, V).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) max(beta, vmax) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param vmax
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void max2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            float vmax,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = min(alpha1*X1 + beta1, alpha2*X2 + beta2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) min(beta, vmin) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha1
     * @param dX1_address
     * @param beta1
     * @param alpha2
     * @param dX2_address
     * @param beta2
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void max_dual2D(long cudaStream_address,
            float alpha1, long dX1_address, float beta1,
            float alpha2, long dX2_address, float beta2,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    /**
     * <pre>
     * Clip: Y = clip(alpha*X + beta, vmax, vmin)
     *  if alpha*X + beta > Vmax: Y = vmax;
     *  else if alpha*X + beta > Vmin: Y = alpha*X + beta;
     *  else: Y = vmin.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) clip(beta, vmin, max) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.109000, Speed = 71.674309 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param vmin
     * @param vmax
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void clip2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            float vmin, float vmax,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    //<editor-fold defaultstate="collapsed" desc="expoential">
    /**
     * <pre>
     * Y = exp(alpha*X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void exp2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logarithm">
    /**
     * <pre>
     * Y = log(alpha*X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void log2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1)Y = log(alpha*X + beta);
     * (2) Y' = alpha / (alpha*X + beta), As: exp(Y) = alpha*X + beta
     *     = alpha * exp(-Y)
     * (3) gradient: deltaX = deltaY <*> deriY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void log2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    /**
     * <pre>
     * Y = relu(X) = max(X, 0).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.106000, Speed = 73.702827 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void relu2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = max(X, 0)
     * if X > 0: Y = X, Y' = 1, Y > 0
     * else: Y = 0, Y' = 0,  0 >= y
     * Y' = (X > 0) = sign(Y) = Y> 0.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void relu2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = max(X, 0)
     * if X > 0: Y = X, Y' = 1, Y > 0
     * else: Y = 0, Y' = 0,  0 >= y
     * Y' = (X > 0) = sign(Y) = Y> 0.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void relu2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    /**
     * <pre>
     * Y = leakyRelu(X, k)
     * if X > 0: Y = X
     * else: Y = k * X.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) k belongs to [0, 1)
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param k
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void leakyRelu2D(long cudaStream_address,
            long dX_address, float k,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y' =  (Y > 0) + (0 >= Y) * k = 1 + (0 >= Y)*(k - 1)
     * deltaX = deltaY <*> Y' =  deltaX <*> {1 + (0 >= Y)*(k - 1)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) k belongs to [0, 1)
     * (5) V1: holdY(), Y is not changed
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.140000, Speed = 83.705353 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param k
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void leakyRelu2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float k,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * Y' =  (Y > 0) + (0 >= Y) * k = 1 + (0 >= Y)*(k - 1)
     * deltaX = deltaY <*> Y' =  deltaX <*> {1 + (0 >= Y)*(k - 1)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) k belongs to [0, 1)
     * (5) V2: holdX(), X is not changed
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.140000, Speed = 83.705353 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param k
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void leakyRelu2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address, float k,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    /**
     * <pre>
     * Y = alpha * elu(X, k)
     * if X > 0: Y = alpha * X
     * else: Y = alpha * k *(e^X - 1).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha >= 0
     * (5) k belongs to [0, 1)
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.111000, Speed = 70.382881 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param k
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void elu2D(long cudaStream_address,
            long dX_address, float alpha, float k,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y >  0: Y = alpha * X          , Y' = alpha
     * 0 >= Y: Y = alpha * k * (e^X - 1), Y' = alpha * k * e^X =  Y + alpha*k
     * Y' = (Y > 0)*alpha + (0 >= Y)*(Y + alpha*k)
     *    = alpha + (0 >= Y)*(Y + alpha*k - alpha)
     * let: beta = alpha*k - alpha
     * Y' = alpha + (0 >= Y)*(Y + beta)
     * deltaX = deltaY <*> Y' =  deltaX <*> {alpha + (0 >= Y)*(Y + beta)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha >= 0
     * (5) k belongs to [0, 1)
     * (6) holdY(): Y is not changed
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.140000, Speed = 83.705353 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param k
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void elu2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha, float k,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * Y >  0: Y = alpha * X          , Y' = alpha
     * 0 >= Y: Y = alpha * k * (e^X - 1), Y' = alpha * k * e^X =  Y + alpha*k
     * Y' = (Y > 0)*alpha + (0 >= Y)*(Y + alpha*k)
     *    = alpha + (0 >= Y)*(Y + alpha*k - alpha)
     * let: beta = alpha*k - alpha
     * Y' = alpha + (0 >= Y)*(Y + beta)
     * deltaX = deltaY <*> Y' =  deltaX <*> {alpha + (0 >= Y)*(Y + beta)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) alpha >= 0
     * (5) k belongs to [0, 1)
     * (6) holdX(): X is not changed
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.140000, Speed = 83.705353 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param alpha
     * @param k
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void elu2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address, float alpha, float k,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    /**
     * <pre>
     * Y = log(1 + e^X) = log1p(e^X).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.117000, Speed = 66.773506 GB/
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void softPlus2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = log(1 + e^X) = log1p(e^X)
     * Y' = 1 - 1/(1 + e^X)
     * Y' = 1 - e^(-Y)
     * deltaX = deltaY <*> Y' =  deltaX <*> {1 - e^(-Y)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) holdY(): Y is not changed
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void softPlus2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    
     /**
     * <pre>
     * Y = log(1 + e^X) = log1p(e^X)
     * Y' = 1 - 1/(1 + e^X)
     * Y' = 1 - e^(-Y)
     * deltaX = deltaY <*> Y' =  deltaX <*> {1 - e^(-Y)}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) holdX(): X is not changed
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void softPlus2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    /**
     * <pre>
     * Y = tanh(X)
     *   = (e^X - e^(-X))/(e^X + e^(-X))
     *   = 1 - 2/(e^(2*X) + 1).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.109000, Speed = 71.674309 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void tanh2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = 1 - 2/(e^(2*X) + 1)
     * Y'= 1 - Y^2
     * deltaX = deltaY <*> Y' =  deltaX <*> (1 - Y^2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.153000, Speed = 76.593140 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void tanh2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            int lengthv, int width, int stride);
    
     /**
     * <pre>
     * Y = 1 - 2/(e^(2*X) + 1)
     * Y'= 1 - Y^2
     * deltaX = deltaY <*> Y' =  deltaX <*> (1 - Y^2).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.153000, Speed = 76.593140 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param lengthv
     * @param width
     * @param stride
     */
    @Passed
    public static native void tanh2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    /**
     * <pre>
     * Y = sigmoid(X)
     *   = 1/(1 + e^(-X))
     *   = e^X / (1 + e^X).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.113000, Speed = 69.137169 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sigmoid2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Y = 1/(1 + e^(-X))
     * Y' = Y - Y^2 = Y(1 - Y)
     * deltaX = deltaY <*> Y' =  deltaX <*> (Y(1 - Y)).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.153000, Speed = 76.593140 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sigmoid2D_deltaX_v1(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    
      /**
     * <pre>
     * Y = 1/(1 + e^(-X))
     * Y' = Y - Y^2 = Y(1 - Y)
     * deltaX = deltaY <*> Y' =  deltaX <*> (Y(1 - Y)).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     * 
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.153000, Speed = 76.593140 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sigmoid2D_deltaX_v2(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    /**
     * <pre>
     * (1) Y = softmax(X)
     * (2) deltaX = deltaY<*>Y'
     * (3) deltaY_Y_row_sum = sum_each_row: deltaY[i] * Y[i].
     *
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param d_deltaY_Y_rowSum_address
     * @param row_lengthv
     * @param lenghv
     * @param mem_width
     * @param mem_stride
     */
    public static native void softmax2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            long d_deltaY_Y_rowSum_address, int row_lengthv,
            int lenghv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logsoftmax2D">
    /**
     * <pre>
     * (1) Y = log(softmax(X))
     * (2) maxX = maxEachRow: X[i]
     * (3) expXm_max_rowSum = sumEachRow: exp(X[i] - maxX).
     *
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param d_maxX_address
     * @param d_expXm_max_rowSum_address
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void logsoftmax2D(long cudaStream_address,
            long dX_address,
            long d_maxX_address,
            long d_expXm_max_rowSum_address, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Y = softmax(X)
     * (2) deltaX = deltaY<*>Y'
     * (3) Y_row_sum = sum_each_row: Y[i].
     *
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param d_deltaY_rowSum_address
     * @param row_lengthv
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void logsoftmax2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address,
            long d_deltaY_rowSum_address, int row_lengthv,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin">
    /**
     * <pre>
     * Sine: Y = sin(alpha * X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.173000, Speed = 67.738434 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sin2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Sine: Y = sin(alpha * X + beta)
     * (2) Y' = alpha * cos(alpha * X + beta)
     * (3) gradient: deltaX = deltaY <*> deriY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sin2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dX_address, float alpha, float beta,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="BP: tan">
    /**
     * <pre>
     * Tangent: Y = tan(alpha * X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.115000, Speed = 67.934776 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void tan2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Tangent: Y = tan(alpha*X + beta)
     * (2) Y' = alpha * (1 + tan^2(alpha*X + beta))
     *       = alpha * (1 + y^2)
     *       = alpha * y^2 + alpha
     * (3) gradient: deltaX = deltaY <*> deriY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void tan2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin">
    /**
     * <pre>
     * A half of Sine: Y  = Amp * {2*|sin(alpha*X + beta)| - 1}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     * (5) default Amp = 1.
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.114000, Speed = 68.530701 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param Amp
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void halfSin2D(long cudaStream_address,
            float Amp, float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Y = Amp * {2*|sin(alpha*X + beta)| - 1}
     *     Y = 2*Amp * {|sin(alpha*X + beta)| - 0.5}
     *
     * (2) Y' = 2*Amp * |alpha * cos(alpha*X + beta)|
     *     Y' = 2*Amp*|alpha| * |cos(alpha*X + beta)|
     *
     * As: Y =  2*Amp * {|sin(alpha*X + beta)| - 0.5}
     * So: Y / (2*Amp) + 0.5 = |sin(alpha*X + beta)|
     * So: |cos(alpha*X + beta)| = sqrt(1 - [Y / (2*Amp) + 0.5 ]^2)
     * Y' = 2*Amp*|alpha| * sqrt(1 - [Y / (2*Amp) + 0.5 ]^2)
     * (3) deltaX = deltaY <*> Y'.
     *
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.108000, Speed = 72.337959 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param lengthv
     * @param d_deltaY_address
     * @param mem_width
     * @param alpha
     * @param dY_address
     * @param mem_stride
     * @param Amp
     */
    @Passed
    public static native void halfSin2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float Amp, float alpha,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin">
    /**
     * <pre>
     * Arcsin: Y = arcsin(alpha * X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.173000, Speed = 67.738434 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void arcsin2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Arcsin: Y = arcsin(alpha * X + beta)
     * (2) deriY = alpha / sqrt(1 - (alpha*X + beta)^2)
     *        = alpha / fabsf(cos(Y))
     * (3) gradient: deltaX = deltaY <*> deriY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void arcsin2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="BP: arctan">
    /**
     * <pre>
     * Arctan: Y = arctan(alpha * X + beta).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.173000, Speed = 67.738434 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param alpha
     * @param dX_address
     * @param beta
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void arctan2D(long cudaStream_address,
            float alpha, long dX_address, float beta,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Sine: Y = sin(alpha * X + beta)
     * (2) deriY = alpha / (1 + (alpha*X + beta)^2)
     *       = alpha / (1 + tan^(Y))
     *       = alpha * cos^2(Y)
     * (3) gradient: deltaX = deltaY <*> deriY.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.150000, Speed = 78.125000 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param d_deltaX_address
     * @param d_deltaY_address
     * @param dY_address
     * @param alpha
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void arctan2D_deltaX(long cudaStream_address,
            long d_deltaX_address,
            long d_deltaY_address,
            long dY_address, float alpha,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold> 
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="distance & loss function">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    /**
     * <pre>
     * L1-Loss for each element: L = |Y - Yh|.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void L1_2D(long cudaStream_address,
            long dY_address, long dYh_address,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * L1-loss Gradient of Yh:
     * L = |Y - Yh|
     * deltaYh = dL / dYh = d|Y - Yh| / dYh
     *         = sign(Y - Yh).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param d_deltaYh_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void L1_2D_deltaYh(long cudaStream_address,
            long dY_address, long dYh_address,
            long d_deltaYh_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    /**
     * <pre>
     * L2 - Loss: L = (Y - Yh)^2.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void L2_2D(long cudaStream_address,
            long dY_address, long dYh_address,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * L2 - Loss Gradient of Yh:
     * L = 2*(Y - Yh)^2
     * deltaYh = dY/dYh = Yh - Y.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param d_deltaYh_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void L2_2D_deltaYh(long cudaStream_address,
            long dY_address, long dYh_address,
            long d_deltaYh_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: smoothL1">
    /**
     * <pre>
     * L = smoothL1Loss(Yh, Y)
     * if (div = |Yh - Y|) > 1:
     *     L = L1Loss(Yh, Y) - 0.5 = |Yh - Y| - 0.5
     *  else:
     *     L = L2Loss(Yh, Y) = 0.5 * (Yh - Y)^2
     * Improved:
     * div = |Yh - Y|
     * if div > 1: Z = div - 0.5
     * else: L = 0.5*div*dv
     * L = (div>1)*(div - 0.5) + (1>=div)*(0.5*div*div)
     * L = (div - 0.5) + (1>=div)*(0.5*div*div + 0.5 -div).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.220000, Speed = 53.267040 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void smoothL1_2D(long cudaStream_address,
            long dY_address, long dYh_address,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * find the gradient of deltaYh:
     * div = |Yh - Y|
     * if div>1:
     *      deltaYh = d"{|Yh - Y| - 0.5}" / dYh = sign(Yh - Y)
     * else:
     *      deltaYh = d"{0.5*(Yh - Y)^2}" / dYh = Yh - Y
     * deltaYh = (div>1)*sign(Yh - Y) + (1>=div)*(Yh - Y)
     *          = (div>1)*sign(Yh - Y) + (1>=div)*sign(Yh - Y)*div
     *          = sign(Yh - Y)*{(div>1 + (1>=div)*div}
     *          = sign(Yh - Y)*{(1>=div)*(div - 1) + 1}.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param d_deltaYh_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void smoothL1_2D_deltaYh(long cudaStream_address,
            long dY_address, long dYh_address,
            long d_deltaYh_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    /**
     * <pre>
     * Binary Cross Entropy: L = -alpha*y*log(yh) + beta*(y - 1)*log(1 - yh).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param alpha
     * @param beta
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void binaryCrossEntropy2D(long cudaStream_address,
            long dY_address, long dYh_address,
            float alpha, float beta,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Find the gradient of deltaYh of Binary Cross Entropy:
     *      L = -alpha*y*log(yh) + beta*(y - 1)*log(1 - yh)
     *      deltaYh = -alpha*y/yh + beta*(y - 1)/(yh - 1).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param alpha
     * @param beta
     * @param deltaYh_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void binaryCrossEntropy2D_deltaYh(long cudaStream_address,
            long dY_address, long dYh_address,
            float alpha, float beta,
            long deltaYh_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    /**
     * <pre>
     * (1) Sigmoid: Yh = 1 / (1 + exp(-X))
     * (2) Binary Cross Entropy: L = -alpha*y*log(yh) + beta*(y - 1)*log(1 - yh)
     * (3) L = -alpha * Y * X + {(alpha - beta)*Y + beta} * log(1 + exp(X))
     * (4) when alpha = beta = 1:  -Y * X + log[1 + exp(X)].
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sigmoid_binaryCrossEntropy2D(long cudaStream_address,
            long dY_address, long dX_address,
            float alpha, float beta,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) Binary Cross Entropy: deltaYh = -alpha*Y/Yh + beta*(1 - Y)/(1 - Yh)
     * (2) Sigmoid: deltaX = deltaY * Yh * (1 - Yh)
     * (3) deltaX = Yh * {(alpha - beta)*Y + beta} - alpha*Y
     *        = {(alpha - beta)*Y + beta} /(1 + exp(-X)) - alpha*Y
     * (4) when alpha = beta = 1: deltaX = Yh - Y.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sigmoid_binaryCrossEntropy2D_deltaX(long cudaStream_address,
            long dY_address, long dX_address,
            float alpha, float beta,
            long deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    /**
     * <pre>
     * Cross Entropy: L = -y*log(yh)
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param dL_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void crossEntropy2D(long cudaStream_address,
            long dY_address, long dYh_address,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Find the gradient of deltaYh in Cross Entropy:
     *      L = -y*log(yh) + (y - 1)*log(1 - yh)
     *      deltaYh = -y/yh + (y-1)/(yh-1).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dYh_address
     * @param deltaYh_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    public static native void crossEntropy2D_deltaYh(long cudaStream_address,
            long dY_address, long dYh_address,
            long deltaYh_address,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax_crossEntropy">
    /**
     * <pre>
     * (1) Softmax: Yh = exp(X) / sum(exp(X))
     * (2) Cross Entropy: L = -y*log(yh) + (y - 1)*log(1 - yh)
     * (3) We have: L = Y*(X - M) + log(U) + (Y - 1) * log(U - e^(X-M))
     * [1] M = maxEachRow(X), M.length = X.height
     * [2] U = sum(exp(X - M))
     * (4) L = -Y*(X - maxX) + log(expXsm_rowSum) + (Y - 1)*log(expXsmax_rowSum - exp(X-M)).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param dX_address
     * @param d_maxX_address
     * @param d_expXm_max_rowSum_address//sum(exp(X - maxX)) for each row
     * @param dL_address
     * @param row_lengthv
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void softmax_crossEntropy2D(long cudaStream_address,
            long dY_address, long dX_address,
            long d_maxX_address,
            long d_expXm_max_rowSum_address, int row_lengthv,
            long dL_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * (1) CrossEntropy: deltaYh = -Y/Yh + (1 - Y)/(1 - Yh)
     * (2) Sigmoid: deltaX = deltaY * Yh * (1 - Yh)
     * (3) deltaX = -Y + softmax(X) * sum_row(Y[i]).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.210000, Speed = 55.803574 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dY_address
     * @param X_address
     * @param d_maxX
     * @param d_expXm_max_rowSum expXm_max_rowSum = sumEachRow: exp(X - maxX)
     * @param Y_rowSum_address
     * @param d_deltaX_address
     * @param row_lengthv
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void softmax_crossEntropy2D_deltaX(long cudaStream_address,
            long dY_address, long X_address,
            long d_maxX, long d_expXm_max_rowSum,
            long Y_rowSum_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    /**
     * <pre>
     * Momentum(W)
     * Init: U = alpha,
     * Iterate:
     *      a1 = alpha, a2 = 1-alpha;
     *      (1) lr_t = lr / (1 - U);   # correct the learning rate
     *      (2) V = a1*V + a2*deltaW           # update the velocity depending on the acceleration
     *      (3) W = W - lr_t*V                 # gradient descent
     *       U *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243000, Speed = 80.375511 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param lr_t the corrected learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void momentum2D(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long d_deltaW_address, float lr_t,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Momentum(W)
     * Init: U = alpha,
     * Iterate:
     *      a1 = alpha, a2 = 1-alpha;
     *      (1) lr_t = lr / (1 - U);           # correct the learning rate
     *      (3) deltaW = deltaW + L1coef * sign(W) + L2coef * W
     *      (4) V = a1*V + a2*deltaW           # update the velocity depending on the acceleration
     *      (5) W = W - lr_t*V                 # gradient descent
     *       U *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243000, Speed = 80.375511 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param lr_t the corrected learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void momentum2D_decay(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long d_deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    /**
     * <pre>
     * SGD with momentum and nestrov
     * (1) V = momentum*V + (1 - dampening) * deltaW
     * (2) K = nesterov * momentum + 1 - nesterov
     * (3) step = nesterov * deltaW + K * V
     * (4) W = W - lr*step
     * .
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243000, Speed = 80.375511 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param momentum
     * @param nesterov
     * @param dampen
     * @param lr the corrected learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sgdmn2D(long cudaStream_address,
            long dW_address,
            long dV_address, float momentum, float dampen, float nesterov,
            long d_deltaW_address, float lr,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * SGD with momentum and nestrov
     * (1) deltaW += L1coef * SIGN(deltW) + L2coef * deltW
     * (2) V = momentum*V + (1 - dampening) * deltaW
     * (3) K = nesterov * momentum + 1 - nesterov
     * (4) step = nesterov * deltaW + K * V
     * (5) W = W - lr*step
     * .
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243000, Speed = 80.375511 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param momentum
     * @param nesterov
     * @param dampen
     * @param lr the corrected learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sgdmn2D_decay(long cudaStream_address,
            long dW_address,
            long dV_address, float momentum, float dampen, float nesterov,
            long d_deltaW_address, float lr,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    /**
     * <pre>
     * RMSprop(W)
     * Init: U = alpha
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      (1) lr_t = lr * sqrt(1 - U)      # correct the learning rate
     *      (2) eps_t = lr * sqrt(1 - U)
     *      (3) S = a1*S + a2*deltaW^2               # update the standard derivation depending on the acceleration
     *      (4) W = W - lr_t * deltaW/(sqrt(S) + eps_t)  # gradient descent
     *       U *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243667, Speed = 80.155609 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dS_address
     * @param a1
     * @param a2
     * @param eps_t
     * @param lr_t the corrected learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void rmsprop2D(long cudaStream_address,
            long dW_address,
            long dS_address, float a1, float a2, float eps_t,
            long d_deltaW_address, float lr_t,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * RMSprop(W)
     * Init: U = alpha
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      (1) lr_t = lr * sqrt(1 - U)      # correct the learning rate
     *      (2) eps_t = lr * sqrt(1 - U)
     *      (3) deltaW = deltaW + L1coef * sign(W) + L2coef * W
     *      (4) S = a1*S + a2*deltaW^2               # update the standard derivation depending on the acceleration
     *      (5) W = W - lr_t * deltaW/(sqrt(S) + eps_t)  # gradient descent
     *       U *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.243667, Speed = 80.155609 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param dS_address
     * @param a1
     * @param a2
     * @param eps_t
     * @param d_deltaW_address
     * @param lr_t the corrected learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void rmsprop2D_decay(long cudaStream_address,
            long dW_address,
            long dS_address, float a1, float a2, float eps_t,
            long d_deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>  

    //<editor-fold defaultstate="collapsed" desc="Adam">
    /**
     * <pre>
     * Adam(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      (1) lr_t = lr / (1 - Uv) * sqrt(1 - Us) # correct the learning rate
     *      (2) eps_t = eps * sqrt(1 - Us)
     *      (2) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (3) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (4) W = W - lr_t*V/(sqrt(S) + eps_t)  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param eps_t
     * @param lr_t the learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adam2D(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float b2, float eps_t,
            long d_deltaW_address, float lr_t,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Adam(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      (1) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (2) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (3) Vcorrect = V / (1 - Uv)
     *      (4) Scorrect = S / (1 - Us)
     *      (4) W = W - lr * Vcorrect/(sqrt(Scorrect) + eps)  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param Uv
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param eps
     * @param Us
     * @param lr the learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adam2D_type2(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2, float Uv,
            long dS_address, float b1, float b2, float eps, float Us,
            long d_deltaW_address, float lr,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Adam(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      deltaW += L1coef*L1(W) + L2coef*L2(W)
     *      (1) lr_t = lr / (1 - Uv) * sqrt(1 - Us) # correct the learning rate
     *      (2) deltaW = deltaW + L1coef * sign(W) + L2coef * W
     *      (3) eps_t = eps * sqrt(1 - Us)
     *      (4) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (5) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (6) W = W - lr_t*V/(sqrt(S) + eps_t)  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param eps_t
     * @param lr_t the learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adam2D_decay(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float b2, float eps_t,
            long d_deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    /**
     * <pre>
     * Adamax(W):
     * Init: Uv = alpha
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta
     *      (1) lr_t = lr / (1 - Uv)           # correct the learning rate
     *      (2) V = a1*V + a2*deltaW           # update the velocity depending on the acceleration
     *      (3) S = fmaxf(b1*S, |deltaW|)      # update the standard derivation depending on the acceleration
     *      (4) W = W - lr_t * V/(S + eps)     # gradient descent
     *      Uv *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param eps
     * @param lr_t the learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adamax2D(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float eps,
            long d_deltaW_address, float lr_t,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Adamax(W):
     * Init: Uv = alpha
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta
     *      deltaW += L1coef*L1(W) + L2coef*L2(W)
     *      (1) lr_t = lr / (1 - Uv)           # correct the learning rate
     *      (2) deltaW = deltaW + L1coef * sign(W) + L2coef * W
     *      (3) V = a1*V + a2*deltaW           # update the velocity depending on the acceleration
     *      (4) S = fmaxf(b1*S, |deltaW|)      # update the standard derivation depending on the acceleration
     *      (5) W = W - lr_t * V/(S + eps)     # gradient descent
     *      Uv *= alpha.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param eps
     * @param lr_t the learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adamax2D_decay(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float eps,
            long d_deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    /**
     * <pre>
     * AdamW(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      (1) lr_t = lr / (1 - Uv) * sqrt(1 - Us) # correct the learning rate
     *      (2) eps_t = eps * sqrt(1 - Us)
     *      (3) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (4) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (5) W = W - lr_t*V/(sqrt(S) + eps_t) - lr*(L1coef * sign(W) + L2coef * W)  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param eps_t
     * @param lr_t the learning rate
     * @param lr
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adamW2D(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float b2, float eps_t,
            long d_deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);

    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    /**
     * <pre>
     * Adamod(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      (1) lr_t  = lr  * sqrt(1 - Us)
     *      (2) eps_t = eps * sqrt(1 - Us)
     *      (3) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (4) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (5) neta = lr_t / (sqrt(S) + eps_t)
     *      (6) G = c1*G + c2*neta
     *      (7) W -= min(neta, G) * {V / (1 - Uv)}  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param dG_address
     * @param c1
     * @param c2
     * @param eps_t
     * @param lr_t the learning rate
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adamod2D(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float b2, float eps_t,
            long dG_address, float c1, float c2,
            long d_deltaW_address, float lr_t,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * Adamod(W):
     * Init: Uv = alpha, Us = beta
     * Iterate:
     *      a1 = alpha, a2 = 1 - alpha
     *      b1 = beta, b2 = 1 - beta
     *      deltaW += L1coef*L1(W) + L2coef*L2(W)
     *      (1) lr_t  = lr  * sqrt(1 - Us)
     *      (2) eps_t = eps * sqrt(1 - Us)
     *      (3) V = a1*V + a2*deltaW          # update the velocity depending on the acceleration
     *      (4) S = b1*S + b2*deltaW^2        # update the standard derivation depending on the acceleration
     *      (5) neta = lr_t / (sqrt(S) + eps_t)
     *      (6) G = c1*G + c2*neta
     *      (7) W -= min(neta, G) * {V / (1 - Uv)}  # gradient descent
     *      Uv *= alpha; Us *= beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) linear(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.350667, Speed = 77.976463 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dW_address
     * @param d_deltaW_address
     * @param dV_address
     * @param a1
     * @param Uv
     * @param a2
     * @param dS_address
     * @param b1
     * @param b2
     * @param dG_address
     * @param c1
     * @param c2
     * @param eps_t
     * @param lr_t the learning rate
     * @param L1coef
     * @param L2coef
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void adamod2D_decay(long cudaStream_address,
            long dW_address,
            long dV_address, float a1, float a2,
            long dS_address, float b1, float b2, float eps_t,
            long dG_address, float c1, float c2,
            long d_deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="affine">
    /**
     * <pre>
     * (1) [A, B] -> Tensor[row_lengthv]
     * (2) [X, Y] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) A.width = B.width = Y.width = X.width
     * (6) for each field: Y = A*X + B.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param stream_address
     * @param dX_address
     * @param dA_address
     * @param dB_address
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void affine2D_row(long stream_address,
            long dX_address,
            long dA_address, long dB_address, int row_lengthv,
            long dY_address,
            int lengthv, int width, int stride);
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm2D_row">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    /**
     * <pre>
     * Square Batch Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [X, Y] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_sqmean.width = Y.width = X.width
     * (6) X_stddev = sqrt(X_sqmean - X_mean^2 + eps)
     * (7) for each field: Y = (X - X_mean) / X_stddev.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) affine = false
     * </pre>
     * @param stream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqBatchNorm2D_row(long stream_address,
            long dX_address,
            long dX_mean_address, 
            long dX_sqmean_address, float eps, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * Square Batch Normalization:
     * (1) [A, B, X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [X, Y] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) A.width = B.width = X_mean.width = X_sqmean.width = Y.width = X.width
     * (6) X_stddev = sqrt(X_sqmean - X_mean^2 + eps)
     * (7) for each field: Y = A + (X - X_mean) / X_stddev + B.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) affine = true
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqBatchNorm_affined2D_row(long cudaStream_address,
            long dX_address,
            long dX_mean_address, 
            long dX_sqmean_address, float eps,
            long dA_address, long dB_address, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    /**
     * <pre>
     * find the gradient of X in Square Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affined = false.
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param d_deltaXp1
     * @param d_deltaXp2
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqBatchNorm2D_row_deltaX_v1(long stream_address,
            long d_deltaY_address,
            long dY_address,
            long dX_mean_address, 
            long dX_sqmean_address, float eps,
            long d_deltaXp1, 
            long d_deltaXp2, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * find the gradient of X in Square Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: HoldX(), X is not changed
     * (5) affined = false.
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param d_deltaXp1
     * @param d_deltaXp2
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqBatchNorm2D_row_deltaX_v2(long stream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_sqmean_address, float eps,
            long d_deltaXp1, 
            long d_deltaXp2, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined)">
    /**
     * <pre>
     * find the gradient of X in Square Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affine = true
     * </pre>
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void sqBatchNorm_affined2D_row_deltaX_v1(long cudaStream_address,
            long d_deltaY_address,
            long dY_address,
            long dX_mean_address, long dX_sqmean_address, float eps,
            long dA_address, long dB_address,
            long d_deltaXp1_address,
            long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * find the gradient of X in Square Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     * (5) affine = true
     * </pre>
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_sqmean_address
     * @param eps
     * @param dA_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride 
     */
    public static native void sqBatchNorm_affined2D_row_deltaX_v2(long cudaStream_address,
            long d_deltaY_address, 
            long dX_address,
            long dX_mean_address, long dX_sqmean_address, float eps,
            long dA_address, 
            long d_deltaXp1_address, 
            long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm2D_row">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    /**
     * <pre>
     * Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) affine = false
     * </pre>
     * @param stream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void batchNorm2D_row(long stream_address,
            long dX_address,
            long dX_mean_address, 
            long dX_var_address, float eps, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) affine = true
     * </pre>
     * @param cudaStream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void batchNorm_affined2D_row(long cudaStream_address,
            long dX_address,
            long dX_mean_address, 
            long dX_var_address, float eps,
            long dA_address, long dB_address, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    /**
     * <pre>
     * find the gradient of X in Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affined = false.
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_var_address
     * @param eps
     * @param d_deltaXp1
     * @param d_deltaXp2
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void batchNorm2D_row_deltaX_v1(long stream_address,
            long d_deltaY_address,
            long dY_address,
            long dX_var_address, float eps,
            long d_deltaXp1, 
            long d_deltaXp2, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * find the gradient of X in Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: HoldX(), X is not changed
     * (5) affined = false.
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param d_deltaXp1
     * @param d_deltaXp2
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void batchNorm2D_row_deltaX_v2(long stream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            long d_deltaXp1, 
            long d_deltaXp2, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined)">
    /**
     * <pre>
     * find the gradient of X in Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affine = true
     * </pre>
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_var_address
     * @param eps
     * @param dA_address
     * @param dB_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void batchNorm_affined2D_row_deltaX_v1(long cudaStream_address,
            long d_deltaY_address,
            long dY_address,
            long dX_var_address, float eps,
            long dA_address, long dB_address,
            long d_deltaXp1_address,
            long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * find the gradient of X in Batch Normalization.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     * (5) affine = true
     * </pre>
     * @param cudaStream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_var_address
     * @param eps
     * @param dA_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride 
     */
    public static native void batchNorm_affined2D_row_deltaX_v2(long cudaStream_address,
            long d_deltaY_address,
            long dX_address,
            long dX_mean_address, long dX_var_address, float eps,
            long dA_address, 
            long d_deltaXp1_address, 
            long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: layernorm2D_row">
    /**
     * <pre>
     * Layer Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[field_length]
     * (2) [X, Y] -> Tensor[field_length, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_square_mean.width = Y.width = X.width
     * (6) X_stddev = sqrt(X_square_mean - X_mean*X_mean)
     * (7) for each row: Y = (X - X_mean) / X_stddev.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param row_lengthv
     * @param eps
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm2D_row(long cudaStream_address,
            long dX_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * find the gradient of deltaX in Batch Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [deltaX, deltaY] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_square_mean.width = deltaY.width = deltaX.width
     * (6) X_stddev = sqrt(X_square_mean - X_mean*X_mean + eps).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affine = false
     * [1] deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
     * [2] deltaXp2[N] = row_sum: deltaY * Y
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm2D_row_deltaX_v1(long stream_address,
            long d_deltaY_address, long dY_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            long d_deltaXp1_address, long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * find the gradient of deltaX in Batch Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [deltaX, deltaY] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_square_mean.width = deltaY.width = deltaX.width
     * (6) X_stddev = sqrt(X_square_mean - X_mean*X_mean + eps).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     * (5) affine = false
     * [1] deltaXp1[N] = row_sum: deltaY * (X*X_mean - X_squareMean - eps)
     * [2] deltaXp2[N] = row_sum: deltaY * (X - X_mean)
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param stream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param eps
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm2D_row_deltaX_v2(long stream_address,
            long d_deltaY_address, long dX_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            long d_deltaXp1_address, long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: layernorm2D_row(affined)">
    /**
     * <pre>
     * Layer Normalization:
     * (1) [A, B, X_mean, X_square_mean] -> Tensor[field_length]
     * (2) [X, Y] -> Tensor[field_length, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) A.width = B.width = X_mean.width = X_square_mean.width = Y.width = X.width
     * (6) X_var = sqrt(X_square_mean - X_mean*X_mean)
     * (7) for each row: Y = A + (X - X_mean)/X_var + B.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param dA_address
     * @param dB_address
     * @param row_lengthv
     * @param eps
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm_affined2D_row(long cudaStream_address,
            long dX_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            long dA_address, long dB_address, int row_lengthv,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);

    /**
     * <pre>
     * find the gradient of deltaX in Batch Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [deltaX, deltaY] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_square_mean.width = deltaY.width = deltaX.width
     * (6) X_stddev = sqrt(X_square_mean - X_mean*X_mean + eps).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V1: holdY(), Y is not changed
     * (5) affine = false
     * [1] deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
     * [2] deltaXp2[N] = row_sum: deltaY * Y
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dY_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param dA_address
     * @param dB_addrxess
     * @param eps
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm_affined2D_row_deltaX_v1(long stream_address,
            long d_deltaY_address, long dY_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            long dA_address, long dB_addrxess,
            long d_deltaXp1_address, long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    
    /**
     * <pre>
     * find the gradient of deltaX in Batch Normalization:
     * (1) [X_mean, X_square_mean] -> Tensor[row_lengthv]
     * (2) [deltaX, deltaY] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (5) X_mean.width = X_square_mean.width = deltaY.width = deltaX.width
     * (6) X_stddev = sqrt(X_square_mean - X_mean*X_mean + eps).
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) V2: holdX(), X is not changed
     * (5) affine = false
     * [1] deltaXp1[N] = row_sum: deltaY * (X*X_mean - X_squareMean - eps)
     * [2] deltaXp2[N] = row_sum: deltaY * (X - X_mean)
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]:
     * </pre>
     * @param stream_address
     * @param d_deltaY_address
     * @param dX_address
     * @param dX_mean_address
     * @param dX_square_mean_address
     * @param d_deltaXp1_address
     * @param d_deltaXp2_address
     * @param dA_address
     * @param eps
     * @param row_lengthv
     * @param d_deltaX_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void layerNorm_affined2D_row_deltaX_v2(long stream_address,
            long d_deltaY_address, long dX_address,
            long dX_mean_address, long dX_square_mean_address, float eps,
            long dA_address,
            long d_deltaXp1_address, long d_deltaXp2_address, int row_lengthv,
            long d_deltaX_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
    /**
     * <pre>
     * (1) [X] -> Tensor[height]
     * (2) [Y] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (6) for each field[i](0 : height):
     *      for each field[j](0 : row_lengthv):
     *          if(j == X[i]) Y[i] = alpha;
     *          else Y[i] = beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void onehot2D_row_int(long cudaStream_address,
            long dX_address,
            float alpha, float beta, int row_lengthv,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     * (1) [X] -> Tensor[height]
     * (2) [Y] -> Tensor[height, row_lengthv]
     * (3) lengthv / row_lengthv = height
     * (4) lengthv % row_lengthv == 0
     * (6) for each field[i](0 : height):
     *      for each field[j](0 : row_lengthv):
     *          if(j == X[i]) Y[i] = alpha;
     *          else Y[i] = beta.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param alpha
     * @param beta
     * @param row_lengthv
     * @param dY_address
     * @param lengthv
     * @param width
     * @param stride
     */
    public static native void onehot2D_row_char(long cudaStream_address,
            long dX_address,
            float alpha, float beta, int row_lengthv,
            long dY_address,
            int lengthv, int width, int stride);

    /**
     * <pre>
     * Y[float] = (X[byte] + 128) / 255.0f.
     * (1) height * stride = lengthv
     * (2) height * width = length
     * (3) stride = (width + 3)/4 * 4
     * (4) sign(beta) != 0
     *
     * ----Performace on CudaFloat32 Engine(GTX 1050)[synchronized]-------------
     * for [height, width] from [1, 1] to(+1, +1) [10, 256]: correct
     * [height, width] = [1024, 1024]: Time = 0.111000, Speed = 70.382881 GB/s
     * </pre>
     *
     * @param cudaStream_address
     * @param dX_address
     * @param dY_address
     * @param lengthv
     * @param mem_width
     * @param mem_stride
     */
    @Passed
    public static native void pix2tensor2D(long cudaStream_address,
            long dX_address,
            long dY_address,
            int lengthv, int mem_width, int mem_stride);
    //</editor-fold>
}

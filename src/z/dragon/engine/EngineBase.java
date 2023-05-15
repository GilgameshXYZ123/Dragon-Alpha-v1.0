/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import z.dragon.engine.Result.IndexedResult;

/**
 * (1) datatype sensitive: 
 * (2) ASYNC
 * (3) no check
 * (4) device sensitive.
 * width: the lastDim of Tensor == Tensor.memWidth(the basic and minimum memory unit)
 * stride: (width + 3)/4*4, memory alignment, == Tensor.memStride
 * @author Gilgamesh
 */
public abstract class EngineBase 
{
    protected final long L_sizeof_datatype;//sizeof_datatype = 1<<L_sizeof_datatype
    protected final String datatype;
    protected final String datatype_int32;//sizeof(int32) = 4
    protected final String datatype_int8;//sizeof(int8) = 1
    
    protected EngineCore core;
    
    public EngineBase(String dataType, long LsizeofDatatype, 
            String dataType_int32,
            String dataType_int8) 
    {
        this.datatype = dataType;
        this.L_sizeof_datatype = LsizeofDatatype;
        this.datatype_int32 = dataType_int32;
        this.datatype_int8 = dataType_int8;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public String dataType() { return datatype; }
    public long LsizeOf_dataType() { return L_sizeof_datatype; }
    public long sizeOf_dataType() { return 1 << L_sizeof_datatype; }
    
    public String dataType_int32() { return datatype_int32; }
    public String dataType_int8()  { return datatype_int8; }
    
    public EngineCore enigneCore() { return core; }
    
    public abstract void clear();
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("\n[dataType, sizeof(dataType)] = [")
                .append(datatype).append(", ")
                .append(1 << L_sizeof_datatype).append(']');
        sb.append("\n[dataType_int32, sizeof(dataType)] = [")
                .append(datatype_int32).append(", ")
                .append(4).append(']');
        sb.append("\n[dataType_int8, sizeof(dataType)] = [")
                .append(datatype_int8).append(", ")
                .append(1).append(']');
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="extra: int32">
    public abstract void get1D_int32(long address, int[] value, int length);
    public abstract void get2D_int32(long address, int[] value, int height, int width, int stride);
    
    public abstract void set1D_int32(long address, int[] value, int length);
    public abstract void set2D_int32(long address, int[] value, int height, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="extra: int8">
    public abstract void get1D_int8(long address, byte[] value, int length);
    public abstract void get2D_int8(long address, byte[] value, int height, int width, int stride);
    
    public abstract void set1D_int8(long address, byte[] value, int length);
    public abstract void set2D_int8(long address, byte[] value, int height, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memroy Opt">
    public abstract long malloc(long memsize);
    public abstract void free(long address);
    public abstract Syncer memset(long address, int value, long memsize);
    public abstract Syncer memcpy(long dst_address, long src_address, long memsize);
    
    public abstract void get1D(long address, float[] value, int length);
    public abstract void get2D(long address, float[] value, int height, int width, int stride);
    
    public abstract Syncer set1D(long address, float value, int length);
    public abstract void set2D(long address, float[] value, int height, int width, int stride);
    
    public abstract void set1D(long address, float[] value, int length);
    public abstract Syncer set2D(long address, float value, int height, int width, int stride);
    
    public abstract Syncer setFrom1Dto2D(long src_address, int src_length,
            long dst_address, int dst_height, int dst_width, int dst_stride);
    public abstract Syncer setFrom2Dto1D(long src_address, int src_height, int src_width, int src_stride,
            long dst_address, int dst_length);
    public abstract Syncer setFrom2Dto2D(long src_address, int src_height, int src_width, int src_stride,
            long dst_address, int dst_height, int dst_width, int dst_stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    public abstract Syncer gappedMemcpy2D( 
            long X_address, int Xstart, int strideX,
            long Y_address, int Ystart, int strideY,
            int width, int length);
    
    public abstract Syncer transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim,
            int dimIndex1, int dimIndex2, 
            int strideX, int strideY, 
            int length);
    
    public abstract Syncer rot180(long Y_address,
            long X_address,
            int IH, int IW, int IC, 
            int length);
    
    public abstract Syncer srcIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width, int stride); 
    
    public abstract Syncer dstIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width, int stride); 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //(N, M, K) % 4 == 0
    public abstract Syncer matMul(long C_address, 
            long A_address, long B_address,
            int N, int M, int K);
    
    public abstract Syncer matMul_biased(long C_address, 
            long A_address, long B_address,
            int N, int M, int K,
            long Bias_address,//lengthv = C.lengthv = N*M, stride = C.mem_stride = M
            int lengthv, int width);
    
    public abstract Syncer matMulT1(long C_address,  
            long A_address, long B_address,
            int N, int M, int K);
    
    public abstract Syncer matMulT2(long C_address, 
            long A_address, long B_address,
            int N, int M, int K);
    
    public abstract Syncer batchMatMul(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int BK, int AK);
    
    public abstract Syncer batchMatMulT1(long C_address, 
            long A_address, long B_address,
            int Batch, int CN, int AN, int M, int K);
    
    public abstract Syncer batchMatMulT2(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int CM, int BM, int K); 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convolution3D">
    public abstract Syncer conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw,
            long Bias_address, //stride = OC, lengthv = N*OH*OW*OC
            int lengthv, int width);    
     
    public abstract Syncer dconv3D_deltaW(
            long deltaW_address, int FH, int FW, 
            long X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer dconv3D_deltaX(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pool2D">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public abstract Syncer pool2D_max(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer pool2D_max_indexed(
            long Y_address, long Index_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer pool2D_avg(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer pool2D_avg_ignore_padding(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public abstract Syncer upool2D_max(
            long deltaX_address, long X_address, int IH, int IW,
            long deltaY_address, long Y_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
   
    public abstract Syncer upool2D_max_Indexed(
            long deltaX_address, int IH, int IW,
            long deltaY_address, long Index_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer upool2D_avg(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    
    public abstract Syncer upool2D_avg_ignore_padding(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="greater, equal, linear, rpl, div, quadratic, add_div"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs, linear_greater">
    public abstract Syncer equal_abs2D(long Y_address, 
            long X1_address, long X2_address,
            float min, float max,
            int lengthv, int width, int stride);
    
    public abstract Syncer equal_abs2D_int8(long Y_address, 
            long X1_address, long X2_address,
            byte min, byte max,
            int lengthv, int width, int stride);
    
    public abstract Syncer equal_abs2D_int32(long Y_address, 
            long X1_address, long X2_address,
            int min, int max,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear_greater2D(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride); 
   
    public abstract Syncer linear_greater_dual2D(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta,  float gamma,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear2D">
    public abstract Syncer linear2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear_dual_out2D(long Y1_address, long Y2_address, 
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear2D_int8_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
     
    public abstract Syncer linear2D_float_to_int8(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear2D_int32_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
     
    public abstract Syncer linear2D_dtype_to_int32(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2D_dual">
    public abstract Syncer linear_dual2D(long Y_address,
            long X1_address, 
            long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear_summary2D(long Y_address,
            long[] Xs, float alpha, float beta, //Xs.length >= 2
            int lengthv, int width, int stride);
    
    public abstract Syncer linear_dual2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride);
    
    public abstract Syncer linear_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="quadratic2D">
    public abstract Syncer quadratic2D(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride);
    
    public abstract Syncer quadratic2D_deltaX(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_dual2D">
    
    public abstract Syncer quadratic_dual2D(long Y_address,
            long X1_address,
            long X2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width, int stride);
    
    public abstract Syncer quadratic_dual2D_deltaX(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            int lengthv, int width, int stride);
    
    public abstract Syncer quadratic_dual2D_row(long Y_address,
            long X1_address,
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width, int stride);
    
    public abstract Syncer quadratic_dual2D_field(long Y_address,
            long X1_address,
            long X2_address, int row_lengthv, 
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width, int stride);
    
   public abstract Syncer quadratic_summary2D(long Y_address,//Y = alpha*[sum(Xs, 0, n)]
            long[] Xs, float alpha, float beta, float gamma, 
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="rpl2D">
    public abstract Syncer rpl2D(long Y_address,
            float alpha, long X_address, float beta, float gamma,
            int lengthv, int width, int stride);
    
    public abstract Syncer rpl2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float alpha, float gamma,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D">
    public abstract Syncer div2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma,
            int lengthv, int width, int stride);
    
    public abstract Syncer div2D_deltaX(
            long deltaX1_address, long deltaX2_address,
            long deltaY_address,
            long X1_address, float alpha1, float beta1,
            long X2_address, float alpha2, float beta2,
            int lengthv, int width, int stride);
    
    public abstract Syncer div2D_row(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer div2D_field(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="add_div2D">
    public abstract Syncer add_div2D_row(long Y_address,
            long X1_address, 
            long X2_address,
            long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width, int stride);
    
    public abstract Syncer add_div2D_field(long Y_address,
            long X1_address, 
            long X2_address,
            long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width, int stride);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, zero_nan, sqrt">  
    public abstract Syncer sign2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer ceil2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer floor2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer abs2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer abs2D_deltaX(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer zero_nan2D(long Y_address, 
            long X_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer sqrt2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer sqrt_quadratic_dual2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="min, max, clip"> 
    public abstract Syncer min2D(long Y_address,
            float alpha, long X_address, float beta,
            float vmin,
            int lengthv, int width, int stride);
    
    public abstract Syncer min_dual2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width, int stride);
    
    public abstract Syncer max2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmax,
            int lengthv, int width, int stride);
    
    public abstract Syncer max_dual2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width, int stride);
    
    public abstract Syncer clip2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmin, float vmax, 
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    public abstract Syncer exp2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);

    public abstract Syncer log2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer log2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long Y_address, float alpha,
            int lengthv, int width, int stride);
    
     public abstract Syncer relu2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride);
    
    public abstract Syncer relu2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer relu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer leakyRelu2D(long Y_address,
            long X_address, float k, 
            int lengthv, int width, int stride);
      
    public abstract Syncer leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer leakyRelu2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address, float k,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer elu2D(long Y_address,
            long X_address, float alpha, float k,
            int lengthv, int width, int stride);
    
    public abstract Syncer elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float k,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer softPlus2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride);
     
    public abstract Syncer softPlus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer softPlus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    public abstract Syncer sigmoid2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer tanh2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride);
      
    public abstract Syncer tanh2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride);
    
    public abstract Syncer softmax2D(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer softmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer logsoftmax2D(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer logsoftmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    public abstract Syncer sin2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer sin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer tan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer tan2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width, int stride);
    
    public abstract Syncer halfSin2D(long dY_address,
            float Amp, float alpha, long dX_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer halfSin2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long Y_address, float Amp, float alpha,//alpha = alpha*ampltitude
            int lengthv, int width, int stride);
    
     public abstract Syncer arcsin2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
     
    public abstract Syncer arcsin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width, int stride);
    
    public abstract Syncer arctan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer arctan2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss">
    public abstract Syncer L1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
     
    public abstract Syncer L1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer L2_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
     
    public abstract Syncer L2_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer smoothL1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
     
    public abstract Syncer smoothL1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
    
     public abstract Syncer binaryCrossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width, int stride);
     
    public abstract Syncer binaryCrossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer sigmoid_binaryCrossEntropy2D(long L_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer sigmoid_binaryCrossEntropy_deltaX(long deltaX_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Syncer crossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
     
    public abstract Syncer crossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride);
    
    public abstract Syncer softmax_crossEntropy2D(long L_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer softmax_crossEntropy2D_deltaX(long deltaX_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine_row">
    public abstract Syncer affine2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer affine2D_deltaA_v1(long deltaA_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed 
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer affine2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed 
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer affine2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long X_address,//(V2: X for Affine || V1: Y for Norm)
            int field_length, int row_lengthv,
            int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public abstract Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int row_lengthv, int lengthv, int width, int stride);
    
    public abstract Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public abstract Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public abstract Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public abstract Syncer sqBatchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address,//X2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride); 
    
    public abstract Syncer sqBatchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//X2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride); 
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public abstract Syncer sqBatchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer sqBatchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public abstract Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width, int stride);
    
    public abstract Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public abstract Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public abstract Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public abstract Syncer batchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride); 
    
    public abstract Syncer batchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride); 
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public abstract Syncer batchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer batchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public abstract Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width, int stride);
    
    public abstract Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deltaX">
    public abstract Syncer layerNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer layerNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): deltaX">
    public abstract Syncer layerNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer layerNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): {deltaA, deltaB}">
    public abstract Syncer layerNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer layerNorm2D_deltaAB_v2(long deltaA_address, long deltaB_address,
            long deltaY_address,
            long dX_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride);
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
     public abstract Syncer onehot2D_row_int8( long Y_address,
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride);
     
    public abstract Syncer onehot2D_row_int32( long Y_address,
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride);
    
    public abstract Syncer pix2tensor2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride);
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    public abstract Syncer sgd2D(long W_address,
            long[] gradients, float lr,
            int lengthv, int width, int stride);
    
    //<editor-fold defaultstate="collapsed" desc="sgdmn2D">
    public abstract Syncer sgdmn2D(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr,
            int lengthv, int width, int stride);
    
    public abstract Syncer sgdmn2D(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            int lengthv, int width, int stride);
    
    public abstract Syncer sgdmn2D_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer sgdmn2D_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    public abstract Syncer momentum2D(long W_address,
            long V_address, float a1, float a2,
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride);
    
    public abstract Syncer momentum2D(long W_address,
            long V_address, float a1, float a2,
            long[] gradients, float lr_t,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Momentum_decay">
    public abstract Syncer momentum2D_decay(long W_address,
            long V_address, float a1, float a2,
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
     public abstract Syncer momentum2D_decay(long W_address,
            long V_address, float a1, float a2,
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    public abstract Syncer rmsprop2D(long W_address,
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride);
    
    public abstract Syncer rmsprop2D(long W_address,
            long S_address, float a1, float a2, float eps_t,
            long[] gradients, float lr_t,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop_decay">
    public abstract Syncer rmsprop2D_decay(long W_address,
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer rmsprop2D_decay(long W_address,
            long S_address, float a1, float a2, float eps_t,
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adam">
    public abstract Syncer adam2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long deltaW_address, float lr_t,
           int lengthv, int width, int stride);
    
    public abstract Syncer adam2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long[] gradients, float lr_t,
           int lengthv, int width, int stride);

    public abstract Syncer adam2D_type2(long W_address,
            long V_address, float a1, float a2, float Uv, 
            long S_address, float b1, float b2, float eps, float Us,
            long deltaW_address, float lr,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam_decay">
    public abstract Syncer adam2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer adam2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    public abstract Syncer adamax2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride);
    
    public abstract Syncer adamax2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long[] gradients, float lr_t,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamaxs_decay">
    public abstract Syncer adamax2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer adamax2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    public abstract Syncer adamW2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps,
            long deltaW_address, float lr_t, float lr, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer adamW2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps,
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    public abstract Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2, 
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride);
    
    public abstract Syncer adamod2D(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2,
            long[] gradients, float lr_t,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamodDecay">
    public abstract Syncer adamod2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride);
    
    public abstract Syncer adamod2D_decay(long W_address,
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2,
            long[] gradients, float lr_t,
              float L1coef, float L2coef,
            int lengthv, int width, int stride);
    //</editor-fold.
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    public abstract Syncer bernouli2D(long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride);
    
    public abstract Syncer bernouli_mul2D(long Y_address, long R_address, 
            long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride);
    
    public abstract Syncer uniform2D(long X_address, 
            int seed, 
            float vmin, float vmax,
            int lengthv, int width, int stride);
    
    public abstract Syncer sparse_uniform2D(long X_address, 
            int seed1, int seed2, 
            float p, float vmin, float vmax,
            int lengthv, int width, int stride);
    
    public abstract Syncer gaussian2D(long X_address,
            int seed1, int seed2, 
            float mu, float sigma, 
            int lengthv, int width, int stride);
    
    public abstract Syncer sparse_gaussian2D(long X_address,
            int seed1, int seed2, int seed3, 
            float p, float mu, float sigma,
            int lengthv, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    public abstract Result<Float> straight_linear(long X_address,
            float alpha, float beta,
            int lengthv, int width, int stride);
    
    public abstract Result<Float> straight_quadratic(long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride);
    
    public abstract Result<Float> straight_max(long X_address,
            int lengthv, int width, int stride);
     
    public abstract Result<Float> straight_min(long X_address,
            int lengthv, int width, int stride);
    
    public abstract IndexedResult<Float> straight_max_indexed(long X_address,
            int lengthv, int width, int stride);
     
    public abstract IndexedResult<Float> straight_min_indexed(long X_address,
            int lengthv, int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    public abstract Syncer field_linear(long Y_address,
            long X_address,
            float alpha, float beta,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_linear_dual(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_quadratic(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_quadratic_dual(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_linear_quadratic(long Y1_address, long Y2_address,
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_var(long var_address, 
            long mean_address, long squareMean_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_std(long stddev_address, 
            long mean_address, long squareMean_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer field_max(long Y_address, 
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride);
    
    public abstract Syncer field_min(long Y_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride);
    
    public abstract Syncer field_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride);
    
    public abstract Syncer field_min_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    public abstract Syncer row_linear(long Y_address,
            long X_address,
            float alpha, float beta, 
            int field_length, int row_lengthv,
            int width, int stride);
    
     public abstract Syncer row_linear_dual(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_quadratic(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_quadratic_dual(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_linear_quadratic(long Y1_address, long Y2_address,
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_var(long var_address,
            long mean_address, long squareMean_address,
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_stddev(long stddev_address,
            long mean_address, long squareMean_address,
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_max(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
     
    public abstract Syncer row_min(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
    
    public abstract Syncer row_max_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
     
    public abstract Syncer row_min_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride);
    //</editor-fold>
    //</editor-fold>
}

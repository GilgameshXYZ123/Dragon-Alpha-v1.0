/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.Objects;
import java.util.Random;
import z.dragon.common.MemStatus;
import z.dragon.engine.memp.Mempool;
import z.dragon.engine.Result.IndexedResult;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;

/**
 * 4D[N, IH, IW, IC]
 * dim: 0, 1, 2, 3.
 * @author Gilgamesh
 */
public class EngineCore implements MemStatus
{
    public static final long MEM_1GB = (1L) << 30;
    public static final long MEM_1MB = (1L) << 20;
    public static final long NULL = 0L;
    
    public static final long L_sizeof_int32 = 2;
    public static final long L_sizeof_int8 = 0;
    
    public static final long sizeof_int32 = 4;
    public static final long sizeof_int8 = 1;
    
    //<editor-fold defaultstate="collapsed" desc="member params">
    protected EngineBase base;
    protected long L_sizeof_datatype;//sizeof_datatype = 1<<L_sizeof_datatype
    protected Mempool mempool;
    
    protected boolean check = true;
    
    protected Random exr = new ExRandom();
    //</editor-fold>
    
    protected EngineCore() {}
    public EngineCore(Mempool memp, boolean check) {
        if(memp == null) throw new NullPointerException("Mempool is null");
        this.mempool = memp;
        this.check = check;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean check() { return check; }
    public void check(boolean flag) { this.check = flag; }
    
    public Random random() { return exr; }
    public EngineCore random(Random random) { 
        if(random == null) throw new NullPointerException("random is null");
        exr = random; 
        return this; 
    }
    
    public EngineBase engineBase() {return base;}
    public synchronized EngineCore engineBase(EngineBase base) {
        if(base == null) throw new NullPointerException("EngineBase is null");
        this.base = base; base.core = this;
        this.L_sizeof_datatype = base.L_sizeof_datatype;
        this.mempool.engineBase(base);
        return this;
    }
    
    public String dataType()        { return base.datatype; }
    public String dataType_int32() { return base.datatype_int32; }
    public String dataType_int8()  { return base.datatype_int8; }
    
    public long LsizeOf_dataType() { return L_sizeof_datatype; }
    public long sizeOf_dataType()  { return 1 << L_sizeof_datatype; }
    
    public Mempool mempool() { return this.mempool; }
    public synchronized EngineCore mempool(Mempool memp) {
        if(memp == null) throw new NullPointerException("Mempool is null");
        this.mempool = memp;
        if(base != null) this.mempool.engineBase(base);
        return this;
    }
    
    @Override public long max_mem_size() { return mempool.max_mem_size(); }
    @Override public long total_mem_size() { return mempool.total_mem_size(); }
    @Override public long used_mem_size() { return mempool.used_mem_size(); }

    public EngineCore max_mem_size(long maxMemSize) { mempool.max_mem_size(maxMemSize); return this; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("\ncheck = ").append(check);
        if(mempool != null) mempool.meta_data().forEach((String key, Object value)-> {
             sb.append("\n\t mempool.").append(key).append(" = ").append(value);
        });
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    
    public void clear() {
        mempool.clear();
    }
    
    @Override
    public void finalize() throws Throwable  {
        super.finalize();
        this.clear();
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 31 * hash + Objects.hashCode(this.base);
        hash = 31 * hash + Objects.hashCode(this.mempool);
        return hash;
    }
    
    @Override
    public boolean equals(Object o) {
        if(!(o instanceof EngineCore)) return false;
        EngineCore core = (EngineCore) o;
        return Objects.equals(core.engineBase(), base);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="extra: int32"> 
    public Syncer memset_int32(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must positive");
        }
        return base.memset(address, value, length << 2);//length * 4
    }
    
    public int[] get1D_int32(long address, int length)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must positive");
        }
        int[] value = new int[length];
        base.get1D_int32(address, value, length);
        return value;
    }
    
    public int[] get2D_int32(long address, int height, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("mem_height must positive ");
            if(width <= 0) throw new IllegalArgumentException("mem_width mst positive");
        }
        int stride = ((width + 3) >> 2) << 2;
        int[] value = new int[height * width];
        base.get2D_int32(address, value, height, width, stride);
        return value;
    }
    
    public void set1D_int32(long address, int[] value) {
        if(address == NULL) throw new NullPointerException("Tensor.address is null");
        base.set1D_int32(address, value, value.length);
    }
    
    public void set2D_int32(long address, int[] value, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(value.length < width) throw new IllegalArgumentException("value<int[]>,length must positive");
            if(width <= 0) throw new IllegalArgumentException("mem_width must positive");
            if(value.length % width !=0) throw new IllegalArgumentException();
        }
        
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D_int32(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="extra: int8">
    public Syncer memset_int8(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must positive");
        }
        return base.memset(address, value, length);//length * 1
    }
    
    public byte[] get1D_int8(long address, int length)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must positive");
        }
        byte[] value = new byte[length];
        base.get1D_int8(address, value, length);
        return value;
    }
    
    public byte[] get2D_int8(long address, int height, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("mem_height must positive ");
            if(width <= 0) throw new IllegalArgumentException("mem_width mst positive");
        }
        
        int stride = ((width + 3) >> 2) << 2;
        byte[] value = new byte[height * width];
        base.get2D_int8(address, value, height, width, stride);
        return value;
    }
    
    public void set1D_int8(long address, byte[] value) {
        if(address == NULL) throw new NullPointerException("Tensor.address is null");
        base.set1D_int8(address, value, value.length);
    }
    
    public void set2D_int8(long address, byte[] value, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(value.length < width) throw new IllegalArgumentException("value<char[]>,length must positive");
            if(width <= 0) throw new IllegalArgumentException("mem_width must positive");
            if(value.length % width !=0) throw new IllegalArgumentException();
        }
        
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D_int8(address, value, height, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Pooling: create & delete"> 
    //malloc: return { mem_size, mem_address }
    public long[] malloc(int mem_length) { return mempool.malloc(check, mem_length, L_sizeof_datatype); }    
    public long[] malloc_int32(int mem_length) { return mempool.malloc(check, mem_length, 2); }//log2(32 / 8) = 2
    public long[] malloc_int8(int mem_length) { return mempool.malloc(check, mem_length, 0); }//log2(8 / 8) = 0
   
    //padding to (1 << L_sizeof_datatype), As sizeof_datatype may not a power of 2
    public long[] malloc_dtype(int mem_length, long sizeof_dtype) {
        long mem_size = (mem_length * sizeof_dtype) +  + (1 << L_sizeof_datatype) - 1;
        mem_length = (int) (mem_size >> L_sizeof_datatype);
        return mempool.malloc(check, mem_length, L_sizeof_datatype);
    }
    
    public boolean free(long mem_size, long mem_address) {
        return mempool.free(check, mem_size, mem_address);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Opt">   
    public Syncer memset(long address, int value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException("length must positive");
        }
        return base.memset(address, value, length << L_sizeof_datatype);
    }
    
    public Syncer memcpy(long dst_address, long src_address, int length) {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(length <= 0) throw new IllegalArgumentException();
        }
        return base.memcpy(dst_address, src_address, length << L_sizeof_datatype);
    }
    
    //<editor-fold defaultstate="collapsed" desc="get tensor value">
    public float[] get1D(long address, int length)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException();
        }
        float[] value = new float[length];
        base.get1D(address, value, length);
        return value;
    }
    
    public float[] get2D(long address, int height, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("mem_height <= 0 ");
            if(width <= 0) throw new IllegalArgumentException("mem_width <= 0");
        }
        int stride = ((width + 3) >> 2) << 2;
        float[] value = new float[height * width];
        base.get2D(address, value, height, width, stride);
        return value;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = fill(constant)">
    public Syncer set1D(long address, float value, int length) {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(length <= 0) throw new IllegalArgumentException();
        }
        return base.set1D(address, value, length);
    }

    public Syncer set2D(long address, float value, int height, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(height <= 0) throw new IllegalArgumentException("height must positive");
            if(width <= 0) throw new IllegalArgumentException("height must positive");
        }
        int stride = ((width + 3) >> 2) << 2;
        return base.set2D(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = value<float[]>">
    public void set1D(long address, float[] value) {
        if(address == NULL) throw new NullPointerException("Tensor.address is null");
        base.set1D(address, value, value.length);
    }
    
    public void set2D(long address, float[] value, int width)
    {
        if(check) {
            if(address == NULL) throw new NullPointerException("Tensor.address is null");
            if(value.length < width) throw new IllegalArgumentException();
            if(width <= 0) throw new IllegalArgumentException("mem_width <= 0");
            if(value.length % width !=0) throw new IllegalArgumentException();
        }
        
        int height = value.length / width;
        int stride = ((width + 3) >> 2) << 2;
        base.set2D(address, value, height, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = another tensor">
    public Syncer setFrom1Dto2D(long src_address, int src_length,
            long dst_address, int dst_height, int dst_width)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(src_length <= 0) throw new IllegalArgumentException("src.length <= 0");
            if(dst_width <= 0) throw new IllegalArgumentException("dst.mem_width <= 0");
            if(dst_height <= 0) throw new IllegalArgumentException("dst.mem_height <= 0");
            if(src_length != dst_height * dst_width) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int dst_stride = ((dst_width + 3) >> 2) << 2;
        return base.setFrom1Dto2D(
                src_address, src_length, dst_address, 
                dst_height, dst_width, dst_stride);
    }
    
    public Syncer setFrom2Dto1D(long src_address, int src_height, int src_width,
            long dst_address, int dst_length)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(dst_length <= 0) throw new IllegalArgumentException("dst.length <= 0");
            if(src_width <= 0) throw new IllegalArgumentException("src.mem_width <= 0");
            if(src_height <= 0) throw new IllegalArgumentException("src.mem_height <= 0");
            if(src_height * src_width != dst_length) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int src_stride = ((src_width + 3) >> 2) << 2;
        return base.setFrom2Dto1D(
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_length);
    }
    
    public Syncer setFrom2Dto2D(long src_address, int src_height, int src_width,
            long dst_address, int dst_height, int dst_width)
    {
        if(check) {
            if(src_address == NULL) throw new NullPointerException("Tensor src is null");
            if(dst_address == NULL) throw new NullPointerException("Tensor dst is null");
            if(src_width <= 0) throw new IllegalArgumentException("src.mem_width <= 0");
            if(src_height <= 0) throw new IllegalArgumentException("src.mem_height <= 0");
            if(dst_width <= 0) throw new IllegalArgumentException("dst.mem_width <= 0");
            if(dst_height <= 0) throw new IllegalArgumentException("dst.mem_height <= 0");
            if(src_height * src_width != dst_height * dst_width) throw new IllegalArgumentException("src.length != dst.length");
        }
        
        int src_stride = ((src_width + 3) >> 2) << 2;
        int dst_stride = ((dst_width + 3) >> 2) << 2;
        return base.setFrom2Dto2D(
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_height, dst_width, dst_stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    public Syncer gappedMemcpy2D(
            long X_address, int Xstart, int strideX, 
            long Y_address, int Ystart, int strideY,
            int width, int length)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");

            if(Xstart < 0) throw new IllegalArgumentException("XstartIndex < 0");
            if(Ystart < 0) throw new IllegalArgumentException("YstartIndex < 0");
            
            if(strideX < width) throw new IllegalArgumentException();
            if(strideY < width) throw new IllegalArgumentException();
            
            if(length < width) throw new IllegalArgumentException();
            if(length % width != 0) throw new IllegalArgumentException();
            if(width < 0) throw new IllegalArgumentException();
        }
        return base.gappedMemcpy2D(
                X_address, Xstart, strideX,
                Y_address, Ystart, strideY, 
                width, length);
    }
    
    public Syncer transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int widthX, int widthY, 
            int length) 
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            
            if(length < widthX) throw new NullPointerException();
            if(length < widthY) throw new NullPointerException();
            if(widthX <= 0) throw new NullPointerException();
            if(widthY <= 0) throw new NullPointerException();
            if(length % widthX != 0) throw new NullPointerException();
            if(length % widthY != 0) throw new NullPointerException();
        }
        return base.transpose(
                Y_address, Ydim,
                X_address, Xdim, 
                dimIndex1, dimIndex2, 
                (widthX + 3) >> 2 << 2,
                (widthY + 3) >> 2 << 2,
                length);
    }
    
    public Syncer rot180_3D(long Y_address,
            long X_address,
            int N, int IH, int IW, int IC)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            
            if(N <= 0) throw new IllegalArgumentException();
            if(IH <= 0) throw new IllegalArgumentException();
            if(IW <= 0) throw new IllegalArgumentException();
            if(IC <= 0) throw new IllegalArgumentException();
        }
        IC = ((IC + 3) >> 2) << 2;
        int length = N * IH * IW * IC;
        return base.rot180(Y_address, X_address, IH, IW, IC, length);
    }
    
    public Syncer srcIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor is null");
            
            if(lengthv < width) throw new IllegalArgumentException();
            if(width <= 0) throw new IllegalArgumentException();
            if(lengthv % stride != 0) throw new IllegalArgumentException();
        }
        
        return base.srcIndexedMemcpy(Y_address, X_address, Index_address, 
                lengthv, width, stride);
    }
    
    public Syncer dstIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor is null");
            
            if(lengthv < width) throw new IllegalArgumentException();
            if(width <= 0) throw new IllegalArgumentException();
            if(lengthv % stride != 0) throw new IllegalArgumentException();
        }
        
        return base.dstIndexedMemcpy(Y_address, X_address, Index_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="param check">
    protected static void matMul_check(long C_address, long A_address, long B_address,
            int N, int M, int K)
    {
        if(C_address == NULL) throw new NullPointerException("Tensor C is null");
        if(B_address == NULL) throw new NullPointerException("Tensor B is null");
        if(A_address == NULL) throw new NullPointerException("Tensor A is null");
        
        if(N <= 0) throw new IllegalArgumentException("MatMul: N must be positive");
        if(M <= 0) throw new IllegalArgumentException("MatMul: M must be positive");
        if(K <= 0) throw new IllegalArgumentException("MatMul: K must be positive");
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Normal matMul">
    public Syncer matMul(long C_address,
            long A_address, long B_address,
            int N, int M, int K) 
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMul(C_address, A_address, B_address, 
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    
    public Syncer matMul_biased(long C_address, 
            long A_address, long B_address,
            int N, int M, int K,
            long Bias_address,//lengthv = C.lengthv = N*M, stride = C.mem_stride = M
            int lengthv)
    {
        int M_4x = ((M + 3) >> 2) << 2;
        if(check){
            matMul_check(C_address, A_address, B_address, N, M, K);
            //lengthv = Y.lengthv = N * M_4x
            //row_lengthv = M_4x
            //[width, stride] = M, M_4x
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");
            func_param_check_row(lengthv, M_4x, M, M_4x);
        }
        return base.matMul_biased(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                M_4x, //((M + 3) >> 2) << 2
                ((K + 3) >> 2) << 2,
                Bias_address,
                lengthv, M);
    }
    
    //transpose B
    public Syncer matMulT1(long C_address,
            long A_address, long B_address,
            int N, int M, int K)
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMulT1(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    
    //transpose A
    public Syncer matMulT2(long C_address, long 
            A_address, long B_address,
            int N, int M, int K)
    {
        if(check) matMul_check(C_address, A_address, B_address, N, M, K);
        return base.matMulT2(C_address, A_address, B_address,
                ((N + 3) >> 2) << 2, 
                ((M + 3) >> 2) << 2, 
                ((K + 3) >> 2) << 2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Batch matMul">
    public Syncer batchMatMul(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int K) 
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            EngineCore.matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //A[Batch, N, AK] * B[Batch, BK, M] = C[Batch, N, M]
        return base.batchMatMul(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2,//Batch % 4 == 0
                N, ((M + 3) >> 2) << 2, //N % 4 != 0, M % 4 == 0
                K, ((K + 3) >> 2) << 2);//K = BK % 4 != 0, AK % 4 == 0
    }
    
    //A.transpose(1, 2)
    public Syncer batchMatMulT1(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int K)
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            EngineCore.matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //(A[Batch, K, AN])^T * B[Batch, K, M] = C[Batch, CN, M]
        return base.batchMatMulT1(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2,//Batch % 4 == 0
                N, ((N + 3) >> 2) << 2, //N = CN % 4 != 0, AN % 4 == 0
                ((M + 3) >> 2) << 2, K);//memAligned, M % 4 == 9, K % 4 ! = 0
    }
    
    //B.transpose(1, 2)
    public Syncer batchMatMulT2(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int K)
    {
        if(check) {
            if(Batch <= 0) throw new IllegalArgumentException("Batch MatMul: Batch must be positive");
            EngineCore.matMul_check(C_address, A_address, B_address, N, M, K);
        }
        //A[Batch, N, K] * (B[Batch, BM, K])^T = C[Batch, N, CM]
        return base.batchMatMulT2(C_address, A_address, B_address,
                ((Batch + 3) >> 2) << 2, N, //Batch % 4 == 0, N % 4 != 0
                ((M + 3) >> 2) << 2, M,//M = CM % 4 == 0, BM % 4 != 0
                ((K + 3) >> 2) << 2); //K % 4 == 0
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convlution 3D">
    //<editor-fold defaultstate="collapsed" desc="param check">
    private static void conv3D_param_check(boolean strict,
            int OH, int OW, int IH, int IW, int FH, int FW, 
            int N, int IC, int OC, int sh, int sw, int ph, int pw)
    {
        if(OH <= 0) throw new IllegalArgumentException("output_height <= 0");
        if(OW <= 0) throw new IllegalArgumentException("output_width <= 0");
        if(N <= 0) throw new IllegalArgumentException("Y.N <= 0");
        if(IC <= 0) throw new IllegalArgumentException("input_channel <= 0");
        if(OC <= 0) throw new IllegalArgumentException("output_channel <= 0");
        if(sh <= 0) throw new IllegalArgumentException("stride_height <= 0");
        if(sw <= 0) throw new IllegalArgumentException("srtide_width <= 0");
        if(ph < 0) throw new IllegalArgumentException("padding_height < 0");
        if(pw < 0) throw new IllegalArgumentException("padding_width < 0");
        
        if(FH <= ph) throw new IllegalArgumentException("W.FH <= ph");
        if(FW <= pw) throw new IllegalArgumentException("W.FW <= pw");
        
        if(FH > IW + (pw<<1)) throw new IllegalArgumentException("X.IW <= W.FW");
        if(FH > IH + (ph<<1)) throw new IllegalArgumentException("X.IH <= W.FH");
        
        if(sh > FH) throw new IllegalArgumentException("sh > FH");
        if(sw > FW) throw new IllegalArgumentException("sw > FW");
        
        if(IH - FH + (ph << 1) < (OH - 1)*sh) {
            throw new IllegalArgumentException(String.format("Illegal conv3D size, we must have:\n"
                    + "(input_height - kernel_height + padding_height*2) >= (out_height - 1) * stride_height\n"
                    + "{ %d - %d + %d*2 >= (%d - 1)*%d }",
                    IH, FH, ph, OH, sh));
        }
        if(strict && IH - FH + (ph << 1) != (OH - 1)*sh) {
            throw new IllegalArgumentException(String.format("Illegal conv3D size, we must have:\n"
                    + "(input_height - kernel_height + padding_height*2) != (out_height - 1) * stride_height\n"
                    + "{ %d - %d + %d*2 != (%d - 1)*%d }",
                    IH, FH, ph, OH, sh));
        }

        if(IW - FW + (pw << 1) < (OW - 1)*sw) {
            throw new IllegalArgumentException( String.format( "Illegal conv3D size, we must have:\n"
                    + "(input_width - kernel_width + padding_width) >= (out_width - 1) * stride_width\n"
                    + "{ %d - %d + %d*2 != (%d - 1)*%d }",
                    IW, FW, pw, OW, sw));
        }
        if(strict && IW - FW + (pw << 1) != (OW - 1)*sw) {
            throw new IllegalArgumentException( String.format( "Illegal conv3D size, we must have:\n"
                    + "(input_width - kernel_width + padding_width) != (out_width - 1) * stride_width\n"
                    + "{ %d - %d + %d*2 != (%d - 1)*%d }",
                    IW, FW, pw, OW, sw));
        }   
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW,
            long W_address, int FH, int FW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FH, FW, IC]
            conv3D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.conv3D(Y_address, OH, OW, 
                X_address, IH, IW,
                W_address, FH, FW, 
                ((N  + 3) >> 2) << 2,
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
   
    public Syncer conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw,
            long Bias_address, int lengthv) //width = OC, lengthv = N*OH*OW*OC
    {
        int OC_4x = ((OC + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");//X[N, IH, IW, IC]
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");//Y[N, OH, OW, OC]
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");//W[OC, FH, FW, IC]
            conv3D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
            
            //lengthv = Y.lengthv = N * OH * OW * OC_4x
            //row_lengthv = OC_4x
            //[width, stride] = OC, OC_4x
            if(Bias_address == NULL) throw new NullPointerException("Tensor Bias is null");//Bias[OC]
            func_param_check_row(lengthv, OC_4x, OC, OC_4x);
        }
        return base.conv3D_biased(Y_address, OH, OW,
                X_address, IH, IW,
                W_address, FH, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                OC_4x, //((OC + 3) >> 2) << 2
                sh, sw, ph, pw, 
                Bias_address,
                lengthv, OC);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="back propagation">
    public Syncer dconv3D_deltaX(boolean strict,
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph ,int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            conv3D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.dconv3D_deltaX(
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, 
                W_address, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    public Syncer dconv3D_deltaW(boolean strict, 
            long deltaW_address, int FH, int FW, 
            long X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            conv3D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, OC, sh, sw, ph, pw);
        }
        return base.dconv3D_deltaW(
                deltaW_address, FH, FW, 
                X_address, IH, IW, 
                deltaY_address, OH, OW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                ((OC + 3) >> 2) << 2, 
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pooling 2D"> 
    //<editor-fold defaultstate="collapsed" desc="param check">
    private static void pool2D_param_check(boolean strict,
            int OH, int OW, int IH, int IW, int FH, int FW, 
            int N, int IC, int sh, int sw, int ph, int pw)
    {
        if(OH <= 0) throw new IllegalArgumentException("Y.OH <= 0");
        if(OW <= 0) throw new IllegalArgumentException("Y.OW <= 0");
        if(N <= 0) throw new IllegalArgumentException("N <= 0");
        if(IC <= 0) throw new IllegalArgumentException("IC <= 0");
        if(sh <= 0) throw new IllegalArgumentException("sh <= 0");
        if(sw <= 0) throw new IllegalArgumentException("sw <= 0");
        if(ph < 0) throw new IllegalArgumentException("ph < 0");
        if(pw < 0) throw new IllegalArgumentException("pw < 0");
        
        if(FH <= ph) throw new IllegalArgumentException("W.FH <= ph");
        if(FW <= pw) throw new IllegalArgumentException("W.FW <= pw");
         
        if(FH > IH + (ph << 1)) throw new IllegalArgumentException("X.IH <= W.FH + 2*ph");
        if(FH > IW + (pw << 1)) throw new IllegalArgumentException("X.IW <= W.FW + 2*pw");
        
        if(sh > FH) throw new IllegalArgumentException("sh > FH");
        if(sw > FW) throw new IllegalArgumentException("sw > FW");
        if(sh > IH) throw new IllegalArgumentException("sh > IH");
        if(sw > IW) throw new IllegalArgumentException("sw > IW");
        if(sh * sw < 2 && FH * FW < 2) throw new IllegalArgumentException("sh * sw < 2 && FH * FW < 2");
        if(FH * FW < 2) throw new IllegalArgumentException("FH * FW < 2");
        
        if(IH - FH + (ph << 1) < (OH - 1)*sh) {
            throw new IllegalArgumentException(String.format("Illegal conv3D size, we must have:\n"
                    + "input_height - kernel_height + padding_height*2 >= (out_height - 1) * stride_height"
                    + "{%d - %d + %d*2 >= (%d - 1)*%d}",
                    IH, FH, ph, OH, sh));
        }
        if(strict && IH - FH + (ph << 1) != (OH - 1)*sh) {
            throw new IllegalArgumentException(String.format("Illegal conv3D size, we must have:\n"
                    + "input_height - kernel_height + padding_height*2 != (out_height - 1) * stride_height"
                    + "{%d - %d + %d*2 != (%d - 1)*%d}",
                    IH, FH, ph, OH, sh));
        }

        if(IW - FW + (pw << 1) < (OW - 1)*sw) {
            throw new IllegalArgumentException( String.format( "Illegal conv3D size, we must have:\n"
                    + "input_width - kernel_width + padding_width >= (out_width - 1) * stride_width"
                    + "{%d - %d + %d*2 != (%d - 1)*%d}",
                    IW, FW, pw, OW, sw));
        }
        if(strict && IW - FW + (pw << 1) != (OW - 1)*sw) {
            throw new IllegalArgumentException( String.format( "Illegal conv3D size, we must have:\n"
                    + "input_width - kernel_width + padding_width != (out_width - 1) * stride_width"
                    + "{%d - %d + %d*2 != (%d - 1)*%d}",
                    IW, FW, pw, OW, sw));
        }   
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer pool2D_max(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.pool2D_max(Y_address, OH, OW, 
                X_address, IH, IW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer pool2D_max_indexed(
            long Y_address, long Index_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Index is null");
            pool2D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.pool2D_max_indexed(Y_address, Index_address, OH, OW, 
                X_address, IH, IW, FH, FW,
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer pool2D_avg(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.pool2D_avg(Y_address, OH, OW, 
                X_address, IH, IW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer pool2D_avg_ignore_padding(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            pool2D_param_check(false, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.pool2D_avg_ignore_padding(Y_address, OH, OW, 
                X_address, IH, IW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    public Syncer upool2D_max(boolean strict,
            long deltaY_address, long Y_address, int OH, int OW, 
            long deltaX_address, long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            pool2D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.upool2D_max(
                deltaX_address, X_address, IH, IW, 
                deltaY_address, Y_address, OH, OW,
                FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer upool2D_max_Indexed(boolean strict,
            long deltaX_address, int IH, int IW,
            long deltaY_address, long Index_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor<int32> Index is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            pool2D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.upool2D_max_Indexed(
                deltaX_address, IH, IW, 
                deltaY_address, Index_address, OH, OW, 
                FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer upool2D_avg(boolean strict,
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            pool2D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.upool2D_avg(
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    
    public Syncer upool2D_avg_ignore_padding(boolean strict,
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            pool2D_param_check(strict, OH, OW, IH, IW, FH, FW, N, IC, sh, sw, ph, pw);
        }
        
        return base.upool2D_avg_ignore_padding(
                deltaX_address, IH, IW, 
                deltaY_address, OH, OW, FH, FW, 
                ((N  + 3) >> 2) << 2, 
                ((IC + 3) >> 2) << 2,
                sh, sw, ph, pw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="param check">
    private void func_param_check(int lengthv, 
            int width, int stride) 
    {
        if(lengthv < width) throw new IllegalArgumentException("lengthv < width");
        if(width <= 0) throw new IllegalArgumentException("width <= 0");
        if(lengthv % stride != 0) throw new IllegalArgumentException("lengthv % stride != 0");
    }
    
    private void func_param_check_row(int lengthv, 
            int row_lengthv, 
            int width, int stride) 
    {
        if(lengthv < row_lengthv) throw new IllegalArgumentException("lengthv < row_lengthv");
        if(width <= 0) throw new IllegalArgumentException("width < 0");
        if(lengthv % row_lengthv != 0) throw new IllegalArgumentException("lengthv % row_lengthv != 0");
        if(row_lengthv % stride != 0) throw new IllegalArgumentException("row_lengthv % stride != 0");
    }
    
    private void func_param_check_field(int lengthv, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        if(lengthv < row_lengthv) throw new IllegalArgumentException("lengthv < row_lengthv");
        if(width <= 0) throw new IllegalArgumentException("width <= 0");
        if(lengthv % field_length != 0) throw new IllegalArgumentException("lengthv % field_length != 0");
        if(row_lengthv % stride != 0) throw new IllegalArgumentException("row_lengthv % stride != 0");
    }
    
    private void func_param_check_softmax(int lengthv,
            int row_length, int row_lengthv, 
            int width) 
    {
        if(lengthv < row_length) throw new IllegalArgumentException("lengthv < row_length");
        if(row_length < width) throw new IllegalArgumentException("row_length < width");
        if(width <= 0) throw new IllegalArgumentException("width <= 0");
        if(row_length % width != 0) throw new IllegalArgumentException("row_length % width != 0");
        if(lengthv % row_lengthv != 0) throw new IllegalArgumentException("lengthv % row_lengthv != 0");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="greater, equal, linear, rpl, div, quadratic"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    public Syncer equal_abs2D(long Y_address, 
            long X1_address, long X2_address,
            float min, float max,
            int lengthv, int width)
    {
        if(min > max) { float t = min; min = max; max = t; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException("min must be non-negative");
            if(max < 0) throw new IllegalArgumentException("max must be non-negative");
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D(Y_address,
                X1_address, X2_address,
                min, max, 
                lengthv, width, stride);
    }
    
    public Syncer equal_abs2D_int8(long Y_address, 
            long X1_address, long X2_address,
            byte min, byte max,
            int lengthv, int width)
    {
        if(min > max) { byte t = min; min = t; t = max; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException("min must be non-negative");
            if(max < 0) throw new IllegalArgumentException("max must be non-negative");
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D_int8(Y_address,
                X1_address, X2_address, 
                min, max, 
                lengthv, width, stride);
    }
    
    public Syncer equal_abs2D_int32(long Y_address, 
            long X1_address, long X2_address,
            int min, int max,
            int lengthv, int width)
    {
        if(min > max) { int t = min; min = max; max = t; }
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(min < 0) throw new IllegalArgumentException("min must be non-negative");
            if(max < 0) throw new IllegalArgumentException("max must be non-negative");
            func_param_check(lengthv, width, stride);
        }
        return base.equal_abs2D_int32(Y_address,
                X1_address, X2_address,
                min, max, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater">   
    public Syncer linear_greater2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address  == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }

    public Syncer linear_greater_dual2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address  == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address  == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_greater_dual2D(Y_address, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    public Syncer linear2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer linear_dual_out2D(long Y1_address, long Y2_address,
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_dual_out2D(Y1_address, Y2_address,
                X_address, 
                alpha1, beta1, 
                alpha2, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to dtype">
    public Syncer linear2D_int8_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_int8_to_dtype(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
     
    public Syncer linear2D_dtype_to_int8(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_float_to_int8(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to dtype">
    public Syncer linear2D_int32_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_int32_to_dtype(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
     
    public Syncer linear2D_dtype_to_int32(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear2D_dtype_to_int32(Y_address, alpha, X_address, beta, 
                lengthv, width, stride);
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="linear_dual">
    public Syncer linear_dual2D(long Y_address,
            long X1_address, 
            long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_dual2D(Y_address, 
                X1_address,
                X2_address,
                alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer linear_summary2D(boolean inplaced, long Y_address,
            float alpha, float beta, long[] Xs,//X2.length >= 2
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(Xs, "Tensor Xs");
            if(Xs.length < 2) throw new IllegalArgumentException("At least 2 Tensors to find their summary");
            func_param_check(lengthv, width, stride);
        }
        return base.linear_summary2D(Y_address, 
                Xs, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_dual: element/row/field">
    public Syncer linear_dual2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.linear_dual2D_row(Y_address, 
                X1_address,
                X2_address, row_lengthv, 
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    
    public Syncer linear_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int field_length,
            float alpha, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_field(lengthv,  field_length, row_lengthv, width, stride); 
        }
        return base.linear_dual2D_field(Y_address,
                X1_address,
                X2_address, row_lengthv,
                alpha, beta, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    public Syncer quadratic2D(long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2D(Y_address,
                X_address, alpha, beta, gamma, 
                lengthv, width, stride);
    }

    public Syncer quadratic2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic2D_deltaX(deltaX_address,
                deltaY_address,
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic_dual">
    public Syncer quadratic_dual2D(long Y_address,
            long X1_address,
            long X2_address,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic_dual2D(Y_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, C, 
                lengthv, width, stride);
    }

    public Syncer quadratic_dual2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(deltaY_address  == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic_dual2D_deltaX(deltaX1_address, deltaX2_address,
                deltaY_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, 
                lengthv, width, stride);
    }
    
    public Syncer quadratic_summary2D(boolean inplaced, long Y_address,
            float alpha, float beta, float gamma, long[] Xs,//X2.length >= 2
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(Xs, "Tensor Xs");
            if(Xs.length < 2) throw new IllegalArgumentException("At least 2 Tensors to find their summary");
            func_param_check(lengthv, width, stride);
        }
        return base.quadratic_summary2D(Y_address, 
                Xs, alpha, beta, gamma, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_dual: row/field">
    public Syncer quadratic_dual2D_row(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.quadratic_dual2D_row(Y_address, 
                X1_address,
                X2_address, row_lengthv,
                k11, k12, k22, 
                k1, k2, C, 
                lengthv, width, stride);
    }
    
    public Syncer quadratic_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int field_length,
            float k11, float k12, float k22,
            float k1, float k2,
            float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.quadratic_dual2D_field(Y_address,
                X1_address, 
                X2_address, row_lengthv,
                k11, k12, k22,
                k1, k2, C,
                lengthv, width, stride);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    public Syncer rpl2D(long Y_address,
            float alpha, long X_address, float beta, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.rpl2D(Y_address, 
                alpha, X_address, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Syncer rpl2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float alpha, float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check(lengthv, width, stride);
        }
        return base.rpl2D_deltaX(deltaX_address,
                deltaY_address, 
                Y_address, alpha, gamma,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    public Syncer div2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(
                    "(alpha2*X2 + beta2) identically equals to zero");
            func_param_check(lengthv, width, stride);
        }
        return base.div2D(Y_address,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma,
                lengthv, width, stride);
    }
    
    public Syncer div2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, float alpha1, float beta1,
            long X2_address, float alpha2, float beta2, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX1_address == NULL) throw new NullPointerException("Tensor deltaX1 is null");
            if(deltaX2_address == NULL) throw new NullPointerException("Tensor deltaX2 is null");
            if(deltaY_address  == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(
                    "(alpha2*X2 + beta2) identically equals to zero");
            func_param_check(lengthv, width, stride);
        }
        return base.div2D_deltaX(deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, alpha1, beta1,
                X2_address, alpha2, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D: row/field">
    public Syncer div2D_row(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
             if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(
                    "(alpha2*X2 + beta2) identically equals to zero");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.div2D_row(Y_address, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                gamma, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer div2D_field(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(alpha2 == 0 && beta2 == 0) throw new IllegalArgumentException(
                    "(alpha2*X2 + beta2) identically equals to zero");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.div2D_field(Y_address,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="add_div: row\field">
    public Syncer add_div2D_row(long Y_address,
            long X1_address, 
            long X2_address,
            long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(X3_address == NULL) throw new NullPointerException("Tensor X3 is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.add_div2D_row(Y_address, 
                X1_address,
                X2_address, 
                X3_address, row_lengthv, 
                alpha, beta, gamma, delta, 
                lengthv, width, stride);
    }
    
    public Syncer add_div2D_field(long Y_address,
            long X1_address,
            long X2_address,
            long X3_address, int field_length,
            float alpha, float beta, float gamma, float delta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(X3_address == NULL) throw new NullPointerException("Tensor X3 is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.add_div2D_field(Y_address, 
                X1_address, 
                X2_address, 
                X3_address, row_lengthv, 
                alpha, beta, gamma, delta, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, sqrt"> 
    public Syncer sign2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sign2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer ceil2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.ceil2D(Y_address,
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
   
    public Syncer floor2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.floor2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs">
    public Syncer abs2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.abs2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer abs2D_deltaX(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.abs2D_deltaX(deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer zero_nan2D(long Y_address, 
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.zero_nan2D(Y_address,
                X_address,
                lengthv, width, stride);
    }
    
    public Syncer sqrt2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sqrt2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer sqrt_quadratic_dual2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
         if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sqrt_quadratic_dual2D(Y_address, 
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="min, max, clip">
    //<editor-fold defaultstate="collapsed" desc="min, min_dual">
    public Syncer min2D(long Y_address,
            float alpha, long X_address, float beta, 
            float vmin,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.min2D(Y_address, 
                alpha, X_address, beta, vmin, 
                lengthv, width, stride);
    }
    
    public Syncer min_dual2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.min_dual2D(Y_address,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max, max_dual">
    public Syncer max2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.max2D(Y_address, 
                alpha, X_address, beta, vmax,
                lengthv, width, stride);
    }
    
    public Syncer max_dual2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            func_param_check(lengthv, width, stride);
        }
        return base.max_dual2D(Y_address,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer clip2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        if(vmin > vmax) {float t = vmin; vmin = vmax; vmax = t;}  
        return base.clip2D(Y_address, 
                alpha, X_address, beta, vmin, vmax,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    public Syncer exp2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.exp2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP-log">
    public Syncer log2D(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.log2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer log2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.log2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>          
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    public Syncer relu2D(long Y_address,
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer relu2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D_deltaX_v1(deltaX_address, 
                deltaY_address, Y_address, 
                lengthv, width, stride);
    }
    
     public Syncer relu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.relu2D_deltaX_v2(deltaX_address, 
                deltaY_address, X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    public Syncer leakyRelu2D(long Y_address,
            long X_address, float k, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(k<0 || k>=1) throw new IllegalArgumentException("Leaky Relu: k must belong to [0,1)");
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D(Y_address,
                X_address, k, 
                lengthv, width, stride);
    }
    
    public Syncer leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(k<0 || k>=1) throw new IllegalArgumentException("Leaky Relu: k must belong to [0,1)");
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, k, 
                lengthv, width, stride);
    }
    
    public Syncer leakyRelu2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(k<0 || k>=1) throw new IllegalArgumentException("Leaky Relu: k must belong to [0,1)");
            func_param_check(lengthv, width, stride);
        }
        return base.leakyRelu2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address, k,
                lengthv, width, stride);
    }        
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    public Syncer elu2D(long Y_address,
            long X_address, float alpha, float k,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(alpha < 0) throw new IllegalArgumentException("Elu: alpha must >=0");
            if(k < 0) throw new IllegalArgumentException("Elu: negative_slope(k) must >= 0");
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D(Y_address, 
                X_address, alpha, k, 
                lengthv, width, stride);
    }
    
    public Syncer elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float k,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            if(alpha < 0) throw new IllegalArgumentException("Elu: alpha must >=0");
            if(k < 0) throw new IllegalArgumentException("Elu: negative_slope(k) must >= 0");
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, alpha, k, 
                lengthv, width, stride);
    }
    
    public Syncer elu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float k,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(alpha < 0) throw new IllegalArgumentException("Elu: alpha must >=0");
            if(k < 0) throw new IllegalArgumentException("Elu: negative_slope(k) must >= 0");
            func_param_check(lengthv, width, stride);
        }
        return base.elu2D_deltaX_v2(deltaX_address,
                deltaY_address, 
                X_address, alpha, k, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    public Syncer softPlus2D(long Y_address,
            long X_address, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D(Y_address, X_address, 
                lengthv, width, stride);
    }
    
    public Syncer softPlus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D_deltaX_v1(deltaX_address, 
                deltaY_address, 
                Y_address, 
                lengthv, width, stride);
    }
    
    public Syncer softPlus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.softPlus2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    public Syncer tanh2D(long Y_address,
            long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D(Y_address, X_address,
                lengthv, width, stride);
    }
    
    public Syncer tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, 
                lengthv, width, stride);
    }
    
    public Syncer tanh2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tanh2D_deltaX_v2(deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    public Syncer sigmoid2D(long Y_address,
            long X_address,
           int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D(Y_address, 
                X_address, 
                lengthv, width, stride);
    }
    
    public Syncer sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
    }
    
    public Syncer sigmoid2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    public Syncer softmax2D(long Y_address, 
            long X_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax2D(Y_address,
                X_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer softmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logsoftmax">  
    public Syncer logsoftmax2D(long Y_address, 
            long X_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.logsoftmax2D(Y_address,
                X_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer logsoftmax2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, int row_length,//lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.logsoftmax2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin">
    public Syncer sin2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sin2D(Y_address,
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer sin2D_deltaX(long deltaX_address, 
            long deltaY_address,
            long X_address, float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.sin2D_deltaX(deltaX_address,
                deltaY_address,
                X_address, alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tan">
    public Syncer tan2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tan2D(Y_address, 
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer tan2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.tan2D_deltaX(deltaX_address, 
                deltaY_address, 
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin">
    public Syncer halfSin2D(long Y_address,
            float Amp, float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.halfSin2D(Y_address, 
                Amp, alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer halfSin2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float Amp, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.halfSin2D_deltaX(deltaX_address, 
                deltaY_address, 
                Y_address, Amp, alpha,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin2D">
    public Syncer arcsin2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arcsin2D(Y_address,
                alpha, X_address, beta,
                lengthv, width, stride);
    }
    
    public Syncer arcsin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arcsin2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">        
    public Syncer arctan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arctan2D(Y_address, 
                alpha, X_address, beta, 
                lengthv, width, stride);
    }
    
    public Syncer arctan2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is NULL");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is NULL");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is NULL");
            func_param_check(lengthv, width, stride);
        }
        return base.arctan2D_deltaX(deltaX_address,
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    public Syncer L1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L1_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer L1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y  is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L1_2D_deltaYh(deltaYh_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    public Syncer L2_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L2_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer L2_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.L2_2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: SmoothL1">
    public Syncer smoothL1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.smoothL1_2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer smoothL1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.smoothL1_2D_deltaYh(deltaYh_address,
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    public Syncer binaryCrossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.binaryCrossEntropy2D(L_address, 
                Y_address, Yh_address, 
                alpha, beta,
                lengthv, width, stride);
    }
     
    public Syncer binaryCrossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.binaryCrossEntropy2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    public Syncer sigmoid_binaryCrossEntropy2D(long L_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid_binaryCrossEntropy2D(L_address,
                Y_address, X_address, 
                alpha, beta,
                lengthv, width, stride);
    }
    
    public Syncer sigmoid_binaryCrossEntropy2D_deltaX(long deltaX_address,
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X(Yh = sigmoid(X)) is null");
            func_param_check(lengthv, width, stride);
        }
        return base.sigmoid_binaryCrossEntropy_deltaX(deltaX_address,
                Y_address, X_address, 
                alpha, beta,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    public Syncer crossEntropy2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(L_address  == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.crossEntropy2D(L_address, 
                Y_address, Yh_address, 
                lengthv, width, stride);
    }
     
    public Syncer crossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(deltaYh_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Yh_address == NULL) throw new NullPointerException("Tensor Yh is null");
            func_param_check(lengthv, width, stride);
        }
        return base.crossEntropy2D_deltaYh(deltaYh_address,
                Y_address, Yh_address,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax_crossEntropy">
    public Syncer softmax_crossEntropy2D(long L_address,
            long Y_address, long X_address, int row_length, //lengthv = field_length * row_lengthv
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;

        if(check) {
            if(L_address == NULL) throw new NullPointerException("Tensor L is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax_crossEntropy2D(L_address,
                Y_address, X_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer softmax_crossEntropy2D_deltaX(long deltaX_address,
            long Y_address, long X_address, int row_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaYh is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X(Yh = softmax(X)) is null");
            func_param_check_softmax(lengthv, row_length, row_lengthv, width);
        }
        return base.softmax_crossEntropy2D_deltaX(deltaX_address,
                Y_address, X_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer affine2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        
        return base.affine2D(Y_address, 
                X_address, 
                A_address, B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    public Syncer affine2D_deltaA_v1(long deltaA_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaA_v1(deltaA_address, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer affine2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaAB_v1(
                deltaA_address,//result0
                deltaB_address,//tesult1
                deltaY_address,
                Y_address,
                A_address, B_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer affine2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long X_address,//(V2: X for Affine || V1: Y for Norm)
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.affine2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                field_length, row_lengthv, 
                width, stride);
     }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                row_lengthv, lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                row_lengthv, lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, int row_lengthv,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps, 
                A_address,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public Syncer sqBatchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address,
                dX_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    
   public Syncer sqBatchNorm2D_deltaAB_v2(
           long deltaA_address,//result0
           long deltaB_address,//result1
           long deltaY_address, long X_address, //V2: holdX(), X is not changed
           long X_mean_address, long X_sqmean_address, float eps,
           int row_lengthv, int lengthv, int width)
   {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_squmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
       
        return base.sqBatchNorm2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv,
                width, stride);
   }
   //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public Syncer sqBatchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        
        return base.sqBatchNorm2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, 
                Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer sqBatchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.sqBatchNorm2D_gradients_v2(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                row_lengthv, lengthv, width, stride);
    }
    
    public Syncer batchNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_row(lengthv, row_lengthv, width, stride);
        }
        return base.batchNorm2D(Y_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                A_address, B_address, 
                row_lengthv, lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address,
                Y_address, 
                X_var_address, eps,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address, 
                X_var_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps, 
                A_address,
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>  
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    public Syncer batchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address,
                dX_address,
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    
   public Syncer batchNorm2D_deltaAB_v2(
           long deltaA_address,//result0
           long deltaB_address,//result1
           long deltaY_address,
           long X_address,//V2: holdX(), X is not changed
           long X_mean_address, long X_var_address, float eps,
           int row_lengthv, int lengthv, int width)
   {
        int stride = ((width + 3) >> 2) << 2;
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_deltaAB_v2(
                deltaA_address,//result0
                deltaB_address,//result1
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv,
                width, stride);
   }
   //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    public Syncer batchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_gradients_v1(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2
                deltaY_address, 
                Y_address, 
                X_var_address, eps, 
                A_address, B_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer batchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int row_lengthv, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int field_length = lengthv / row_lengthv;
        if(check) {
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_var_address == NULL) throw new NullPointerException("Tensor X_var is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.batchNorm2D_gradients_v2(
                deltaX_address,//result0
                deltaA_address,//result1
                deltaB_address,//result2 
                deltaY_address, X_address,
                X_mean_address, X_var_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D(Y_address,
                X_address, 
                X_mean_address,
                X_sqmean_address, eps, row_lengthv, 
                lengthv, width, stride);
    }
     
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D(Y_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deltaX">
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v1(deltaX_address,
                deltaY_address, 
                Y_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v2(deltaX_address, 
                deltaY_address,
                X_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): deltaX">
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, 
            long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(B_address == NULL) throw new NullPointerException("Tensor B is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v1(deltaX_address, 
                deltaY_address,
                Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;  
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(deltaX_address == NULL) throw new NullPointerException("Tensor deltaX is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_means is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(A_address == NULL) throw new NullPointerException("Tensor A is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaX_v2(deltaX_address,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, 
                field_length, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer layerNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address,
            long dX_address, //V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps, 
            int field_length, int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaA_v2(deltaA_address, 
                deltaY_address, 
                dX_address, 
                X_mean_address, 
                X_sqmean_address, eps, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): {deltaA, deltaB}">
    public Syncer layerNorm2D_deltaAB_v2(long deltaA_address, long deltaB_address,
            long deltaY_address, long X_address, //V2: holdX(), X is not changed
            long X_mean_address,
            long X_sqmean_address, float eps, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;//lengthv = X.lengthv
        if(check) {
            if(deltaA_address == NULL) throw new NullPointerException("Tensor deltaA is null");
            if(deltaB_address == NULL) throw new NullPointerException("Tensor deltaB is null");
            if(deltaY_address == NULL) throw new NullPointerException("Tensor deltaY is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(X_mean_address == NULL) throw new NullPointerException("Tensor X_mean is null");
            if(X_sqmean_address == NULL) throw new NullPointerException("Tensor X_sqmsean is null");
            if(eps < 0) throw new IllegalArgumentException("eps must >= 0");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.layerNorm2D_deltaAB_v2(deltaA_address, deltaB_address, 
                deltaY_address, X_address,
                X_mean_address,
                X_sqmean_address, eps,
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
    public Syncer onehot2D_row_int8(long Y_address,
            long X_address, 
            float alpha, float beta, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.onehot2D_row_int8(Y_address, 
                X_address, alpha, beta, row_lengthv,
                lengthv, width, stride);
    }
    
    public Syncer onehot2D_row_int32(long Y_address,
            long X_address, 
            float alpha, float beta, int field_length,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = lengthv / field_length;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check_field(lengthv, field_length, row_lengthv, width, stride);
        }
        return base.onehot2D_row_int32(Y_address,
                X_address, alpha, beta, row_lengthv, 
                lengthv, width, stride);
    }
    
    public Syncer pix2tensor2D(long Y_address, 
            long X_address, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.pix2tensor2D(Y_address,
                X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    public Syncer sgd(long W_address,//Y += alpha * Xs[i] + beta, [from 1 to X.lenght]
            long[] gradients, float lr, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check){
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            Vector.requireNonNull(gradients, "W.gradients");
            func_param_check(lengthv, width, stride);
        }
        return base.sgd2D(W_address, gradients, lr,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    public Syncer sgdmn(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(momentum < 0) throw new IllegalArgumentException("momentum < 0");
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D(W_address,
                V_address, momentum, dampen, nesterov, 
                deltaW_address, lr, 
                lengthv, width, stride);
    }
    
   public Syncer sgdmn(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            int lengthv, int width)
    {
         int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(momentum < 0) throw new IllegalArgumentException("momentum < 0");
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D(W_address,
                V_address, momentum, dampen, nesterov, 
                gradients, lr, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN-decay">
    public Syncer sgdmn_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(momentum < 0) throw new IllegalArgumentException("momentum < 0");
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D_decay(W_address,
                V_address, momentum, dampen, nesterov, 
                deltaW_address, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer sgdmn_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
         int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(momentum < 0) throw new IllegalArgumentException("momentum < 0");
            func_param_check(lengthv, width, stride);
        }
        return base.sgdmn2D_decay(W_address,
                V_address, momentum, dampen, nesterov, 
                gradients, lr,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D(W_address, 
                V_address, a1, a2,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D(W_address, 
                V_address, a1, a2,
                gradients, lr_t,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Momentum-decay">
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <=0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <=0 ) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D_decay(W_address,
                V_address, a1, a2,
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <=0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <=0 ) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.momentum2D_decay(W_address, 
                V_address, a1, a2,
                gradients, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D(W_address,
                S_address, a1, a2, eps_t,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients , float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D(W_address,
                S_address, a1, a2, eps_t,
                gradients, lr_t,
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop-decay">
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D_decay(W_address,
                S_address, a1, a2, eps_t,
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients , float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.rmsprop2D_decay(W_address,
                S_address, a1, a2, eps_t,
                gradients, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adam">
    //<editor-fold defaultstate="collapsed" desc="adam2D_type2">
    public Syncer adam2D_type2(long W_address,
            long V_address, float a1, float a2, float Uv, 
            long S_address, float b1, float b2, float eps, float Us,
            long deltaW_address, float lr,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            if(a1 + a2 < 0.99999f || a1 + a2 > 1.00001f) throw new IllegalArgumentException("a1 + a2 != 1");
            if(b1 + b2 < 0.99999f || b1 + b2 > 1.00001f) throw new IllegalArgumentException("b1 + b2 != 1");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_type2(W_address, 
                V_address, a1, a2, Uv,
                S_address, b1, b2, eps, Us, 
                deltaW_address, lr,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
       public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <=0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <=0 ) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <=0 ) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <=0 ) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                gradients, lr_t,
                lengthv, width, stride);
    }
       
   
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam-decay">
     public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <0 ) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <0 ) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <0 ) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
    }
    
    public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <=0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <=0 ) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <=0 ) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <=0 ) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adam2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                gradients, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
     }
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D(W_address,
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                lengthv, width, stride);
    }
    
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D(W_address,
                V_address, a1, a2,
                S_address, b1, eps,
                gradients, lr_t, 
                lengthv, width, stride);
     }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax-decay">
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D_decay(W_address,
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamax2D_decay(W_address, 
                V_address, a1, a2,
                S_address, b1, eps,
                gradients, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
     }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
     public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamW2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t, lr,
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamW2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                gradients, lr_t, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
     }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(G_address == NULL) throw new NullPointerException("Tensor G is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            if(c1 <= 0) throw new IllegalArgumentException("c1 must be postitive");
            if(c2 <= 0) throw new IllegalArgumentException("c2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamod2D(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                lengthv, width, stride);
    }
    
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            for(int i=0; i< gradients.length; i++) 
                if(gradients[i] == NULL) throw new NullPointerException(
                        "deltaW.gradients[]" + i +" is null");
            
            if(lengthv < width) throw new IllegalArgumentException();
            if(width <= 0) throw new IllegalArgumentException();
            if(lengthv % stride != 0) throw new IllegalArgumentException();
            
            if(a1 <0 ) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <0 ) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <0 ) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <0 ) throw new IllegalArgumentException("b2 must be postitive");
            if(c1 <0 ) throw new IllegalArgumentException("c1 must be postitive");
            if(c2 <0 ) throw new IllegalArgumentException("c2 must be postitive");
        }

        return base.adamod2D(W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                G_address, c1, c2,
                gradients, lr_t, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod-decay">
    public Syncer adamod2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            if(G_address == NULL) throw new NullPointerException("Tensor G is null");
            if(deltaW_address == NULL) throw new NullPointerException("Tensor deltaW is null");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            if(c1 <= 0) throw new IllegalArgumentException("c1 must be postitive");
            if(c2 <= 0) throw new IllegalArgumentException("c2 must be postitive");
        }
        return base.adamod2D_decay(W_address,
                V_address, a1, a2,
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    
    public Syncer adamod2D_decay(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(W_address == NULL) throw new NullPointerException("Tensor W is null");
            if(V_address == NULL) throw new NullPointerException("Tensor V is null");
            if(S_address == NULL) throw new NullPointerException("Tensor S is null");
            Vector.requireNonNull(gradients, "W.gradients");
            if(a1 <= 0) throw new IllegalArgumentException("a1 must be postitive");
            if(a2 <= 0) throw new IllegalArgumentException("a2 must be postitive");
            if(b1 <= 0) throw new IllegalArgumentException("b1 must be postitive");
            if(b2 <= 0) throw new IllegalArgumentException("b2 must be postitive");
            if(c1 <= 0) throw new IllegalArgumentException("c1 must be postitive");
            if(c2 <= 0) throw new IllegalArgumentException("c2 must be postitive");
            func_param_check(lengthv, width, stride);
        }
        return base.adamod2D_decay(W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                gradients, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    protected int next_seed() { return exr.nextInt(); } 
    
    public Syncer bernouli2D(long X_address, 
            float p, float v1, float v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException("p must belong to [0,1]");
            func_param_check(lengthv, width, stride);
        }
        return base.bernouli2D(X_address,
                next_seed(),
                p, v1, v2, 
                lengthv, width, stride);
    }
    
    public Syncer bernouli2D_mul(long Y_address, long R_address,
            long X_address, 
            float p, float v1, float v2,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(R_address == NULL) throw new NullPointerException("Tensor R is null");
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException("p must belong to [0,1]");
            func_param_check(lengthv, width, stride);
        }
        return base.bernouli_mul2D(Y_address, R_address,
                X_address, 
                next_seed(), 
                p, v1, v2,
                lengthv, width, stride);
    }
    
    public Syncer uniform2D(long X_address, 
            float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        if(vmin < vmax) {float t = vmin; vmin = vmax; vmax = t;}
        return base.uniform2D(X_address, 
                next_seed(), 
                vmin, vmax, 
                lengthv, width, stride);
    }
     
    public Syncer sparse_uniform2D(long X_address, 
            float p, float vmin, float vmax,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(p<0 || p>1) throw new IllegalArgumentException("p must belong to [0,1]");
            func_param_check(lengthv, width, stride);
        }
        if(vmin < vmax) {float t = vmin; vmin = vmax; vmax = t;}
        return base.sparse_uniform2D(X_address,
                next_seed(), next_seed(),
                p, vmin, vmax, 
                lengthv, width, stride);
    }
    
    public Syncer gaussian2D(long X_address,
            float mu, float sigma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(sigma < 0) throw new IllegalArgumentException("Sigma: the standard deviation sigma must be positive");
            func_param_check(lengthv, width, stride);
        }
        return base.gaussian2D(X_address, 
                next_seed(), next_seed(), 
                mu, sigma, 
                lengthv, width, stride);
    }
    
    public Syncer sparse_gaussian2D(long X_address,
            float p, float mu, float sigma,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(sigma < 0) throw new IllegalArgumentException("Sigma: standard deviation sigma must be positive");
            if(p<0 || p>1) throw new IllegalArgumentException("p must belong to [0, 1]");
            func_param_check(lengthv, width, stride);
        }
        return base.sparse_gaussian2D(X_address, 
                next_seed(), next_seed(), next_seed(), 
                p, mu, sigma,
                lengthv, width, stride);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="param_check">
    private void field_reduce_param_check(int length, int row_length, int width) {
        if(length < width) throw new IllegalArgumentException();
        if(width <= 0) throw new IllegalArgumentException();
        if(length % row_length != 0) throw new IllegalArgumentException();
        if(row_length % width != 0) throw new IllegalArgumentException(); 
    }
    
    private void row_reduce_param_check(int field_length, int row_length, int width){
        if(field_length <= 0) throw new IllegalArgumentException();
        if(width <= 0) throw new IllegalArgumentException();
        if(row_length % width != 0) throw new IllegalArgumentException();
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    public Result<Float> straight_linear(long X_address,
            float alpha, float beta, 
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_linear(X_address, alpha, beta,  
                lengthv, width, stride);
    }
    
    public Result<Float> straight_quadratic(long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_quadratic(X_address, alpha, beta, gamma, 
                lengthv, width, stride);
    }
    
    public Result<Float> straight_max(long X_address,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_max(X_address, 
                lengthv, width, stride);
    }
    
    public Result<Float> straight_min(long X_address,
            int lengthv, int width) 
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_min(X_address, 
                lengthv, width, stride);
    }
    
    public IndexedResult<Float> straight_max_indexed(long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_max_indexed(X_address,
                lengthv, width, stride);
    }
     
    public IndexedResult<Float> straight_min_indexed(long X_address,
            int lengthv, int width)
    {
        int stride = ((width + 3) >> 2) << 2;
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            func_param_check(lengthv, width, stride);
        }
        return base.straight_min_indexed(X_address, 
                lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    //<editor-fold defaultstate="collapsed" desc="field linear">
    public Syncer field_linear(long Y_address, 
            long X_address, float alpha, float beta, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_linear(Y_address,
                X_address, alpha, beta,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_linear_dual(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int length, int row_length, int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        
        return base.field_linear_dual(Y_address,
                X1_address, X2_address,
                alpha, beta, gamma, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic">
    public Syncer field_quadratic(long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_quadratic(Y_address,
                X_address, alpha, beta, gamma, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_quadratic_dual(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int length, int row_length, int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_quadratic_dual(Y_address, 
                X1_address, X2_address, 
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_linear_quadratic & var, std">
    public Syncer field_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int length, int row_length, int width) 
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_linear_quadratic(Y1_address, Y2_address, 
                X_address, 
                alpha1, beta1, 
                alpha2, beta2, gamma2, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_var(long var_address, 
            long mean_address, long squareMean_address,
            long X_address,
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(var_address == NULL) throw new NullPointerException("Tensor var is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(squareMean_address == NULL) throw new NullPointerException("Tensor squareMean is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_var(var_address,
                mean_address, squareMean_address, 
                X_address,
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer field_std(long stddev_address, 
            long mean_address, long sqmean_address,
            long X_address,
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(stddev_address == NULL) throw new NullPointerException("Tensor stddev is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(sqmean_address == NULL) throw new NullPointerException("Tensor squareMean is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_std(stddev_address,
                mean_address, sqmean_address, 
                X_address,
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field max, min">
    public Syncer field_max(long Y_address,
            long X_address, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_max(Y_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer field_min(long Y_address,
            long X_address, 
            int length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_min(Y_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    
     public Syncer field_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int length, int row_length, int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_max_indexed(Y_address, Index_address,
                X_address, field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer field_min_indexed(long Y_address, long Index_address,
            long X_address, 
            int length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            field_reduce_param_check(length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int field_length = length / row_length;
        int row_lengthv = row_length / width * stride;
        return base.field_min_indexed(Y_address, Index_address,
                X_address, field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    //<editor-fold defaultstate="collapsed" desc="row linear">
    public Syncer row_linear(long Y_address, 
            long X_address, float alpha, float beta, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_linear(Y_address, 
                X_address, alpha, beta,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_linear_dual(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        
        return base.row_linear_dual(Y_address, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                field_length, 
                row_lengthv, width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic">
    public Syncer row_quadratic(long Y_address, 
            long X_address, float alpha, float beta, float gamma, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_quadratic(Y_address, 
                X_address, alpha, beta, gamma,
                field_length, row_lengthv,
                width, stride);
    }

    public Syncer row_quadratic_dual(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22, 
            float k1, float k2, float C, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X1_address == NULL) throw new NullPointerException("Tensor X1 is null");
            if(X2_address == NULL) throw new NullPointerException("Tensor X2 is null");
            if(Y_address  == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_quadratic_dual(Y_address,
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_lengthv, 
                width, stride);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic  & var & std">
    public Syncer row_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1, 
            float alpha2, float beta2, float gamma2,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y1_address == NULL) throw new NullPointerException("Tensor Y1 is null");
            if(Y2_address == NULL) throw new NullPointerException("Tensor Y2 is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_linear_quadratic(Y1_address, Y2_address,
                X_address, 
                alpha1, beta1,
                alpha2, beta2, gamma2, 
                field_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer row_var(long var_address,
            long mean_address, long squareMean_address,
            long X_address, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(var_address == NULL) throw new NullPointerException("Tensor var is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(squareMean_address == NULL) throw new NullPointerException("Tensor squareMean is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_lengthv  = ((field_length + 3) >> 2) << 2;
        return base.row_var(var_address, 
                mean_address, squareMean_address,
                X_address,
                field_length, field_lengthv, 
                row_length, row_lengthv, 
                width, stride);
    }
    
    public Syncer row_stddev(long stddev_address,
            long mean_address, long squareMean_address,
            long X_address, 
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(stddev_address == NULL) throw new NullPointerException("Tensor stddev is null");
            if(mean_address == NULL) throw new NullPointerException("Tensor mean is null");
            if(squareMean_address == NULL) throw new NullPointerException("Tensor squareMean is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        return base.row_stddev(stddev_address, 
                mean_address, squareMean_address,
                X_address,
                field_length, field_lengthv,
                row_length, row_lengthv,
                width, stride);
    }
    //</editor-fold> 
    //<editor-fold defaultstate="collapsed" desc="row max, min">
    public Syncer row_max(long Y_address, 
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_max(Y_address,
                X_address,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_min(long Y_address, 
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_min(Y_address,
                X_address,
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_max_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_max_indexed(Y_address, Index_address,
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    
    public Syncer row_min_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_length,
            int width)
    {
        if(check) {
            if(X_address == NULL) throw new NullPointerException("Tensor X is null");
            if(Y_address == NULL) throw new NullPointerException("Tensor Y is null");
            if(Index_address == NULL) throw new NullPointerException("Tensor Index<int32> is null");
            row_reduce_param_check(field_length, row_length, width);
        }
        int stride = ((width + 3) >> 2) << 2;
        int row_lengthv = row_length / width * stride;
        return base.row_min_indexed(Y_address, Index_address,
                X_address, 
                field_length, row_lengthv,
                width, stride);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
}

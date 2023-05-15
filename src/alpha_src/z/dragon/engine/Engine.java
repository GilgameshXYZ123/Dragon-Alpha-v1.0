 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import z.dragon.common.MemStatus;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Result.IndexedResult;
import z.dragon.engine.Syncer.ChainSyncer;
import z.dragon.engine.memp.Mempool;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;
import z.dragon.nn.unit.Unit;

/**
 *
 * @author Gilgamesh
 */
public class Engine implements MemStatus
{
    protected EngineCore core;
    protected boolean check = true;
    protected boolean sync = true;
    
    public static final float HALF_PI= (float) (0.5 * Math.PI);
    
    public Engine() {}
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Random random() { return core.exr; }
    public Engine random(Random random) { core.random(random); return this; }
    
    public EngineCore engineCore() { return core; }
    public Mempool mempool() { return core.mempool; } 
    public synchronized Engine engineCore(EngineCore core) {
        if(core == null) throw new NullPointerException("EngineCore is null");
        this.core = core;
        this.check = core.check;
        return this;
    }
    
    public EngineBase engineBase() { return core.base; }

    public boolean check() {return check;}
    public Engine check(boolean flag) { check = flag; core.check(flag); return this; }
    
    public boolean sync() { return sync; }
    public Engine sync(boolean flag) { this.sync = flag; return this; }
    
    @Override public long max_mem_size() { return core.max_mem_size(); }
    @Override public long total_mem_size() { return core.total_mem_size(); }
    @Override public long used_mem_size() {return core.used_mem_size_MB();}
    
    public Engine max_mem_size(long maxMemSize) { core.max_mem_size(maxMemSize); return this; }

    public long bufferedMemSize() {return core.buffered_mem_size();}
    public long bufferedMemSize_MB() {return core.buffered_mem_size_MB();}
    
    public String dataType() { return core.dataType(); }
    public String dataType_int32() { return core.dataType_int32(); }
    public String dataType_int8() { return core.dataType_int8(); }
    public long sizeOf_dataType() { return core.sizeOf_dataType(); }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append("\nsync = ").append(sync);
        sb.append("\ncheck = ").append(check);
        Mempool mempool = mempool();
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

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 37 * hash + Objects.hashCode(this.core);
        return hash;
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof Engine)) return false;
        Engine eg = (Engine) o;
        return Objects.equals(eg.core, core);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="built-int-checkers">
    private void require_dataType(Tensor[] Xs) { for(Tensor X : Xs) require_dataType(X); }
    private void require_dataType(Collection<Tensor> Xs) { for(Tensor X : Xs) require_dataType(X); }
    private void require_dataType(Tensor X) {
        if(!X.dataType.equals(core.dataType())) 
            throw new IllegalArgumentException("dataType != dataType");
        if(!this.equals(X.engine())) 
            throw new IllegalArgumentException("Invalid Engine");
    }
    
    private void require_int32(Tensor X) {
        if(!X.dataType.equals(core.dataType_int32()))
            throw new IllegalArgumentException("dataType != int32");
        if(!this.equals(X.engine())) 
            throw new IllegalArgumentException("Invalid Engine");
    }
    
    private void require_int8(Tensor X) {
        if(!X.dataType.equals(core.dataType_int8()))
            throw new IllegalArgumentException("dataType != int8");
        if(!this.equals(X.engine())) 
            throw new IllegalArgumentException("Invalid Engine");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: int32">
    public Tensor empty_int32_like(Tensor X) { return empty_int32(X.dim); }
    public Tensor empty_int32(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType_int32(), dim);
        ts.bindMemory(core.malloc_int32(ts.length_4x));
        
        int compare_length = ((ts.length << 2) + (1 << core.L_sizeof_datatype) - 1) >> core.L_sizeof_datatype;
        if(ts.mem_size >= compare_length) {//(1) -> (2)
            Syncer sc = core.memset_int32(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    
    public Tensor zeros_int32_like(Tensor X) { return zeros_int32(X.dim); }
    public Tensor zeros_int32(int... dim) { 
        Tensor ts = new Tensor(check, this, core.dataType_int32(), dim);
        ts.bindMemory(core.malloc_int32(ts.length_4x)); 
        Syncer sc = core.memset_int32(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }
    
    public Tensor tensor_int32_like(int[] value, Tensor X) { return tensor_int32(value, X.dim); }
    public Tensor tensor_int32(int[] value, int... dim) {
        negativeDim(value.length, dim);
        Tensor ts = this.empty_int32(dim).c();
        set_int32(ts, value);
        return ts;
    }
    
    public int[] valueOf_int32(Tensor X) {
        if(check) { require_int32(X); }
        
        if(X.ndim() == 1) { X.c(); return core.get1D_int32(X.address, X.length); }
        
        int width = X.lastDim(); X.c();
        return (width & 3) == 0 ? 
                core.get1D_int32(X.address, X.length)://no memory alignment
                core.get2D_int32(X.address, X.length / width, width);//[height, width]
    }
    
    @StrictSync
    public Tensor set_int32(Tensor X, int[] value) 
    {
        if(check) { 
            require_int32(X); 
            equals(value.length, "value<int[]>.length", X.length, "Tensor<int32> X.length");
        }
       
        if(X.ndim() == 1) core.set1D_int32(X.address, value);
        else {
            int width = X.lastDim(); 
            if((width & 3) == 0) core.set1D_int32(X.address, value);//no memory alignment
            else core.set2D_int32(X.address, value, width);//[height, width]
        }
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Tensor: int8"> 
    public Tensor emtpy_int8_like(Tensor X) { return empty_int8(X.dim); }
    public Tensor empty_int8(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType_int8(), dim);
        ts.bindMemory(core.malloc_int8(ts.length_4x));

        int compare_length = (ts.length + (1 << core.L_sizeof_datatype) - 1) >> core.L_sizeof_datatype;
        if(ts.mem_size >= compare_length) {//(1) -> (2)
            Syncer sc = core.memset_int8(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    
    public Tensor zeros_int8_like(Tensor X) { return zeros_int8(X.dim); }
    public Tensor zeros_int8(int... dim) { 
        Tensor ts = new Tensor(check, this, core.dataType_int8(), dim);
        ts.bindMemory(core.malloc_int8(ts.length_4x)); 
        Syncer sc = core.memset_int8(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }
    
    public Tensor tensor_int8_like(byte[] value, Tensor X) { return tensor_int8(value, X.dim); }
    public Tensor tensor_int8(byte[] value, int... dim) {
        negativeDim(value.length, dim);
        Tensor ts = this.empty_int8(dim).c();
        set_int8(ts, value);
        return ts;
    }
    
    public byte[] valueOf_int8(Tensor X) {
        if(check) { require_int8(X); }
        
        if(X.ndim() == 1) { X.c(); return core.get1D_int8(X.address, X.length); }
        
        int width = X.lastDim(); X.c();
        return (width & 3) == 0 ? 
                core.get1D_int8(X.address, X.length)://no memory alignment
                core.get2D_int8(X.address, X.length / width, width);//[height, width]
    }
    
    @StrictSync
    public Tensor set_int8(Tensor X, byte[] value) 
    {
        if(check) { 
            require_int8(X); 
            equals(value.length, "value<char[]>.length", X.length, "Tensor<int8> X.length");
        }
       
        if(X.ndim() == 1) core.set1D_int8(X.address, value);
        else {
            int width = X.lastDim(); 
            if((width & 3) == 0) core.set1D_int8(X.address, value);//no memory alignment
            else core.set2D_int8(X.address, value, width);//[height, width]
        }
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: create & delete">
    //<editor-fold defaultstate="collapsed" desc="negativeDim">
    private static void negativeDim(int length, int... dim) {//default: firstDim = -1
        for(int i=0; i<dim.length; i++) {
            if(dim[i] == -1) {
                int mul = -Vector.mul(dim);
                if(length % mul != 0) throw new IllegalArgumentException(
                        "Illegal dimension:" + Arrays.toString(dim));
                dim[i] = length / mul;
                return;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="checkMatrix">
    public static void checkMatrix(byte[][] mat, String name)
    {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++)
        {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(
                    name + "[" + i + "].length != width");
        }
    }
    
    public static void checkMatrix(int[][] mat, String name)
    {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++)
        {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(
                    name + "[" + i + "].length != width");
        }
    }
    
    public static void checkMatrix(float[][] mat, String name)
    {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++)
        {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(
                    name + "[" + i + "].length != width");
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="fast: Mat to Vector ">
    private static byte[] toByteVector(byte[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        byte[] vec = new byte[dim0 * dim1];
        
        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    
    private static int[] toIntVector(int[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        int[] vec = new int[dim0 * dim1];
        
        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    
    private static float[] toFloatVector(float[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        float[] vec = new float[dim0 * dim1];

        int index = 0;
        for(int i=0; i<dim0; i++) {
            System.arraycopy(mat[i], 0, vec, index, dim1);
            index += dim1;
        }
        return vec;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="create: empty(dim)">
    public Tensor empty_like(Tensor X) { return empty(X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor empty(int...dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x)); 
       
        //memset zero: (1) when tensor.mem_length > tensor.length_4x or (2) ts.isMemAligned.
        if(ts.mem_size > ts.length) {// (1) -> (2)
            Syncer sc = core.memset(ts.address, 0, ts.length_4x);
            if(sync) sc.sync(); else ts.setSyncer(sc);
        }
        return ts;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: constant(C, dim)">
    public Tensor zeros_like(Tensor X) { return zeros(X.dim); }
    public Tensor zeros(int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x)); 
        Syncer sc = core.memset(ts.address, 0, ts.length_4x);
        if(sync) sc.sync(); else ts.setSyncer(sc);
        return ts;
    }

    
    public Tensor ones_like(Tensor X) { return Engine.this.constant(1.0f, X.dim); }
    public Tensor ones(int... dim) { return Engine.this.constant(1.0f, dim); }
    
    public Tensor constant_like(float value, Tensor X) { return Engine.this.constant(value, X.dim);}
    @Passed("CudaFloat32EngieBase")
    public Tensor constant(float value, int... dim) {
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x));
        
        if(ts.mem_size > ts.length) {
            core.memset(ts.address, 0, ts.length_4x).sync(); 
        }
        return constant(ts, value);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(byte[], dim), byte = int8">
    public Tensor tensor_like(byte[] value, Tensor X) {  return tensor(value, X.dim);  }
    public Tensor tensor_like(float alpha, byte[] value, float beta, Tensor X) {  
        return tensor(alpha, value, beta, X.dim);  
    }
    
    public Tensor tensor(byte[] value, int... dim) { return tensor(1.0f, value, 0.0f, dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float alpha, byte[] value, float beta, int... dim) {
        negativeDim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, alpha, value, beta);
        return ts;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor pix2tensor(byte[][] pixels, int... dim) {
        if(check) { checkMatrix(pixels, "pixels"); }
        Tensor BX = tensor_int8(toByteVector(pixels), dim);
        Tensor Y = this.empty(dim).c();
        
        Syncer sc = core.pix2tensor2D(Y.address, BX.address, 
                Y.lengthv, Y.lastDim());
        sc.sync(); delete(BX);
        return Y;
    }
    
    public Tensor tensor(byte[][] values, int... dim) { return tensor(1.0f, values, 0.0f, dim); }
    public Tensor tensor(float alpha, byte[][] values, float beta, int... dim) {
        if(check) { checkMatrix(values, "batch_values"); }
        byte[] value = toByteVector(values);
        return tensor(alpha, value, beta, dim);
    }
    
    public Tensor onehot(byte[] labels, int num_class) {
        return onehot(labels, 1.0f, 0.0f, num_class);
    }
    public Tensor onehot(byte[] labels, float alpha, int num_class)  {
        float beta = (1.0f - alpha) / (num_class - 1);
        return onehot(labels, alpha, beta, num_class);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(byte[] labels, float alpha, float beta, int num_class) 
    {
        Tensor BX = this.tensor_int8(labels, labels.length);
        Tensor Y = this.empty(labels.length, num_class).c();
        
        Syncer sc = core.onehot2D_row_int8(Y.address, BX.address, 
                alpha, beta, BX.length, //IX.length = field_length
                Y.lengthv, Y.lastDim());
        sc.sync(); delete(BX);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(int[], dim), byte = int32">
    public Tensor tensor_like(int[] value, Tensor X) {  return tensor(value, X.dim);  }
    public Tensor tensor_like(float alpha, int[] value, float beta, Tensor X) {  
        return tensor(alpha, value, beta, X.dim);  
    }
    
    public Tensor tensor(int[] value, int... dim) { return tensor(1.0f, value, 0.0f, dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float alpha, int[] value, float beta, int... dim) {
        negativeDim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, alpha, value, beta);
        return ts;
    }
    
    public Tensor tensor(int[][] values, int... dim) { return tensor(1.0f, values, 0.0f, dim); }
    public Tensor tensor(float alpha, int[][] values, float beta, int... dim) {
        if(check) { checkMatrix(values, "batch_values"); }
        int[] value = toIntVector(values);
        return tensor(alpha, value, beta, dim);
    }
    
    public Tensor onehot(int[] labels, int num_class) {
        return onehot(labels, 1.0f, 0.0f, num_class);
    }
    public Tensor onehot(int[] labels, float alpha, int num_class)  {
        float beta = (1.0f - alpha) / (num_class - 1);
        return onehot(labels, alpha, beta, num_class);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(int[] labels, float alpha, float beta, int num_class) 
    {
        Tensor IX = this.tensor_int32(labels, labels.length);
        Tensor Y = this.empty(labels.length, num_class).c();
        
        Syncer sc = core.onehot2D_row_int32(Y.address, IX.address, 
                alpha, beta, IX.length, //IX.length = field_length
                Y.lengthv, Y.lastDim());
        sc.sync(); IX.delete();
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(float[], dim)">
    public Tensor tensor_like(float[] value, Tensor X) { return tensor(value, X.dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float[] value, int... dim) {
        negativeDim(value.length, dim);
        Tensor ts = this.empty(dim).c();
        set(ts, value);
        return ts;
    }
    
    public Tensor tensor(float[][] values, int... dim) {
        if(check) { checkMatrix(values, "batch_values");}
        float[] value = toFloatVector(values);
        return tensor(value, dim);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="create: tensor(Tensor)">
    public Tensor tensor(Tensor value, int... dim) {
        if(dim == null || dim.length == 0) return copy(value);
        
        negativeDim(value.length, dim);
        
        Tensor ts = new Tensor(check, this, core.dataType(), dim);
        ts.bindMemory(core.malloc(ts.length_4x));
        if(ts.mem_size > ts.length) {
            core.memset(ts.address, 0, ts.length_4x).sync(); 
        }
        return set(ts, value);
    }

    public Tensor copy(Tensor X) {
        Tensor ts = this.empty(X.dim).c();
        Syncer sc = core.memcpy(ts.address, X.address, X.lengthv);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return ts;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="delete Tensor">
    public void delete(Tensor... Xs) {
        if(Xs == null || Xs.length == 0) return;
        for(Tensor ts : Xs) delete(ts); 
    }
    
    public void delete(Collection<Tensor> Xs) { 
        if(Xs == null || Xs.isEmpty()) return;
        Xs.forEach((ts) -> { delete(ts); }); 
    }
    
    public void delete(Tensor X)  {
        if(X == null) return;
        if(X.eg != this) { X.eg.delete(X); return; }
        
        //release the memory of the tensor
        //must wait until the compute of the tensor is completed
        //if you delete a tensor participating in a computation, it may effects other tensor
        core.free(X.c().mem_size, X.address);
        
        synchronized (X) {
            if(X.grad != null) { delete(X.grad); X.grad = null; }//X.clear_grad
            if(X.carrier != null) {//clear X.carrier
                if(!X.carrier.isEmpty()) {
                    X.carrier.forEach((ts) -> { if(ts != X) delete(ts); });
                    X.carrier.clear();
                }
                X.carrier = null;
            }
        }
        
        X.address = 0L;
        X.dim = null; 
        X.syncer = null;
        X.trace = null;
        X.mod_counter = null;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="delete Parameter">
    public void delete(Parameter... params) {
        if(params == null || params.length == 0) return;
        for(Parameter ts : params) delete(ts); 
    }
    
    public void delete(Parameter param)  {
        if(param == null) return;
        delete(param.tensor);
        if(!param.grads.isEmpty()) {//param.clear_grads()
            param.grads.forEach((Tensor g) -> { delete(g); });
            param.grads.clear();
        }
    }    
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor: set & valueOf">
    //<editor-fold defaultstate="collapsed" desc="constant(C)">
    public Tensor zero(Tensor X) {
        if(check) { require_dataType(X); }
        
        Syncer sc = core.memset(X.address, 0, X.length_4x);
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public Tensor constant(Tensor X, float value) {//constant value setter
        if(check) { require_dataType(X); }
        Syncer sc;
        if(X.ndim() == 1) sc = core.set1D(X.address, value, X.length);
        else {
            int width = X.lastDim();
            sc = ((width & 3) == 0?
                    core.set1D(X.address, value, X.length):
                    core.set2D(X.address, value, X.length / width, width));
        }
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(byte[]), byte=int8">
    @StrictSync
    public Tensor set(Tensor X, byte[] value) { return set(X, 1.0f, value, 0.0f); }
    public Tensor set(Tensor X, float alpha, byte[] value, float beta) 
    {
        if(check) { equals(value.length, "value<byte[]>.length", X.length, "X.length"); } 
        Tensor BX = this.tensor_int8(value, X.dim).c();
        Syncer sc = core.linear2D_int8_to_dtype(X.address, 
                alpha, BX.address, beta,
                X.lengthv, X.lastDim()); 
        sc.sync(); delete(BX);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(int[])", int=int32">
    @StrictSync
    public Tensor set(Tensor X, int[] value) { return set(X, 1.0f, value, 0.0f); }
    public Tensor set(Tensor X, float alpha, int[] value, float beta) 
    {
        if(check) { equals(value.length, "value<int[]>.length", X.length, "X.length"); } 
        Tensor IX = this.tensor_int32(value, X.dim).c();
        
        Syncer sc = core.linear2D_int32_to_dtype(X.address, 
                alpha, IX.address, beta, 
                X.lengthv, X.lastDim());
        sc.sync(); delete(IX);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(float[])">
    @StrictSync
    public Tensor set(Tensor X, float[] value) {
        if(check) { 
            require_dataType(X);
            equals(value.length, "value<float[]>.length", X.length, "X.length"); 
        } 
        
        if(X.ndim() == 1) core.set1D(X.address, value);
        else {
            int width = X.lastDim();
            if((width & 3) == 0) core.set1D(X.address, value);//no memory alignment
            else core.set2D(X.address, value, X.lastDim());
        }
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="set(Tensor)">
    public Tensor set(Tensor X, Tensor value) 
    {
        if(X.address == value.address) return X;
        if(check) {
            require_dataType(X);
            require_dataType(value);
            equals(X.length, "X.length", value.length, "value.length");
        }
        
        Syncer sc;//ts.length != value.length
        if(X.valueStrucEquals(value)) {//directly copy, All at once
            sc = core.memcpy(X.address, value.address, X.lengthv);
            if(sync) sc.sync(); else X.setSyncer(sc);
            return X;
        }
        
        int value_ndim = value.ndim(), ts_ndim = X.ndim();
        if(value_ndim > 1 && ts_ndim > 1) {//ND to ND, no need to memset(0)
            int src_width = value.lastDim(), dst_width = X.lastDim();
            sc = core.setFrom2Dto2D(
                    value.address, value.length / src_width, src_width,
                    X.address    , X.length     / dst_width, dst_width);
        }
        else if(value_ndim == 1) {//1D to ND
            int dst_width = X.lastDim();
            sc = core.setFrom1Dto2D(
                    value.address, value.length,
                    X.address, X.length / dst_width, dst_width);
        }
        else{//X.ndim == 1, ND to 1D
            int src_width = value.lastDim();
            sc = core.setFrom2Dto1D(
                    value.address, value.length / src_width, src_width,
                    X.address, X.length);
        }
        
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    @StrictSync
    public Tensor set(Tensor X, String line) {
        float[] value; try { value = Vector.toFloatVector(line); }
        catch(Exception e) { throw new RuntimeException(e); }
        return set(X, value);
    }
    
    public static boolean SET_STRING_LINES_PRINT = false;
    
    @StrictSync
    public Tensor set(Tensor X, List<String> lines) {
        float[] value = Vector.toFloatVector(lines, X.length);
        if(SET_STRING_LINES_PRINT) {//print if need to check
            float sp = Vector.samePercentAbsolute(value, X.value());
            System.out.println("sp = " + sp);
        }
        return set(X, value);
    }
    
    public void set(Tensor X, StateValue value, boolean partial, String msg) {
        if(value == null && !partial) throw new RuntimeException(msg);
        if(value != null) try { set(X, value.toStringLines()); }
        catch(Exception e) {
            throw new RuntimeException(msg, e);
        }
    }
            
    @StrictSync
    public float[] valueOf(Tensor ts)  
    {
        if(ts.ndim() == 1) { ts.c(); return core.get1D(ts.address, ts.length); }
        int width = ts.lastDim(); ts.c();
        return (width & 3) == 0 ? 
                core.get1D(ts.address, ts.length)://no memory alignment
                core.get2D(ts.address, ts.length / width, width);//[height, width]
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="Normal  MatMul(2D, 2D)">
    //<editor-fold defaultstate="collapsed" desc="matMul">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul(Tensor A, Tensor B)
    {
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];
        int dimB[] = B.dim, BH = dimB[0], BW = dimB[1];
        
        if(check) {
            require_dataType(A); 
            require_dataType(B); 
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(AW, "A.width", BH, "B.height");//K = AW = BH
        }
        
        Tensor C = this.empty(AH, BW).c();
        Syncer sc = core.matMul(C.address, A.address, B.address, AH, BW, dimA[1]);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
 
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul(Tensor C, Tensor A, Tensor B)
    {
        int dimA[] = A.dim, dimB[] = B.dim;
        
        if(check) {//[CH, CW] = [AH, AW] * [BH * BW]
            require_dataType(C);
            require_dataType(A); 
            require_dataType(B); 
            
            int AH = dimA[0], AW = dimA[1];
            int BH = dimB[0], BW = dimB[1];
            int dimC[] = C.dim, CH = dimC[0], CW = dimC[1];
            if(C.ndim() != 2) throw new IllegalArgumentException("Tensor C.ndim != 2");
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(CH, "C.height", AH, "A.height");//N = CH = AH
            equals(CW, "C.width",  BW, "B.width"); //M = CW = BW
            equals(AW, "A.width",  BH, "B.height");//K = AW = BH
        }
        
        Syncer sc = core.matMul(C.address, A.address, B.address, dimA[0], dimB[1], dimA[1]);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMul_biased(Tensor A, Tensor B, Tensor Bias)
    {
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];
        int dimB[] = B.dim, BH = dimB[0], BW = dimB[1];
        
        if(check) {//K = AW = BH
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(AW, "A.width", BH, "B.height");//K = AW = BH
            equals(Bias.lastDim(), "Bias.lastDim", BW, "B.width");
        }
        
        Tensor C = this.empty(AH, BW).c();
        Syncer sc = core.matMul_biased(C.address, A.address, B.address, AH, BW, dimA[1],
                Bias.address, C.lengthv);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT1">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT1(Tensor A, Tensor B)//C = (A^T)*B
    {
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];
        int dimB[] = B.dim, BH = dimB[0], BW = dimB[1];
        
        if(check) {//[CH, CW] = [AW, AH] * [BH * BW]
            require_dataType(A);
            require_dataType(B);
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(AH, "A.height", BH, "B.height");//K = AH = BH
        }
        
        Tensor C = this.empty(AW, BW).c();
        Syncer sc = core.matMulT1(C.address, A.address, B.address, AW, BW, dimA[0]);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT1(Tensor C, Tensor A, Tensor B)
    {
        int dimA[] = A.dim, dimB[] = B.dim;
        
        if(check) {//[CH, CW] = [AW, AH] * [BH * BW]
            require_dataType(C);
            require_dataType(A);
            require_dataType(B);
            
            int AH = dimA[0], AW = dimA[1];
            int BH = dimB[0], BW = dimB[1];
            int dimC[] = C.dim, CH = dimC[0], CW = dimC[1];
            if(C.ndim() != 2) throw new IllegalArgumentException("Tensor C.ndim != 2");
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(CH, "C.height", AW, "A.width"); //N = CH = AW
            equals(CW, "C.width",  BW, "B.width"); //M = CW = BW
            equals(AH, "A.height", BH, "B.height");//K = AH = BW
        }
        
        Syncer sc = core.matMulT1(C.address, A.address, B.address, dimA[1], dimB[1], dimA[0]);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT2">
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT2(Tensor A, Tensor B)//C = A*(B^T)
    {
        int dimA[] = A.dim, AH = dimA[0], AW = dimA[1];
        int dimB[] = B.dim, BH = dimB[0], BW = dimB[1];
        
        if(check) {//[CH, CW] = [AH, AW] * [BW * BH]
            require_dataType(A);
            require_dataType(B);
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            equals(AW, "A.width", BW, "B.width");//K = AW = BW
        }
        
        Tensor C = this.empty(AH, BH).c();
        Syncer sc = core.matMulT2(C.address, A.address, B.address, AH, BH, dimA[1]);
        if(sync) sc.sync(); else C.setSyncer(sc);//1101 1119
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor matMulT2(Tensor C, Tensor A, Tensor B)
    {
        int dimA[] = A.dim, dimB[] = B.dim;
        if(check) {//[CH, CW] = [AH, AW] * [BW * BH]
            require_dataType(C);
            require_dataType(A);
            require_dataType(B);
            
            int AH = dimA[0], AW = dimA[1];
            int BH = dimB[0], BW = dimB[1];
            int dimC[] = C.dim, CH = dimC[0], CW = dimC[1];
            if(C.ndim() != 2) throw new IllegalArgumentException("Tensor C.ndim != 2");
            if(A.ndim() != 2) throw new IllegalArgumentException("Tensor A.ndim != 2");
            if(B.ndim() != 2) throw new IllegalArgumentException("Tensor B.ndim != 2");
            if(CH != AH) throw new IllegalArgumentException();//
            if(CW != BH) throw new IllegalArgumentException();
            equals(CH, "C.height", AH, "A.height");//N = CH = AH
            equals(CW, "C.width",  BH, "B.height");//M = CW = BH
            equals(AW, "A.width",  BW, "B.width"); //K = AW = BW
        }
        
        Syncer sc = core.matMulT2(C.address, A.address, B.address, dimA[0], dimB[0], dimA[1]);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Batch   MatMul(3D+, 3D+)">
    //<editor-fold defaultstate="collapsed" desc="batchMatMul">
    public Tensor batchMatMul(Tensor A, Tensor B) {
        return batchMatMul(true, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMul(boolean likeA, Tensor A, Tensor B)
    {
        int ndimA = A.ndim(), ndimB = B.ndim();
        if(ndimA == 2 && ndimB == 2) return matMul(A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        if(check) {//A[N: AH, K: AW] * B[K: BH, M: BW] -> C[N: AH, M: BW]
            require_dataType(A);
            require_dataType(B);
            
            int BH = dimB[ndimB - 2];
            int batchB = B.length / (BH * BW);
            if(ndimA < 3) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB < 3) throw new IllegalArgumentException("B.ndim at least greater than 2");
            equals(AW, "A.width", BH, "B.height");//K = AW = BH
            equals(batchA, "A.batch", batchB, "B.batch");
        }
        
        int[] dimC;//A[N: AH, K: AW] * B[K: BH, M: BW] -> C[N: AH, M: BW]
        if(likeA) {//likeA
            dimC = new int[ndimA];
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
            dimC[ndimA - 2] = AH; dimC[ndimA - 1] = BW;
        }
        else {//like B
            dimC = new int[ndimB];
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
            dimC[ndimB - 2] = AH; dimC[ndimB - 1] = BW;
        }
        
        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMul(C.address, A.address, B.address,
                batchA, AH, BW, AW);//N = AH, M = BW, K = AW
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMul(Tensor C, Tensor A, Tensor B) 
    {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMul(C, A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        if(check) {//A[N: AH, K: AW] * B[K: BH, M: BW] -> C[N: AH, M: BW]
            require_dataType(C);
            require_dataType(A);
            require_dataType(B);
            
            int BH = dimB[ndimB - 2];
            int dimC[] = C.dim, CH = dimC[ndimC - 2], CW = dimC[ndimC - 1];
            int batchB = B.length / (BH * BW);
            int batchC = C.length / (CH * CW);
           
            if(ndimA < 3) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB < 3) throw new IllegalArgumentException("B.ndim at least greater than 2");
            if(ndimC < 3) throw new IllegalArgumentException("C.ndim at least greater than 2");
            equals(CH, "C.height", AH, "A.height");//N = CH = AH
            equals(CW, "C.width",  BW, "B.width"); //M = CW = BW
            equals(AW, "A.width",  BH, "B.height");//K = AW = BH
            equals(batchA, "A.batch", batchB, "B.batch");
            equals(batchA, "A.batch", batchC, "C.batch");
        }
        
        Syncer sc = core.batchMatMul(C.address, A.address, B.address,
                batchA, AH, BW, AW);//N = AH, M = BW, K = AW
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT1">
    public Tensor batchMatMulT1(Tensor A, Tensor B) {
        return batchMatMulT1(true, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT1(boolean likeA, Tensor A, Tensor B)
    {
        int ndimA = A.ndim(), ndimB = B.ndim();
        if(ndimA == 2 && ndimB == 2) return matMulT1(A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        if(check) {//A^T[N: AW, K: AH] * B[K: BH, M: BW] -> C[N: AW, M: BW]
            require_dataType(A);
            require_dataType(B);
            
            int BH = dimB[ndimB - 2];
            int batchB = B.length / (BH * BW);
            
            if(ndimA < 3) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB < 3) throw new IllegalArgumentException("B.ndim at least greater than 2");
            equals(AH, "A.height", BH, "B.height");//K = AH = BH
            equals(batchA, "A.batch", batchB, "B.batch");
        }
        
        int[] dimC;//A^T[N: AW, K: AH] * B[K: BH, M: BW] -> C[N: AW, M: BW]
        if(likeA) {//likeA
            dimC = new int[ndimA]; 
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
            dimC[ndimA - 2] = AW; dimC[ndimA - 1] = BW;
        }
        else {//like B
            dimC = new int[ndimB]; 
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
            dimC[ndimB - 2] = AW; dimC[ndimB - 1] = BW;
        }
        
        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMulT1(C.address, A.address, B.address,
                batchA, AW, BW, AH);//N = AW, M = BW, K = AH
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT1(Tensor C, Tensor A, Tensor B) 
    {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMulT1(C, A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BW = dimB[ndimB - 1];
        int batchA = A.length / (AH * AW);
        
        if(check) {//A^T[N: AW, K: AH] * B[K: BH, M: BW] -> C[N: AW, M: BW]
            require_dataType(C);
            require_dataType(A);
            require_dataType(B);
            
            int BH = dimB[ndimB - 2];
            int dimC[] = C.dim, CH = dimC[ndimC - 2], CW = dimC[ndimC - 1];
            int batchB = B.length / (BH * BW);
            int batchC = C.length / (CH * CW);
            
            if(ndimA <= 2) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB <= 2) throw new IllegalArgumentException("B.ndim at least greater than 2");
            if(ndimC <= 2) throw new IllegalArgumentException("C.ndim at least greater than 2");
            equals(CH, "C.height", AW, "A.width"); //N = CH = AH
            equals(CW, "C,width",  BW, "B.wdith"); //M = CW = BW
            equals(AH, "A.height", BH, "B.height");//K = AW = BH
            equals(batchA, "A.batch", batchB, "B.batch");
            equals(batchA, "A.batch", batchC, "C.batch");
        }
        
        Syncer sc = core.batchMatMulT1(C.address, A.address, B.address,
                batchA, AW, BW, AH);//N = AW, M = BW, K = AH
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2">
    public Tensor batchMatMulT2(Tensor A, Tensor B) {
        return batchMatMulT2(true, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT2(boolean likedA, Tensor A, Tensor B) 
    {
        int ndimA = A.ndim(), ndimB = B.ndim();
        if(ndimA == 2 && ndimB == 2) return matMulT2(A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BH = dimB[ndimB - 2];
        int batchA = A.length / (AH * AW);
        
        if(check) {//A[N: AH, K: AW] * B^T[K: BW, M: BH] -> C[N: AH, M: BH]
            require_dataType(A);
            require_dataType(B);
            
            int BW = dimB[ndimB - 1];
            int batchB = B.length / (BH * BW);
            
            if(ndimA <= 2) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB <= 2) throw new IllegalArgumentException("B.ndim at least greater than 2");
            equals(AW, "A.width", BW, "B.width");//K = AW = BW
            equals(batchA, "A.batch", batchB, "B.batch");
        }
        
        int[] dimC;//A[N: AH, K: AW] * B^T[K: BW, M: BH] -> C[N: AH, M: BH]
        if(likedA) {//likeA
            dimC = new int[ndimA]; 
            for(int i=0; i<ndimA - 2; i++) dimC[i] = dimA[i];
            dimC[ndimA - 2] = AH; dimC[ndimA - 1] = BH;
        }
        else {//likedB
            dimC = new int[ndimB];
            for(int i=0; i<ndimB - 2; i++) dimC[i] = dimB[i];
            dimC[ndimB - 2] = AH; dimC[ndimB - 1] = BH;
        }
        
        Tensor C = this.empty(dimC).c();
        Syncer sc = core.batchMatMulT2(C.address, A.address, B.address, 
                batchA, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchMatMulT2(Tensor C, Tensor A, Tensor B) 
    {
        int ndimA = A.ndim(), ndimB = B.ndim(), ndimC = C.ndim();
        if(ndimA == 2 && ndimB == 2 && ndimC == 2) return matMulT2(C, A, B);
        
        int dimA[] = A.dim, AH = dimA[ndimA - 2], AW = dimA[ndimA - 1];
        int dimB[] = B.dim, BH = dimB[ndimB - 2];
        int batchA = A.length / (AH * AW);
        if(check) {//A[N: AH, K: AW] * B^T[K: BW, M: BH] -> C[N: AH, M: BH]
            int BW = dimB[ndimB - 1];
            int dimC[] = C.dim, CH = dimC[ndimC - 2], CW = dimC[ndimC - 1];
            int batchB = B.length / (BH * BW);
            int batchC = C.length / (CH * CW);
            
            if(ndimA <= 2) throw new IllegalArgumentException("A.ndim at least greater than 2");
            if(ndimB <= 2) throw new IllegalArgumentException("B.ndim at least greater than 2");
            if(ndimC <= 2) throw new IllegalArgumentException("C.ndim at least greater than 2");
            equals(CH, "C.height", AH, "A.height");//N = CH = AH
            equals(CW, "C.width",  BH, "B.height");//M = CW = BH
            equals(AW, "A.width",  BW, "B.width"); //K = AW = BW
            equals(batchA, "A.batch", batchB, "B.batch");
            equals(batchA, "A.batch", batchC, "C.batch");
        }
        
        Syncer sc = core.batchMatMulT2(C.address, A.address, B.address, 
                batchA, AH, BH, AW);
        if(sync) sc.sync(); else C.setSyncer(sc);
        return C;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="full_connect(3D+, 2D)">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect(Tensor X, Tensor W)
    {
        int ndimX = X.ndim(); 
        if(ndimX == 2) return matMul(X, W);
        
        int dimX[] = X.dim, XH = dimX[ndimX - 2], XIW = dimX[ndimX - 1];//[N, H, IW]
        int dimW[] = W.dim, WOW = dimW[1];//[IW, OW]
        int batchX = X.length / (XH * XIW);
        
        if(check) {//[batch, H, IW] * [IW, OW] -> [batch, H, OW]
            require_dataType(X);
            require_dataType(W);
            
            int WIW = dimW[0];
            if(X.ndim() < 3) throw new IllegalArgumentException("X.ndim must greater than 2");
            if(W.ndim() != 2) throw new IllegalArgumentException("W.ndim != 2");
            equals(XIW, "X.features", WIW, "W.input_features");
        }
        
        int[] dimY = new int[ndimX];
        for(int i=0; i<ndimX - 1; i++) dimY[i] = dimX[i];
        dimY[ndimX - 1] = WOW;
        Tensor Y = this.empty(dimY).c();
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul(Y.address, X.address, W.address, 
                (batchX * XH), WOW, XIW);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect(Tensor X, Tensor W, Tensor Bias)
    {
        int ndimX = X.ndim();
        if(ndimX == 2) return matMul_biased(X, W, Bias); 
        
        int dimX[] = X.dim, XH = dimX[ndimX - 2], XIW = dimX[ndimX - 1];//[N, H, IW]
        int dimW[] = W.dim, WOW = dimW[1];//[IW, OW]
        int batchX = X.length / (XH * XIW);
        
        if(check) {//[batch, H, IW] * [IW, OW] -> [batch, H, OW]
            require_dataType(X);
            require_dataType(W);
            require_dataType(Bias);
            
            int WIW = dimW[0];
            if(X.ndim() < 3) throw new IllegalArgumentException("X.ndim must greater than 2");
            if(W.ndim() != 2) throw new IllegalArgumentException("W.ndim != 2");
            equals(XIW, "X.features", WIW, "W.input_features");//K = XW = WH
            equals(Bias.length, "Bias.length", WOW, "W.output_features");
        }
        
        int[] dimY = new int[ndimX];
        for(int i=0; i<ndimX - 1; i++) dimY[i] = dimX[i];
        dimY[ndimX - 1] = WOW;
        Tensor Y = this.empty(dimY).c();
        
        //reshape: [batch * H, IW] * [IW, OW] -> [batch * H, OW]
        Syncer sc = core.matMul_biased(Y.address, X.address, W.address, 
                (batchX * XH), WOW, XIW, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward_propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_deltaX(Tensor deltaY, Tensor W)
    {
        int ndimY = deltaY.ndim();
        if(ndimY == 2) return matMulT2(deltaY, W);
        
        int dimY[] = deltaY.dim, YH = dimY[ndimY - 2], YOW = dimY[ndimY - 1];//[batch, H, OW]
        int dimW[] = W.dim, WIW = dimW[0];//[IW, OW]
        int batchY = deltaY.length / (YH * YOW);
        
        if(check) {//[batch, H, OW] * [OW, IW] = [batch, H, IW]: deltaX = deltaY * W^T
            require_dataType(deltaY);
            require_dataType(W);
            
            int WOW = dimW[1];
            if(deltaY.ndim() < 3) throw new IllegalArgumentException("deltaY.ndim must >= 3");
            if(W.ndim() != 2) throw new IllegalArgumentException("W.ndim != 2");
            equals(YOW, "deltaY.features", WOW, "W.output_features");//K = YW = WW
        }

        int[] dimX = new int[ndimY];
        for(int i=0; i<ndimY - 1; i++) dimX[i] = dimY[i];
        dimX[ndimY - 1] = WIW;
        Tensor deltaX = this.empty(dimX).c();
        
        //reshape: [batch * H, OW] * [OW, IW] = [batch * H, IW]: deltaX = deltaY * W^T
        Syncer sc = core.matMulT2(deltaX.address, deltaY.address, W.address,
                (batchY * YH), WIW, YOW);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor fullconnect_deltaW(Tensor X, Tensor deltaY)
    {
        int ndimY = deltaY.ndim(), ndimX = X.ndim();
        if(ndimY == 2 && ndimX == 2) return matMulT1(X, deltaY);
        
        int dimY[] = deltaY.dim, YH = dimY[ndimY - 2], YOW = dimY[ndimY - 1];
        int dimX[] = X.dim, XIW = dimX[ndimX - 1];
        int batchY = deltaY.length / (YH * YOW);
        
        if(check) {//[IW, batch * H] * [batch * H, OW] = [IW, OW]: X^T * deltaY
            require_dataType(X);
            require_dataType(deltaY);
            
            int XH = dimX[ndimX - 2];
            int batchX = X.length / (XH * XIW);
            if(X.ndim() < 3) throw new IllegalArgumentException("X.ndim must greater than 2");
            if(deltaY.ndim() < 3) throw new IllegalArgumentException("deltaY.ndim must greater than 2");
            equals((batchY * YH), "(deltaY.batch * Y.height)", (batchX * XH), "(X.batch * X.height)");
        }
        
        Tensor deltaW = this.empty(XIW, YOW).c();
        
        //[K, batch * N] * [batch * N, M] = [K, M]: X^T * deltaY
        Syncer sc = core.matMulT1(deltaW.address,
                X.address, deltaY.address,
                XIW, YOW, (batchY * YH));
        if(sync) sc.sync(); else deltaW.setSyncer(sc);//1101 1119
        return deltaW;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    //<editor-fold defaultstate="collapsed" desc="reshape && view: X[oldDim] -> X'[newDim]">
    public Tensor view(boolean inplaced, Tensor X, int... dim)  
    {
        negativeDim(X.length, dim);//newDim.length == oldDim.length
        
        if(!X.memStrucEquals(X.length, dim)) 
            throw new IllegalArgumentException("the old MemStructure is different from the New one");
        
        if(inplaced) { X.setDim(check, dim); return X; }
        
        Tensor Y = new Tensor(check, this, core.dataType(), dim);
        Y.copy_memoryMetaData(X);
        Y.mod_counter = X.mod_counter;//important to let y inherit X.mod_count
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor reshape(boolean inplaced, Tensor X, int...dim)
    {
        negativeDim(X.length, dim);//newDim.length == oldDim.length
        
        if(!inplaced) return tensor(X, dim);
        
        //inplaced = true, use the old memory, no need to memcpy
        if(X.memStrucEquals(X.length, dim)) { X.setDim(check, dim); return X; } 
        
        //inplaced = true, but need to reorganize the mem structure
        Tensor Y = tensor(X, dim);//Y[newDim, newAddress]
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim, X.address = Y.address
        
        Syncer sc = Syncer.dual(Y.syncer, ()->{ core.free((int) old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="concat & stack: X[] -> Y">
    public Tensor stack(Tensor... X) {return concat(0, X);}
    public Tensor concat(Tensor... X) { return concat(-1, X); }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor concat(int dimIdx, Tensor... X) 
    {
        final int ndim = X[0].ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int dimX[][] = new int[X.length][];
        for(int i=0; i<X.length; i++) dimX[i] = X[i].dim;
        
        if(check) {
            require_dataType(X);
            
            if(X.length < 2) throw new IllegalArgumentException("At least input two tensors to concat");
            for(int[] dimXi : dimX) 
                if (dimXi.length <= dimIdx) throw new IllegalArgumentException("dimIndex exceeds X.ndim");
            
            for(int i=1; i<X.length; i++) 
                for(int j=0; j<ndim; j++) {
                    if(j == dimIdx) continue; //only dimIndex is different
                    if(dimX[0][j] != dimX[i][j]) throw new IllegalArgumentException(
                            String.format("X[%d].dim[%d](%d) != X[0].dim[%d](%d)",i, j, dimX[i][j], j, dimX[0][j]));
                }
        }
        
        //compute the dim of output tensor--------------------------------------
        int[] dimY = Vector.arrayCopy(dimX[0]);//concat_dim = sum(X[i].dim(dimIndex), 0, n-1) 
        for(int i=1; i<X.length; i++) dimY[dimIdx] += dimX[i][dimIdx];//dimIndex: the concat dim
        Tensor Y = this.empty(dimY);
        
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize multiple: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[0][i];
        
        int[] copyWidth = new int[X.length];
        int[] strideX = new int[X.length];
        for(int i=0; i<X.length; i++) {
            copyWidth[i] = commonWidth * dimX[i][dimIdx];//from dimIndex to End
            
            int width = X[i].lastDim();//consider memAlignment: 
            int stride = ((width + 3) >> 2) << 2;
            strideX[i] = copyWidth[i] / width * stride;
            if(dimIdx != ndim - 1) copyWidth[i] = strideX[i];
        } 
        
        int strideY = commonWidth * dimY[dimIdx]; {//from dimIndex to End
            int width = Y.lastDim();//consider mem alignment
            int stride = ((width + 3) >> 2) << 2;
            strideY = strideY / width * stride;
        }
                
        Syncer[] sc = new Syncer[X.length]; Y.c();//Y is synchronized 
        for(int i=0, Ystart = 0; i<X.length; Ystart += copyWidth[i++]) 
        {
            int length = (dimIdx == ndim - 1 ? X[i].length : X[i].lengthv);
            sc[i] = core.gappedMemcpy2D(
                    X[i].address, 0, strideX[i], 
                    Y.address, Ystart, strideY, 
                    copyWidth[i], length);
        }
        if(sync) { for(Syncer syncer : sc) syncer.sync(); }
        else Y.setSyncer(new ChainSyncer(sc));
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="split & chunk: X -> Y[]">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] chunk(Tensor X, int dimIdx, int n) 
    {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        if(check) {
            require_dataType(X);
            if(n < 1) throw new IllegalArgumentException("n at least 2 to chunk a tensor");
            if(n > dimSize) throw new IllegalArgumentException("n > X.dim[dimIdx]");
        }
        
        int[] section = new int[n];
        int div = dimSize  / n, rm = dimSize % n;
        for(int i=0; i<n; i++) section[i] = div;
        section[n - 1] += rm;
        
        return __split(X, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="negativeSection">
    private int negativeSection(int dimSize, int...section) {
        int index = -1;
        for(int i=0; i<section.length; i++) 
            if(section[i] == -1) { index = i; break; }
        
        int sectionSum = Vector.sum(section);
        if(index != -1) // -1 -> 0;
            section[index] = dimSize - (sectionSum + 1);
        return sectionSum;
    }
    //</editor-fold>
    @Passed("CudaFloat32EngieBase") 
    public Tensor[] split(Tensor X, int dimIdx, int...section) 
    {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        int sectionSum = negativeSection(dimSize, section);//exclude -1 in section
         
        if(check) {
            require_dataType(X);
            if(section.length < 2) throw new IllegalArgumentException("At least 2 sections to split a tensor");
            for(int sec : section) 
                if(sec <= 0) throw new IllegalArgumentException("elements of section must positive");
            if(sectionSum != dimSize) 
                throw new IllegalArgumentException("X.dim[dimIdx] doesn't match the input sections");
        }
        
        return __split(X, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="__split">
    private Tensor[] __split(Tensor X, int dimIdx, int[] section)
    {
        //create sub Tensor[] Y based on section--------------------------------
        int dimX[] = X.dim, ndim = X.ndim();
        
        Tensor[] Y = new Tensor[section.length];
        for(int i=0, dimY[] = Vector.arrayCopy(dimX); i<section.length; i++) {
            dimY[dimIdx] = section[i]; 
            Y[i] = this.empty(dimY);
        }
       
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize multiple: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[i];
        
        int[] copyWidth = new int[Y.length];
        int[] strideY = new int[Y.length];
        for(int i = 0; i<copyWidth.length; i++){
            copyWidth[i] = commonWidth * section[i];//from dimIndex to End
            
            int width = Y[i].lastDim();//consider memory alginment
            int stride = ((width + 3) >> 2) << 2;
            strideY[i] = copyWidth[i] / width * stride;
            
            //width the same mem_struture, is dimIdex != -1
            if(dimIdx != ndim - 1) copyWidth[i] = strideY[i];
        }
        
        //compute the start index in X(src) for each element of Y[](dst)--------
        int strideX = commonWidth * dimX[dimIdx]; {//from dimIndex to End
            int width = X.lastDim();//consider memAlignment
            int stride = ((width + 3) >> 2) << 2;
            strideX = strideX / width * stride;
        }
        
        Syncer[] scs = new Syncer[Y.length];
        for(int i=0, Xstart = 0; i<Y.length; Xstart += copyWidth[i++]) {
            int length = ((dimIdx == ndim - 1) ? Y[i].length : Y[i].lengthv);
            scs[i] = core.gappedMemcpy2D(
                    X.address, Xstart, strideX, 
                    Y[i].c().address, 0, strideY[i],//Y[i] is synchronized
                    copyWidth[i], length);
        }
        
        if(sync) for (Syncer sc : scs) sc.sync();
        else for(int i=0; i<Y.length; i++) Y[i].setSyncer(scs[i]);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Transpose(2D-4D): X -> X^T">
    @Passed("CudaFloat32EngieBase")
    public Tensor transpose(boolean inplaced, Tensor X, int dimIdx1, int dimIdx2) 
    {
        final int[] Xdim = X.dim;
        if(dimIdx1 < 0) dimIdx1 = Xdim.length + dimIdx1;
        if(dimIdx2 < 0) dimIdx2 = Xdim.length + dimIdx2;
        
        if(check) {
            require_dataType(X);
            if(Xdim.length < 2) throw new IllegalArgumentException("ndim must >= 2");
            if(dimIdx1 == dimIdx2) throw new IllegalArgumentException("dimIdx1 == dimIdx2");
            if(dimIdx1 >= Xdim.length) throw new IllegalArgumentException("dimIndex1 >= ndim");
            if(dimIdx2 >= Xdim.length) throw new IllegalArgumentException("dimIndex2 >= ndim");
        }
        
        int[] Ydim = Vector.arrayCopy(Xdim);
        int t = Ydim[dimIdx1]; Ydim[dimIdx1] = Ydim[dimIdx2]; Ydim[dimIdx2] = t;
        Tensor Y = empty(Ydim).c();//Y[newDim, newAddress]
        
        Syncer sc1 = core.transpose(
                Y.address, Ydim, 
                X.address, Xdim,
                dimIdx1, dimIdx2,
                X.lastDim(), Y.lastDim(),
                X.length);
      
        if(!inplaced) {//inplaced = false, return the new Tensor Y==============
            if(sync) sc1.sync(); else Y.setSyncer(sc1);
            return Y;
        }
        
        //inplaced = true, return the old Tensor X==============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ core.free((int) old_memLen, oldAddr); });
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="rot180: X -> Xr">
    public Tensor rot180(boolean inplaced, Tensor X)
    {
        if(check) { 
            require_dataType(X);
            if(X.ndim() != 3 || X.ndim() != 4) throw new IllegalArgumentException(
                    "X.ndim != 3 || X.ndim != 4");
        }
        
        int dimX[] = X.dim, ndim = dimX.length;
        int IH = dimX[ndim - 3], IW = dimX[ndim - 2], IC = dimX[ndim - 1];
        int N = (ndim == 3 ? 1 : dimX[0]);//4D or 3D
       
        Tensor Y = (inplaced? X : this.empty(X.dim).c());
        Syncer sc = core.rot180_3D(Y.address, X.address, N, IH, IW, IC);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Convlution 3D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: conv3D">
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D(Tensor X, Tensor W, int sh, int sw, int ph, int pw)
    {
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(X); 
            require_dataType(W); 
            if(X.ndim() != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(WIC, "W.in_channels, ", XIC, "X.in_channels");
        }
        
        int OH = (IH - FH + (ph << 1)) / sh + 1;
        int OW = (IW - FW + (pw << 1)) / sw + 1;
        Tensor Y = this.empty(XN, OH, OW, WOC).c();
        Syncer sc = core.conv3D(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw)
    {
        int[] dimY = Y.dim, dimX = X.dim, dimW = W.dim;
        int YN  = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(Y); 
            require_dataType(X); 
            require_dataType(W); 
            if(Y.ndim() != 4) throw new IllegalArgumentException("Tensor Y.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(WOC, "W.out_channels", YOC, "Y.out_channels");
            equals(WIC, "W.in_channels, ", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        int pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        Syncer sc = core.conv3D(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor conv3D_biased(Tensor X, Tensor W, int sh, int sw, int ph, int pw, Tensor Bias)
    { 
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(X);
            require_dataType(W); 
            if(X.ndim() != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(WIC, "W.in_channels" , XIC, "X.in_channels");
            equals(Bias.lastDim(), "Bias.lastDim", WOC, "W.out_channels");
        }
        
        int OH = (IH - FH + (ph << 1)) / sh + 1;
        int OW = (IW - FW + (pw << 1)) / sw + 1;
        Tensor Y = this.empty(XN, OH, OW, WOC).c();
        
        Syncer sc = core.conv3D_biased(
                Y.address, OH, OW,
                X.address, IH, IW,
                W.address, FH, FW, 
                XN, WIC, WOC, 
                sh, sw, ph, pw, 
                Bias.address, Y.lengthv);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deconv3D_deltaW">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaW(Tensor X, Tensor deltaY, int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(X); 
            require_dataType(deltaY); 
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("Tensor deltaY.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
        }
        
        int FH = IH + (ph << 1) - (OH - 1)*sh;
        int FW = IW + (pw << 1) - (OW - 1)*sw;
        Tensor deltaW = this.empty(YOC, FH, FW, XIC).c();
        
        Syncer sc = core.dconv3D_deltaW(true, 
                deltaW.address, FH, FW, 
                X.address, IH, IW, 
                deltaY.address, OH, OW, 
                XN, XIC, YOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false]: (IH + 2*ph - IH)%sh != 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaW(Tensor deltaW, Tensor X, Tensor deltaY, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimX = X.dim, dimW = deltaW.dim;
        int YN  = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(deltaW); 
            require_dataType(X);
            require_dataType(deltaY); 
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("Tensor deltaY.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            if(deltaW.ndim() != 4) throw new IllegalArgumentException("Tensor deltaW.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(WOC, "W.out_channels", YOC, "Y.out_channels");
            equals(WIC, "W.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        int pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        
        Syncer sc = core.dconv3D_deltaW(false,
                deltaW.address, FH, FW,
                X.address, IH, IW,
                deltaY.address, OH, OW,
                XN, WIC, WOC,
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaW(Tensor X, Tensor deltaY, int FH, int FW,
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
        }
        
        Tensor deltaW = this.empty(YOC, FH, FW, XIC).c();
        Syncer sc = core.dconv3D_deltaW(false,
                deltaW.address, FH, FW,
                X.address, IH, IW, 
                deltaY.address, OH, OW, 
                XN, XIC, YOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaW.setSyncer(sc);
        return deltaW;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deconv3D_deltaX">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaX(Tensor deltaY, Tensor W, 
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimW = W.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(deltaY); 
            require_dataType(W); 
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("Tensor deltaY.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(WOC, "W.out_channels", YOC, "Y.out_channels");
        }
        
        int IH = (OH - 1)*sh + FH - (ph << 1);
        int IW = (OW - 1)*sw + FW - (pw << 1);
        Tensor deltaX = this.empty(YN, IH, IW, WIC).c();
        
        Syncer sc = core.dconv3D_deltaX(true,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address, FH, FW, 
                YN, WIC, WOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false]: (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimW = W.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
        
        if(check) {
            require_dataType(deltaX); 
            require_dataType(deltaY); 
            require_dataType(W); 
            if(deltaX.ndim() !=4 ) throw new IllegalArgumentException("Tensor deltaX.ndim != 4");
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("Tensor deltaY.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(WOC, "W.out_channels", YOC, "Y.out_channels");
            equals(XN, "X.batch", YN, "Y.batch");
            equals(XIC, "X.in_channesl", WIC, "W.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH + 1) >> 1;//ceiling
        int pw = ((OW - 1)*sw + FW - IW + 1) >> 1;//ceiling
        Syncer sc = core.dconv3D_deltaX(false,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address, FH, FW, 
                YN, WIC, WOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor dconv3D_deltaX(Tensor deltaY, Tensor W, int IH, int IW,
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimW = W.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YOC = dimY[3];//Y[ N, OH, OW, OC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2], WIC = dimW[3];//W[OC, FH, FW, IC]
        
        if(check) {
            require_dataType(deltaY); 
            require_dataType(W); 
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("Tensor deltaY.ndim != 4");
            if(W.ndim() != 4) throw new IllegalArgumentException("Tensor W.ndim != 4");
            equals(WOC, "W.out_channels", YOC, "Y.out_channels");
        }
        
        Tensor deltaX = this.empty(YN, IH, IW, WIC).c();
        Syncer sc = core.dconv3D_deltaX(false,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW, 
                W.address, FH, FW, 
                YN, WIC, WOC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Deconvolution 3D">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor dconv3D(Tensor X, Tensor W, int sh, int sw, int ph, int pw) {
        return dconv3D_deltaX(X, W, sh, sw, ph, pw);
    }
    
    public Tensor dconv3D(Tensor Y, Tensor X, Tensor W, int sh, int sw) {
        return dconv3D_deltaX(Y, X, W, sh, sw);
    }
    
    public Tensor dconv3D(Tensor X, Tensor W, int OH, int OW, 
            int sh, int sw, int ph, int pw) {
        return dconv3D_deltaX(X, W, OH, OW, sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: conv3D_deltaW">
    public Tensor conv3D_deltaW(Tensor deltaY, Tensor X, int sh, int sw, int ph, int pw) {//X is the conv_kernel
        return dconv3D_deltaW(X, deltaY, sh, sw, ph, pw);
    }
    
    public Tensor conv3D_deltaW(Tensor deltaW, Tensor deltaY, Tensor X, int sh, int sw) {//X is the conv_kernel
        return dconv3D_deltaW(deltaW, X, deltaY, sh, sw);
    }
    
    public Tensor conv3D_deltaW(Tensor deltaY, Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) {//X is the conv_kernel
        return dconv3D_deltaW(X, deltaY, FH, FW, sh, sw, ph, pw);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: conv3D_deltaX">
    public Tensor conv3D_deltaX(Tensor deltaY, Tensor W, int sh, int sw, int ph, int pw) {
        return conv3D(deltaY, W, sh, sw, ph, pw);
    }
    
    public Tensor conv3D_deltaX(Tensor deltaX, Tensor deltaY, Tensor W, int sh, int sw) {
        return conv3D(deltaX, deltaY, W, sh, sw);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Pooling2D">
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: pool2D_max">
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max(Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        int[] dimX = X.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(X);
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
        }
        
        int OH = (IH - FH + (ph << 1))/sh + 1;
        int OW = (IW - FW + (pw << 1))/sw + 1;
        Tensor Y = this.empty(XN, OH, OW, XIC).c();
        Syncer sc = core.pool2D_max(
                Y.address, OH, OW, 
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max(Tensor Y, Tensor X, int FH, int FW, int sh, int sw) 
    {
        int[] dimY = Y.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(Y);
            require_dataType(X);
            if(Y.ndim() != 4) throw new IllegalArgumentException("Y.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.pool2D_max(
                Y.address, OH, OW, 
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: upool2D_max">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_max(Tensor deltaX,
            Tensor deltaY, Tensor Y, Tensor X,
            int FH, int FW, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X);
            
            int YIC = dimY[3];
            if(deltaX.ndim() != 4) throw new IllegalArgumentException("deltaX.ndim != 4");
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals_valueStructure(deltaX, "deltaX", X, "X");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.upool2D_max(true,
                deltaY.address, Y.address, OH, OW, 
                deltaX.address, X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false] : (IH + 2*ph - IH)%sh != 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_max(Tensor deltaY, Tensor Y, Tensor X,
            int FH, int FW, int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[N, IH, IW, IC]
                
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X);
            
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(XN, "X.batch", YN, "Y.batch");
            equals(XIC, "X.in_channels", YIC, "Y.in_channels");
        }
        
        Tensor deltaX = this.empty(XN, IH, IW, XIC).c();        
        Syncer sc = core.upool2D_max(false,
                deltaY.address, Y.address, OH, OW, 
                deltaX.address, X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_max(Tensor deltaX,
            Tensor deltaY, Tensor Y, Tensor X,
            int FH, int FW, int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            int YIC = dimY[3];
            if(deltaX.ndim() != 4) throw new IllegalArgumentException("deltaX.ndim != 4");
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals_valueStructure(deltaX, "deltaX", X, "X");
            equals(XN, "X.batch", YN, "Y.batch");
            equals(XIC, "X.in_channels", YIC, "Y.in_channels");
        }
        
        Syncer sc = core.upool2D_max(false,
                deltaY.address, Y.address, OH, OW, 
                deltaX.address, X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Max Pooling 2D indexed">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: pool2D_max_indexed">
    @Passed("CudaFloat32EngieBase") //return [Y, Index]
    public Tensor[] pool2D_max_indexed(Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        int[] dimX = X.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) { 
            require_dataType(X);
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4"); 
        }
        
        int OH = (IH - FH + (ph << 1))/sh + 1;
        int OW = (IW - FW + (pw << 1))/sw + 1;
        Tensor Y = this.empty(XN, OH, OW, XIC);
        Tensor Index = this.empty_int32(XN, OH, OW, XIC);
        
        Syncer sc = core.pool2D_max_indexed(
                Y.c().address, Index.c().address, OH, OW,
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[]{ Y, Index };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_max_indexed(Tensor Y, Tensor Index,
            Tensor X, int FH, int FW, int sh, int sw) 
    {
        int[] dimY = Y.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(Y);
            require_int32(Index);
            require_dataType(X);
            
            int YIC = dimY[3];
            if(Y.ndim() != 4) throw new IllegalArgumentException("Y.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            if(Index.ndim() != 4) throw new IllegalArgumentException("Index.ndim != 4");
            equals_dim(Index, "Index<int32>", Y, "Y");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.pool2D_max_indexed(
                Y.address, Index.address, OH, OW,
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: upool2D_max">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_max_indexed(Tensor deltaX,
            Tensor deltaY, Tensor Index, 
            int FH, int FW, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimX = deltaX.dim;//Y[ N, OH, OW, IC]
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(deltaX);
            require_int32(Index);
            require_dataType(deltaY);
            
            int YIC = dimY[3];
            if(deltaX.ndim() != 4) throw new IllegalArgumentException("deltaX.ndim != 4");
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals_valueStructure(Index, "Index<int32>", deltaY, "Y");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.upool2D_max_Indexed(true, 
                deltaX.address, IH, IW, 
                deltaY.address, Index.address, OH, OW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false] : (IH + 2*ph - IH)%sh != 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_max_indexed(Tensor deltaY, Tensor Index, 
            int IH, int IW, int FH, int FW, 
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        
        if(check) {
            require_dataType(deltaY);
            require_int32(Index);
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals_valueStructure(Index, "Index<in32>", deltaY, "deltaY");
        }
        
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        Syncer sc = core.upool2D_max_Indexed(false, 
                deltaX.address, IH, IW, 
                deltaY.address, Index.address, OH, OW, 
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Avg Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: pool2D_avg">
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg(Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        int[] dimX = X.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) { 
            require_dataType(X);
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4"); 
        }
        
        int OH = (IH - FH + (ph << 1))/sh + 1;
        int OW = (IW - FW + (pw << 1))/sw + 1;
        Tensor Y = this.empty(XN, OH, OW, XIC).c();
        
        Syncer sc = core.pool2D_avg(
                Y.address, OH, OW, 
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg(Tensor Y, Tensor X, int FH, int FW, int sh, int sw) 
    {
        int[] dimY = Y.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(Y);
            require_dataType(X);
            
            int YIC = dimY[3];
            if(Y.ndim() != 4) throw new IllegalArgumentException("Y.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.pool2D_avg(
                Y.address, OH, OW, 
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: upool2D_avg">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg(Tensor deltaY, int FH, int FW, int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        
        if(check) {
            require_dataType(deltaY);
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4"); 
        }
        
        int IH = (OH - 1)*sh + FH - (ph << 1);
        int IW = (OW - 1)*sw + FW - (pw << 1);
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        
        Syncer sc = core.upool2D_avg(true,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg(Tensor deltaX, Tensor deltaY, int FH, int FW, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            int YIC = dimY[3];
            if(deltaX.ndim()!=4) throw new IllegalArgumentException("deltaX.ndim != 4");
            if(deltaY.ndim()!=4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_chhannels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.upool2D_avg(true,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false] : (IH + 2*ph - IH)%sh != 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg(Tensor deltaY,
            int IH, int IW, int FH, int FW, 
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        
        if(check) { 
            require_dataType(deltaY);
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4"); 
        }
        
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        Syncer sc = core.upool2D_avg(false,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Avg Pooling 2D ignore padding">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: pool2D_avg">
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg_ignore_padding(Tensor X, int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        int[] dimX = X.dim;
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(X);
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4"); 
        }
        
        int OH = (IH - FH + (ph << 1))/sh + 1;
        int OW = (IW - FW + (pw << 1))/sw + 1;
        Tensor Y = this.empty(XN, OH, OW, XIC).c();
        
        Syncer sc = core.pool2D_avg_ignore_padding(
                Y.address, OH, OW, 
                X.address, IH, IW,
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor pool2D_avg_ignore_padding(Tensor Y, Tensor X, int FH, int FW, int sh, int sw) 
    {
        int[] dimY = Y.dim, dimX = X.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, IC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(Y);
            require_dataType(X);
            
            int YIC = dimY[3];
            if(Y.ndim() != 4) throw new IllegalArgumentException("Y.ndim != 4");
            if(X.ndim() != 4) throw new IllegalArgumentException("X.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_channels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.pool2D_avg_ignore_padding(
                Y.address, OH, OW, 
                X.address, IH, IW, 
                FH, FW, XN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: upool2D_avg">
    //<editor-fold defaultstate="collapsed" desc="strict[true] : (IH + 2*ph - IH)%sh == 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg_ignore_padding(Tensor deltaY, int FH, int FW, int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        
        if(check) { 
            require_dataType(deltaY);
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4"); 
        }
        
        int IH = (OH - 1)*sh + FH - (ph << 1);
        int IW = (OW - 1)*sw + FW - (pw << 1);
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        
        Syncer sc = core.upool2D_avg_ignore_padding(true,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg_ignore_padding(Tensor deltaX, Tensor deltaY, int FH, int FW, int sh, int sw)
    {
        int[] dimY = deltaY.dim, dimX = deltaX.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2];//Y[ N, OH, OW, OC]
        int XN = dimX[0], IH = dimX[1], IW = dimX[2], XIC = dimX[3];//X[ N, IH, IW, IC]
        
        if(check) {
            require_dataType(deltaX);
            require_dataType(deltaY);
            
            int YIC = dimY[3];
            if(deltaX.ndim()!=4) throw new IllegalArgumentException("deltaX.ndim != 4");
            if(deltaY.ndim()!=4) throw new IllegalArgumentException("deltaY.ndim != 4");
            equals(YN, "Y.batch", XN, "X.batch");
            equals(YIC, "Y.in_channels", XIC, "X.in_chhannels");
        }
        
        int ph = ((OH - 1)*sh + FH - IH) >> 1;
        int pw = ((OW - 1)*sw + FW - IW) >> 1;
        Syncer sc = core.upool2D_avg_ignore_padding(true,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, XIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="strict[false] : (IH + 2*ph - IH)%sh != 0">
    @Passed("CudaFloat32EngieBase")
    public Tensor upool2D_avg_ignore_padding(Tensor deltaY,
            int IH, int IW, int FH, int FW, 
            int sh, int sw, int ph, int pw)
    {
        int[] dimY = deltaY.dim;
        int YN = dimY[0], OH = dimY[1], OW = dimY[2], YIC = dimY[3];//Y[ N, OH, OW, OC]
        
        if(check) { 
            require_dataType(deltaY);
            if(deltaY.ndim() != 4) throw new IllegalArgumentException("deltaY.ndim != 4"); 
        }
        
        Tensor deltaX = this.empty(YN, IH, IW, YIC).c();  
        Syncer sc = core.upool2D_avg_ignore_padding(false,
                deltaX.address, IH, IW, 
                deltaY.address, OH, OW,
                FH, FW, YN, YIC, 
                sh, sw, ph, pw);
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="param check">
    private void equals(int V1, String name1, int V2, String name2) {
        if(V1 != V2) throw new IllegalArgumentException(
                name1 + '[' + V1 + "] != " + name2+ " [" + V2 + ']');
    }
    
    private void equals_dim(Tensor X1, String name1, Tensor X2, String name2) {
        if(!X1.dimEquals(X2) ) throw new IllegalArgumentException(
                name1 + ".dim != " + name2 + ".dim");
    }
    
    private void equals_dataType(Tensor X1, String name1, Tensor X2, String name2) {
        if(!X1.dataType.equals(X2.dataType)) throw new IllegalArgumentException(String.format(
                "%s.dataType[%s] != %s.dataType[%s]",
                name1, X1.dataType, name2, X2.dataType));
    }
    
    //<editor-fold defaultstate="collapsed" desc="equals_valueStructure">
    private void equals_valueStructure(Tensor X1, String name1, Tensor X2, String name2){
        if(!X1.valueStrucEquals(X2)) 
            throw new IllegalArgumentException(
                    "the value structure of Tensor " + name1 + " and " + name2 + " are different");
    }
    
    private void equals_valueStructure(String name, Tensor X0, Iterator<Tensor> iter){
        int index = 1;
        while(iter.hasNext()) {
            if(!X0.valueStrucEquals(iter.next())) 
                throw new IllegalArgumentException(name + "[" + 0 + "]"
                        + ".valueSturcture is different from " + name + "[" + index + "]"); 
            index++;
        }
    }
    
    private void equals_valueStructure(String name, Tensor[] Xs){
        for(int i=1; i<Xs.length; i++) {
            if(!Xs[0].valueStrucEquals(Xs[i]))
                throw new IllegalArgumentException(name + "[" + 0 + "]"
                        + ".valueSturcture is different from " + name + "[" + i + "]"); 
        }
    }
    
    private void equals_valueStructure(Tensor X1, String name1, Collection<Tensor> Xs, String name2) {
        int index = 0;
        for(Tensor X2 : Xs) {
            if(!X1.valueStrucEquals(X2))
                throw new IllegalArgumentException(
                    "the value structure of " + name1 + 
                            " is different from" + name2 + "[" + index + "]");
            index++;
        }
    }
    //</editor-fold>
    
    private void check_row(Tensor X1, String name1, Tensor X2row, String name2) {
        if(X1.ndim() <= 1) //X1.width == X2.width -> X1.stride == X2.stride
            throw new IllegalArgumentException(name1 + ".ndim must > 1");
        if(X1.lastDim() != X2row.lastDim()) 
            throw new IllegalArgumentException(name1 + ".lastDim != " + name2 + ".lastDim");
    }   
    
    private void check_field(Tensor X1, String name1, Tensor X2field, String name2) {
        if(X1.ndim() <= 1) 
            throw new IllegalArgumentException(name1 + ".ndim must > 1");
        if(X2field.ndim() != 1) 
            throw new IllegalArgumentException(name2 + ".ndim must == 1");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="equal, linear, quadratic, rpl, div, add_div"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    public Tensor equal(Tensor X1, Tensor X2) {
        return equal_abs(true, X1, X2, 0, 0);
    }
    public Tensor equal(boolean likeX1, Tensor X1, Tensor X2) {
        return equal_abs(likeX1, X1, X2, 0, 0);
    }
    
    public Tensor equal_abs(Tensor X1, Tensor X2, float min, float max) {
        return equal_abs(true, X1, X2, min, max);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor equal_abs(boolean likeX1, Tensor X1, Tensor X2, float min, float max)
    {
        if(X1.dataType.equals(core.dataType_int8()) && 
           X2.dataType.equals(core.dataType_int8()))
            return equal_abs_int8(likeX1, X1, X2, (byte)min, (byte)max);
        
        if(X1.dataType.equals(core.dataType_int32()) &&
           X2.dataType.equals(core.dataType_int32()))
            return equal_abs_int32(likeX1, X1, X2, (int)min, (int)max);
 
        Tensor Y1 = X1.dataType.equals(core.dataType())? X1 : this.to_dtype(false, X1);
        Tensor Y2 = X2.dataType.equals(core.dataType())? X2 : this.to_dtype(false, X2);
        
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D(Y.address,
                X1.address, X2.address, min, max, 
                Y.lengthv, Y.lastDim());
        if(sync) { sc.sync(); if(Y1 != X1) Y1.delete(); if(Y2 != X2) Y2.delete(); }
        else { Y.setSyncer(Syncer.dual(sc, ()->{  if(Y1 != X1) Y1.delete(); if(Y2 != X2) Y2.delete(); })); }
        return Y;
    }
    
    public Tensor equal_abs_int8(Tensor X1, Tensor X2, byte min, byte max) {
        return equal_abs_int8(true, X1, X2, min, max);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor equal_abs_int8(boolean likeX1, Tensor X1, Tensor X2, byte min, byte max)
    {
        if(check) { require_int8(X1); require_int8(X2); }
        
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D_int8(Y.address, 
                X1.address, X2.address, min, max,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
     
    public Tensor equal_abs_int32(Tensor X1, Tensor X2, int min, int max) {
        return equal_abs_int32(true, X1, X2, min, max);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor equal_abs_int32(boolean likeX1, Tensor X1, Tensor X2, int min, int max)
    {
        if(check) { require_int32(X1); require_int32(X2); }
        
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.equal_abs2D_int32(Y.address, 
                X1.address, X2.address, min, max,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater">
    public Tensor gt(boolean inplace, Tensor X, float v) {//X > v => X - v > 0
        return linear_greater(inplace, 1.0f, X, -v); 
    }
    public Tensor lt(boolean inplace, Tensor X, float v) {//X < v => X - v < 0 -> -X + v > 0
        return linear_greater(inplace, -1.0f, X, v);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_greater(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = (inplace? X : this.empty(X.dim)).c();
        Syncer sc = core.linear_greater2D(Y.address, 
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater2">
    public Tensor gt2(Tensor X1, Tensor X2) { 
        return linear_greater2(X1, X2, 1.0f, -1.0f, 0.0f); 
    }
    public Tensor gt2(boolean likeX1, Tensor X1, Tensor X2) {//X1 > X2 -> X1 - X2 > 0
        return linear_greater2(likeX1, X1, X2, 1.0f, -1.0f, 0.0f);
    }
    
    public Tensor lt2(Tensor X1, Tensor X2) {
        return linear_greater2(X1, X2, -1.0f, 1.0f, 0.0f);
    }
    public Tensor lt2(boolean likeX1, Tensor X1, Tensor X2) {//X1 < X2 -> X1 - X2 < 0 -> -X1 + X2 > 0
        return linear_greater2(likeX1, X1, X2, -1.0f, 1.0f, 0.0f);
    }
    
    public Tensor linear_greater2(Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) {
        return linear_greater2(true, X1, X2, alpha, beta, gamma);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_greater2(boolean likeX1, Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) { equals_valueStructure(X1, "X1", X2, "X2"); }
        Tensor Y = this.empty(likeX1? X1.dim : X2.dim).c();
        Syncer sc = core.linear_greater_dual2D(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    public Tensor sadd(boolean inplace, Tensor X, float C) {
        return linear(inplace, 1.0f, X, C);
    }
    public Tensor ssub(boolean inplace, Tensor X, float C) {
        return linear(inplace, 1.0f, X, -C);
    }
    public Tensor smul(boolean inplace, Tensor X, float C) {
        return linear(inplace, C, X, 0.0f);
    }
    public Tensor sdiv(boolean inplaced, Tensor X, float C) { 
        return linear(inplaced, 1.0f / C, X, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear(boolean inplace, float alpha, Tensor X, float beta)
    {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.linear2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to dtype">
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_int8_to_dtype(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = this.empty(X.dim).c();
        Syncer sc = core.linear2D_int8_to_dtype(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        if(!inplace) { if(sync) sc.sync(); Y.setSyncer(sc); return Y; }
        if(sync) { sc.sync(); delete(X);}
        else Y.setSyncer(Syncer.dual(sc, ()->{ delete(X); }));
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_dtype_to_int8(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = this.empty_int8(X.dim).c();
        Syncer sc = core.linear2D_dtype_to_int8(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        if(!inplace) { if(sync) sc.sync(); else Y.setSyncer(sc); return Y; }
        if(sync) {sc.sync(); this.delete(X);}
        else Y.setSyncer(Syncer.dual(sc, ()-> { delete(X); } ));
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to dtype">
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_int32_to_dtype(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = this.empty(X.dim).c();
        Syncer sc = core.linear2D_int32_to_dtype(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        if(!inplace) { if(sync) sc.sync(); Y.setSyncer(sc); return Y; }
        if(sync) { sc.sync(); delete(X);}
        else Y.setSyncer(Syncer.dual(sc, ()->{ delete(X); }));
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_dtype_to_int32(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = this.empty_int32(X.dim).c();
        Syncer sc = core.linear2D_dtype_to_int32(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        if(!inplace) { if(sync) sc.sync(); else Y.setSyncer(sc); return Y; }
        if(sync) {sc.sync(); this.delete(X);}
        else Y.setSyncer(Syncer.dual(sc, ()-> { delete(X); } ));
        return Y;
    }
    //</editor-fold>
    
    public Tensor to_dtype(boolean inplace, Tensor X) {
        if(X.dataType.equals(core.dataType_int32())) 
            return linear_int32_to_dtype(inplace, 1.0f, X, 0.0f);
        if(X.dataType.equals(core.dataType_int8()))
            return linear_int8_to_dtype(inplace, 1.0f, X, 0.0f);
        return inplace? X : copy(X);
    }
    
    public Tensor to_dtype(boolean inplace, float alpha, Tensor X, float beta) {
        if(X.dataType.equals(core.dataType_int32())) 
            return linear_int32_to_dtype(inplace, alpha, X, beta);
        if(X.dataType.equals(core.dataType_int8()))
            return linear_int8_to_dtype(inplace, alpha, X, beta);
        return linear(inplace, alpha, X, beta);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_2out">
    public Tensor[] sadd_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, 0, C1, 0, C2);
    }
    
    public Tensor[] ssub_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, 0, -C1, 0, -C2);
    }
     
    public Tensor[] smul_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, C1, 0, C2, 0);
    }
    
    public Tensor[] sdiv_2out(boolean inplace, Tensor X, float C1, float C2) {
        return linear_2out(inplace, X, (1.0f / C1), 0, (1.0f / C2), 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] linear_2out(boolean inplace, Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2) 
    {
         if(check) { require_dataType(X); }
        
        Tensor Y1 = (inplace? X : this.empty(X.dim));
        Tensor Y2 = this.empty(X.dim);
        
        Syncer sc = core.linear_dual_out2D(
                Y1.c().address,//result0
                Y2.c().address,//result1
                X.address,
                alpha1, beta1, 
                alpha2, beta2,
                X.lengthv, X.lastDim());
      
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[] { Y1, Y2 };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear2">
    public Tensor sub(boolean inpalce, Tensor X1, Tensor X2) {
        return linear2(inpalce, X1, X2, 1.0f, -1.0f, 0.0f);
    }
    public Tensor add(boolean inplace, Tensor X1, Tensor X2) {
        return linear2(inplace, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor add(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2(inplace, X1, X2, alpha, beta, 0.0f);
    }
    
    public Tensor linear2(boolean inplace, Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) {//default likeX1
        return linear2(inplace, true, X1, X2, alpha, beta, gamma);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2(boolean inplace, boolean likeX1, 
            Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma)
    {
        if(check) { 
            if(check) { require_dataType(X1); }
            if(check) { require_dataType(X2); }
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = inplace? 
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c();
        
        Syncer sc = core.linear_dual2D(Y.address,
                X1.address, X2.address,
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_row">
    public Tensor sub_row(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_row(inplace, X1, X2, 1.0f, -1.0f, 0);
    }
    public Tensor add_row(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_row(inplace, X1, X2, 1.0f, 1.0f, 0);
    }
    public Tensor add_row(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_row(inplace, X1, X2, alpha, beta, 0);
    }
    
    public Tensor linear2_row(boolean inplace, Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) 
    {
        if(check) {
            if(check) { require_dataType(X1); }
            if(check) { require_dataType(X2); }
            check_row(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc= core.linear_dual2D_row(Y.address, 
                X1.address,
                X2.address, X2.lengthv, //X2.lengthv = row_lengthv
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear2_field">
    public Tensor sub_field(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_field(inplace, X1, X2, 1.0f, -1.0f, 0.0f);
    }
    public Tensor add_field(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_field(inplace, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor add_field(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return linear2_field(inplace, X1, X2, alpha, beta, 0.0f);
    }
    
    public Tensor linear2_field(boolean inplace, Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if(check) { 
            if(check) { require_dataType(X1); }
            if(check) { require_dataType(X2); }
            check_field(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.linear_dual2D_field(Y.address, 
                X1.address, 
                X2.address, X2.length, //X2.length = field_length
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_summary">
    public Tensor mean(boolean inplace, Collection<Tensor> Xs) {
        float alpha = (float) (1.0 / Xs.size());
        return linear_summary(inplace, alpha, 0.0f, Xs);
    }
    public Tensor summary(boolean inplace, Collection<Tensor> Xs) {
        return linear_summary(inplace, 1.0f, 0.0f, Xs);//Y = sum(Xs[i])
    }
    public Tensor summary(boolean inplace, float alpha, Collection<Tensor> Xs) {
        return linear_summary(inplace, alpha, 0.0f, Xs);//Y = alpha * sum(Xs[i])
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_summary(boolean inplace, //inplace: Xs[0]
           float alpha, float beta, Collection<Tensor> Xs)
    {
        int Xs_size = Xs.size();
        Iterator<Tensor> iter = Xs.iterator(); Tensor X0 = iter.next();
        if(Xs_size == 1) { return linear(inplace, alpha, X0, beta); }
        
        if(check) { 
            require_dataType(Xs);
            equals_valueStructure("Xs", X0, iter);
        }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs_size]; int index = 0;
        for(Tensor X : Xs) addrs[index++] = X.address;
        
        Syncer sc = core.linear_summary2D(inplace, Y.c().address, 
                alpha, beta, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor mean(boolean inplace, Tensor... Xs) {
        float alpha = (float) (1.0 / Xs.length);
        return linear_summary(inplace, alpha, 0.0f, Xs);//Y = sum(Xs[i])
    }
    public Tensor summary(boolean inplace, Tensor... Xs) {
        return linear_summary(inplace, 1.0f, 0.0f, Xs);//Y = sum(Xs[i])
    }
    public Tensor summary(boolean inplace, float alpha, Tensor... Xs) {
        return linear_summary(inplace, alpha, 0.0f, Xs);//Y = alpha * sum(Xs[i])
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_summary(boolean inplace, 
            float alpha, float beta, Tensor... Xs)
    {
        Tensor X0 = Xs[0];
        if(Xs.length == 1) { return linear(inplace, alpha, X0, beta); }
        
        if(check) { 
            require_dataType(Xs);
            equals_valueStructure("Xs", Xs);
        }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs.length]; 
        for(int i=0; i<Xs.length; i++) addrs[i] = Xs[i].address;
        
        Syncer sc = core.linear_summary2D(inplace, Y.c().address, 
                alpha, beta, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    public Tensor square(boolean inplace, Tensor X) { return quadratic(inplace, X, 1.0f, 0.0f, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic(boolean inplace, Tensor X, float alpha, float beta, float gamma)
    {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.quadratic2D(Y.address,
                X.address, alpha, beta, gamma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic_deltaX(boolean inplace, Tensor deltaY, Tensor X, float alpha, float beta)
    {
        if(check) { 
            require_dataType(X);
            require_dataType(deltaY);
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.quadratic2D_deltaX(deltaX.address,
                deltaY.address, 
                X.address, alpha, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic2">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor var(boolean inplace, Tensor X_sqmean, Tensor X_mean) {//X_sqmean - X_mean^2
        return quadratic2(inplace, X_sqmean, X_mean, 0, 0, -1.0f, 1.0f, 0, 0);
    }
    
    public Tensor mul(boolean inplace, Tensor X1, Tensor X2) { 
        return quadratic2(inplace, X1, X2, 0, 1.0f, 0, 0, 0, 0);
    }
    public Tensor mul(boolean inplace, float alpha, Tensor X1, Tensor X2) { 
        return quadratic2(inplace, X1, X2, 0, alpha, 0, 0, 0, 0);
    }
    
    public Tensor squareAdd(boolean inplace, Tensor X1, Tensor X2) { 
        return quadratic2(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0); 
    }
    public Tensor squareAdd(boolean inplace, Tensor X1, Tensor X2, float alpha, float beta) {
        return quadratic2(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    public Tensor quadratic2(boolean inplace, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return quadratic2(inplace, true, X1, X2, k11, k12, k22, k1, k2, C);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2(boolean inplace, boolean likeX1, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = inplace? 
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c();
        
        Syncer sc = core.quadratic_dual2D(Y.address, 
                X1.address, X2.address,
                k11, k12, k22,
                k1, k2, C, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] quadratic2_deltaX(boolean inplace, Tensor deltaY,
            Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2)
    {
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(deltaY, "deltaY", X2, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
        
        Tensor deltaX1 = (inplace?  deltaY : this.empty(X1.dim));
        Tensor deltaX2 = this.empty(X2.dim);
        
        Syncer sc = core.quadratic_dual2D_deltaX(
                deltaX1.c().address,
                deltaX2.c().address,
                deltaY.address,
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, 
                deltaY.lengthv, deltaY.lastDim());
       if(sync) sc.sync(); else {
           deltaX1.setSyncer(sc);
           deltaX2.setSyncer(sc);
       }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_row">
    public Tensor mul_row(boolean inplace, Tensor X1, Tensor X2) {
        return quadratic2_row(inplace, X1, X2, 0, 1.0f, 0, 0, 0, 0);
    }
    public Tensor mul_row(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return quadratic2_row(inplace, X1, X2, 0, alpha, 0, 0, 0, 0);
    }
    public Tensor squareSum_row(boolean inplace, Tensor X1, Tensor X2) {
        return quadratic2_row(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0);
    }
    public Tensor squareSum_row(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return quadratic2_row(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2_row(boolean inplace, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            check_row(X1, "X1", X2, "X2"); 
        }

        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.quadratic_dual2D_row(Y.address,
                X1.address,
                X2.address, X2.lengthv, //X2.lengthv = row_lengthv
                k11, k12, k22,
                k1, k2, C, 
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2_field">
    public Tensor mul_field(boolean inplaced, Tensor X1, Tensor X2) {
        return quadratic2_field(inplaced, X1, X2, 0, 1.0f, 0, 0, 0, 0);
    }
    public Tensor mul_field(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return quadratic2_field(inplace, X1, X2, 0, alpha, 0, 0, 0, 0);
    }
    public Tensor squareSum_field(boolean inplace, Tensor X1, Tensor X2) {
        return quadratic2_field(inplace, X1, X2, 1.0f, 0, 1.0f, 0, 0, 0);
    }
    public Tensor squareSum_field(boolean inplace, float alpha, Tensor X1, float beta, Tensor X2) {
        return quadratic2_field(inplace, X1, X2, alpha, 0, beta, 0, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic2_field(boolean inplace, Tensor X1, Tensor X2, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            check_field(X1, "X1", X2, "X2");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.quadratic_dual2D_field(Y.address, 
                X1.address,
                X2.address, X2.length,//X2.length = field_length
                k11, k12, k22,
                k1, k2, C,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_summary">
    public Tensor square_summary(boolean inplace, Collection<Tensor> Xs) {
        return quadratic_summary(inplace, 1.0f, 0, 0, Xs);//Y = sum(Xs[i]^2)
    }
    public Tensor square_summary(boolean inplace, float alpha, Collection<Tensor> Xs) {
        return quadratic_summary(inplace, alpha, 0, 0, Xs);//Y = alpha*sum(Xs[i]^2)
    }
    public Tensor quadratic_summary(boolean inplace, //inplace: Xs[0]
           float alpha, float beta, float gamma, Collection<Tensor> Xs)
    {
        int Xs_size = Xs.size();
        Iterator<Tensor> iter = Xs.iterator(); Tensor X0 = iter.next();
        if(Xs_size == 1) { return quadratic(inplace, X0, alpha, beta, gamma); }
        
        if(check) { 
            require_dataType(Xs);
            equals_valueStructure("Xs", X0, iter); 
        }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs_size]; int index = 0;
        for(Tensor X : Xs) addrs[index++] = X.address;
        
        Syncer sc = core.quadratic_summary2D(inplace, Y.c().address, 
                alpha, beta, gamma, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor square_summary(boolean inplace, Tensor... Xs) {
        return quadratic_summary(inplace, 1.0f, 0, 0, Xs);//Y = sum(Xs[i]^2)
    }
    public Tensor square_summary(boolean inplace, float alpha, Tensor... Xs) {
        return quadratic_summary(inplace, alpha, 0, 0, Xs);//Y = alpha*sum(Xs[i]^2)
    }
    public Tensor quadratic_summary(boolean inplace, 
            float alpha, float beta, float gamma, Tensor... Xs)
    {
        Tensor X0 = Xs[0];
        if(Xs.length == 1) { return quadratic(inplace, X0, alpha, beta, gamma); }
        
        if(check) { equals_valueStructure("Xs", Xs); }
        
        Tensor Y = (inplace? X0 : this.empty(X0.dim));
        long[] addrs = new long[Xs.length]; 
        for(int i=0; i<Xs.length; i++) addrs[i] = Xs[i].address;
        
        Syncer sc = core.quadratic_summary2D(inplace, Y.c().address, 
                alpha, beta, gamma, addrs,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    public Tensor rpl(boolean inplace, Tensor X) { return rpl(inplace, 1.0f, X, 0.0f, 0.0f); }
    public Tensor rpl(boolean inplace, float alpha, Tensor X) {
         return rpl(inplace, alpha, X, 0.0f, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")// alpha / (X + beta) + gamma
    public Tensor rpl(boolean inplace, float alpha, Tensor X, float beta, float gamma)
    {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.rpl2D(Y.address, 
                alpha, X.address, beta, gamma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rpl_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha, float gamma)
    {
        if(check) { 
            require_dataType(Y);
            require_dataType(deltaY);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.rpl2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, gamma, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    public Tensor div(boolean inplace, Tensor X1, Tensor X2) {
        return div(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div(boolean inplace, float alpha, Tensor X1, Tensor X2) {//Y = alpha *  X1 / X2
        return div(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    
    public Tensor div(boolean inplace, //Y = (alpha1*X1 + beta1) / (alpha1*X2 + beta2) + gamma
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma) {
        return div(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2, gamma);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor div(boolean inplace, boolean likeX1,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = inplace?
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c();
        
        Syncer sc = core.div2D(Y.address, 
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                gamma, X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] div_deltaX(boolean inplace, Tensor deltaY, 
            Tensor X1, float alpha1, float beta1,
            Tensor X2, float alpha2, float beta2)
    {
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(deltaY, "deltaY", X1, "X1");
            equals_valueStructure(deltaY, "deltaY", X2, "X2");
        }
        
        Tensor deltaX1 = (inplace? deltaY : this.empty(X1.dim));
        Tensor deltaX2 = (inplace? deltaY : this.empty(X2.dim));
        
        Syncer sc = core.div2D_deltaX(
                deltaX1.c().address, 
                deltaX2.c().address, 
                deltaY.address,
                X1.address, alpha1, beta1, 
                X2.address, alpha2, beta2, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else {
            deltaX1.setSyncer(sc);
            deltaX2.setSyncer(sc);
        }
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div_row">
    public Tensor div_row(boolean inplace, Tensor X1, Tensor X2) {
        return div_row(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div_row(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return div_row(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor div_row(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            check_row(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.div2D_row(Y.address, 
                alpha1, X1.address, beta1, 
                alpha2, X2.address, beta2, 
                gamma, X2.lengthv,//X2.lengthv = row_lengthv
                X1.lengthv, X1.lastDim());
         if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div_field">
    public Tensor div_field(boolean inplace, Tensor X1, Tensor X2) {
        return div_field(inplace, 1.0f, X1, 0, 1.0f, X2, 0, 0);
    }
    public Tensor div_field(boolean inplace, float alpha, Tensor X1, Tensor X2) {
        return div_field(inplace, alpha, X1, 0, 1.0f, X2, 0, 0);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor div_field(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2,
            float gamma)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            check_field(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim).c());
        Syncer sc = core.div2D_field(Y.address, 
                alpha1, X1.address, beta1, 
                alpha2, X2.address, beta2, 
                gamma, X2.length, //X2.length = field_length
                X1.lengthv, X1.lastDim());
         if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="add_div_row">
    public Tensor normalize_row(boolean inplace, Tensor X, Tensor X_field_mean, Tensor X_field_std) {
        return add_div_row(inplace, X, X_field_std, X_field_mean, 1.0f, 1.0f, 0.0f, 0.0f);
    }
    
    public Tensor add_div_row(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {//(X1 + X2) / X3
        return add_div_row(inplace, X1, X2, X3, 1.0f, 1.0f, 0.0f, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase") //(alpha*X1 + beta*X2 + gamma) / (X3 + delta)
    public Tensor add_div_row(boolean inplace, Tensor X1, Tensor X2, Tensor X3,
            float alpha, float beta, float gamma, float delta)
    {
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            require_dataType(X3);
            check_row(X1, "X1", X2, "X2");
            check_row(X1, "X1", X3, "X3");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim(0)).c());
        Syncer sc = core.add_div2D_row(Y.address, 
                X1.address,
                X2.address, 
                X3.address, X2.lengthv,//X3.lengthv = X2.lengthv = row_lengthv
                alpha, beta, gamma, delta,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="add_div_field">
    public Tensor normalize_field(boolean inplace, Tensor X, Tensor X_row_mean, Tensor X_row_var) {
        return add_div_field(inplace, X, X_row_mean, X_row_var);
    }
            
    public Tensor add_div_field(boolean inplace, Tensor X1, Tensor X2, Tensor X3) {//(X1 + X2) / X3
        return add_div_field(inplace, X1, X2, X3, 1.0f, 1.0f, 0.0f, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")//(alpha*X1 + beta*X2 + gamma) / (X3 + delta)
    public Tensor add_div_field(boolean inplace, Tensor X1, Tensor X2, Tensor X3,
            float alpha, float beta, float gamma, float delta)
    {
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            require_dataType(X3);
            check_field(X1, "X1", X2, "X2");
            check_field(X1, "X1", X3, "X3");
        }
        
        Tensor Y = (inplace? X1 : this.empty(X1.dim(0)).c());
        Syncer sc = core.add_div2D_field(Y.address, 
                X1.address,
                X2.address, 
                X3.address, X2.length, //X3.length = X2.length = field_length
                alpha, beta, gamma, delta,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, sqrt">
    @Passed("CudaFloat32EngieBase")
    public Tensor sign(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sign2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor ceil(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.ceil2D(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor floor(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.floor2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs">
    public Tensor abs(boolean inplace, Tensor X) {
        return abs(inplace, 1.0f, X, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor abs(boolean inplace, float alpha, Tensor X, float beta)
    {
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.abs2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor abs_deltaX(boolean inplace, Tensor deltaY,
            Tensor X, float alpha, float beta)
    {
        if(check) { equals_valueStructure(deltaY, "deltaY", X, "deltaX"); }
            
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.abs2D_deltaX(deltaX.address,
                deltaY.address, 
                X.address, alpha, beta,
                deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    @Passed("CudaFloat32EngieBase")
    public Tensor zero_nan(boolean inplace, Tensor X) {
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.zero_nan2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqrt">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqrt(boolean inplace, float alpha, Tensor X, float beta) 
    {
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqrt2D(Y.address, alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
   //deltaX = 0.5 * alpha * Y * deltaX
    @Passed("CudaFloat32EngieBase")
    public Tensor sqrt_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return div(inplace, 
                0.5f*alpha, deltaY, 0.0f, 
                1.0f, Y, 0.0f, 0.0f);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sqrt_quadratic2">
    public Tensor sqrt_quadratic2(boolean inplace,
            Tensor X1, Tensor X2, 
            float k11, float k12 ,float k22, 
            float k1, float k2, float C) {
        return sqrt_quadratic2(inplace, true, X1, X2, k11, k12, k22, k1, k2, C);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqrt_quadratic2(boolean inplace, boolean likeX1,
            Tensor X1, Tensor X2, 
            float k11, float k12 ,float k22, 
            float k1, float k2, float C)
    {
        if(check) { equals_valueStructure(X1, "X1", X2, "X2"); }
        
        Tensor Y = inplace? 
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c();
        
        Syncer sc = core.sqrt_quadratic_dual2D(Y.address,
                X1.address, X2.address,
                k11, k12, k22,
                k1, k2, C, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="min, max, clip"> 
    //<editor-fold defaultstate="collapsed" desc="min, min2">
    public Tensor min(boolean inplace, Tensor X, float vmin){  return min(inplace, 1, X, 0, vmin);}
    @Passed("CudaFloat32EngieBase")
    public Tensor min(boolean inplace, float alpha, Tensor X, float beta, float vmin) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.min2D(Y.address, 
                alpha, X.address, beta, vmin,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor min2(boolean inplace, Tensor X1, Tensor X2) { 
        return min2(inplace, 1.0f, X1, 0.0f, 1.0f, X2, 0.0f);
    }
    public Tensor min2(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        return min2(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor min2(boolean inplace, boolean likeX1, 
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = (inplace? 
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c());
        
        Syncer sc = core.min_dual2D(Y.address,
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="max, max2">
    public Tensor max(boolean inplace, Tensor X, float vmax) { return max(inplace, 1, X, 0, vmax); }
    @Passed("CudaFloat32EngieBase")
    public Tensor max(boolean inplace, float alpha, Tensor X, float beta, float vmax)
    {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.max2D(Y.address,
                alpha, X.address, beta, vmax,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor max2(boolean inplace, Tensor X1, Tensor X2) {
        return max2(inplace, 1.0f, X1, 0.0f, 1.0f, X2, 0.0f);
    }
    public Tensor max2(boolean inplace,
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2) {
        return max2(inplace, true, alpha1, X1, beta1, alpha2, X2, beta2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor max2(boolean inplace, boolean likeX1, 
            float alpha1, Tensor X1, float beta1,
            float alpha2, Tensor X2, float beta2)
    {
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2"); 
        }
        
        Tensor Y = inplace? 
                    (likeX1? X1 : X2) : 
                    this.empty(likeX1? X1.dim : X2.dim).c();
        
        Syncer sc = core.max_dual2D(Y.address,
                alpha1, X1.address, beta1,
                alpha2, X2.address, beta2,
                X1.lengthv, X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="clip"> 
    public Tensor clip(boolean inplace, Tensor X, float vmin, float vmax) {
        return clip(inplace, 1.0f, X, 0.0f, vmin, vmax);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor clip(boolean inplace, float alpha, Tensor X, float beta, float vmin, float vmax) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.clip2D(Y.address,
                alpha, X.address, beta, vmin, vmax,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    //<editor-fold defaultstate="collapsed" desc="BP: exp">
    @Passed("CudaFloat32EngieBase")
    public Tensor exp(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.exp2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    //deltaX = alpha * deltaY * Y
    public Tensor exp_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return mul(inplace, alpha, deltaY, Y);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: log">
    public Tensor log(boolean inplace, Tensor X) { return log(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor log(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.log2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor log_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        if(check) { 
            require_dataType(Y);
            require_dataType(deltaY);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.log2D_deltaX(deltaX.address,
                deltaY.address, 
                Y.address, alpha,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    @Passed("CudaFloat32EngieBase")
    public Tensor relu(boolean inplace, Tensor X) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.relu2D(Y.address, X.address,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor relu_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.relu2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor relu_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.relu2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu(boolean inplace, Tensor X, float negative_slope) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.leakyRelu2D(Y.address, 
                X.address, negative_slope,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu_deltaX_v1(boolean inplace, Tensor deltaY, 
            Tensor Y, float negative_slope) 
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.leakyRelu2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, negative_slope, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor leakyRelu_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float negative_slope)
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.leakyRelu2D_deltaX_v2(deltaX.address,
                deltaY.address,
                X.address, negative_slope,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    @Passed("CudaFloat32EngieBase")
    public Tensor elu(boolean inplace, Tensor X, float alpha, float negative_slope) {
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.elu2D(Y.address,
                X.address, alpha, negative_slope, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor elu_deltaX_v1(boolean inplace, Tensor deltaY,
            Tensor Y, float alpha, float negative_slope)
    {
       if(check) { 
           require_dataType(deltaY);
           require_dataType(Y);
           equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
       }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.elu2D_deltaX_v1(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, negative_slope,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor elu_deltaX_v2(boolean inplace, Tensor deltaY, 
            Tensor X, float alpha, float negative_slope)
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.elu2D_deltaX_v2(deltaX.address,
                deltaY.address, 
                X.address, alpha, negative_slope, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus(boolean inplace, Tensor X) {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.softPlus2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus_deltaX_v1(boolean inplaced, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplaced ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softPlus2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor softplus_deltaX_v2(boolean inplaced, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplaced ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softPlus2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh(boolean inplace, Tensor X) {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.tanh2D(Y.address,
                X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tanh2D_deltaX_v1(deltaX.address,
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tanh_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tanh2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid(boolean inplace, Tensor X) {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.sigmoid2D(Y.address, X.address, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_deltaX_v1(boolean inplace, Tensor deltaY, Tensor Y) {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sigmoid2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_deltaX_v2(boolean inplace, Tensor deltaY, Tensor X) {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sigmoid2D_deltaX_v2(deltaX.address, 
                deltaY.address,
                X.address, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    public Tensor softmax(boolean inplace, Tensor X) { return softmax(inplace, X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax(boolean inplace, Tensor X, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
            if(X.length % features != 0) throw new IllegalArgumentException("softmax: Illgal features");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.softmax2D(Y.address,
                X.address, features,
                X.lengthv, X.lastDim());//X.lastDim = width = mem_width
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y) {
        return softmax_deltaX(inplace, deltaY, Y, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y, int features) {
        if(features == -1) features = Y.lastDim();
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must > 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must > 1");
            if(deltaY.length % features != 0) throw new IllegalArgumentException("softmax: Illgal features");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.softmax2D_deltaX(deltaX.address, 
                deltaY.address,
                Y.address, features,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: log_softmax">
    public Tensor log_softmax(boolean inplace, Tensor X) { return log_softmax(inplace, X, - 1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor log_softmax(boolean inplace, Tensor X, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
            if(X.length % features != 0) throw new IllegalArgumentException("softmax: Illgal features");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.logsoftmax2D(Y.address,
                X.address, features,
                X.lengthv, X.lastDim());//X.lastDim = width = mem_width
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor log_softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y) {
        return log_softmax_deltaX(inplace, deltaY, Y, -1);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor log_softmax_deltaX(boolean inplace, Tensor deltaY, Tensor Y, int features) {
        if(features == -1) features = Y.lastDim();
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must > 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must > 1");
            if(deltaY.length % features != 0) throw new IllegalArgumentException("softmax: Illgal row_length");
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.logsoftmax2D_deltaX(deltaX.address, 
                deltaY.address,
                Y.address, features,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin, cos">
    public Tensor cos(boolean inplace, Tensor X) { return sin(inplace, 1.0f, X, HALF_PI); }
    public Tensor cos(boolean inplace, float alpha, Tensor X, float beta) {
        return sin(inplace, alpha, X, beta + HALF_PI);
    }
    public Tensor cos_deltaX(boolean inplaced, Tensor deltaY, 
            Tensor X, float alpha, float beta)  {
        return sin_deltaX(inplaced, deltaY, X, alpha, beta + HALF_PI);
    }
    
    public Tensor sin(boolean inplace, Tensor X) { return sin(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor sin(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.sin2D(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sin_deltaX(boolean inplace, Tensor deltaY, 
            Tensor X, float alpha, float beta) 
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(X);
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sin2D_deltaX(deltaX.address, 
                deltaY.address, 
                X.address, alpha, beta, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: tan, cot">
    public Tensor cot(boolean inplace, Tensor X) { return tan(inplace, -1.0f, X, - HALF_PI); }
    public Tensor cot(boolean inplace, float alpha, Tensor X, float beta) {
        return tan(inplace, -alpha, X, -beta - HALF_PI);
    }
    public Tensor cot_deltaX(boolean inplace, Tensor deltaY, Tensor Y, float alpha) {
        return tan_deltaX(inplace, deltaY, Y, -alpha);
    }
    
    public Tensor tan(boolean inplace, Tensor X) { return tan(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tan(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); }
        
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.tan2D(Y.address,
                alpha, X.address, beta, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor tan_deltaX(boolean inplace, Tensor deltaY, 
            Tensor Y, float alpha)
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.tan2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin">
    @Passed("CudaFloat32EngieBase")
    public Tensor halfSin(boolean inplace, float Amp, float alpha, Tensor X, float beta) {
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.halfSin2D(Y.address,
                Amp, alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor halfSin_deltaX(boolean inplace, Tensor deltaY, 
            Tensor Y, float Amp, float alpha)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.halfSin2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, Amp, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin2D">
    //arccos(X) + arcsin(X) = 0.5pi
    public Tensor arcsin(boolean inplace, Tensor X) { return arcsin(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor arcsin(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); } 
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.arcsin2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor arcsin_deltaX(boolean inplaced, Tensor deltaY,
            Tensor Y, float alpha)
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplaced ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.arcsin2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">
    //arctan(X) + arccot(X) = 0.5pi
    public Tensor arctan(boolean inplaced, Tensor X) { return arctan(inplaced, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor arctan(boolean inplace, float alpha, Tensor X, float beta) {
        if(check) { require_dataType(X); } 
        Tensor Y = (inplace ? X : this.empty(X.dim).c());
        Syncer sc = core.arctan2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor arctan_deltaX(boolean inplace, Tensor deltaY, 
            Tensor Y, float alpha) 
    {
        if(check) { 
            require_dataType(deltaY);
            require_dataType(Y);
            equals_valueStructure(deltaY, "deltaY", Y, "Y"); 
        }
        
        Tensor deltaX = (inplace ? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.arctan2D_deltaX(deltaX.address, 
                deltaY.address, 
                Y.address, alpha, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss functions">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    public Tensor L1(Tensor Yh, Tensor Y) { return L1(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L1(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) {
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L1_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor L1_deltaYh(Tensor Yh, Tensor Y)  { return L1_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L1_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L1_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    public Tensor L2(Tensor Yh, Tensor Y) { return L2(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L2(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) {
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L2_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor L2_deltaYh(Tensor Yh, Tensor Y)  { return L2_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor L2_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.L2_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: smoothL1">
    public Tensor smoothL1(Tensor Yh, Tensor Y) { return smoothL1(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor smoothL1(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.smoothL1_2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor smoothL1_deltaYh(Tensor Yh, Tensor Y)  { return smoothL1_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor smoothL1_deltaYh(boolean dimLikeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor deltaYh = this.empty(dimLikeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.smoothL1_2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    public Tensor binaryCrossEntropy(Tensor Yh, Tensor Y) {
        return binaryCrossEntropy(true, Yh, Y, 1.0f, 1.0f);
    }
    public Tensor binaryCrossEntropy(Tensor Yh, Tensor Y, float alpha, float beta) {
        return binaryCrossEntropy(true, Yh, Y, alpha, beta);
    }
    public Tensor binaryCrossEntropy(boolean likeYh, Tensor Yh, Tensor Y) {
        return binaryCrossEntropy(likeYh, Yh, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor binaryCrossEntropy(boolean likeYh, Tensor Yh, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.binaryCrossEntropy2D(L.address, 
                Y.address, Yh.address,
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor binaryCrossEntropy_deltaYh(Tensor Yh, Tensor Y)  {
        return binaryCrossEntropy_deltaYh(true, Yh, Y, 1.0f, 1.0f);
    }
    public Tensor binaryCrossEntropy_deltaYh(Tensor Yh, Tensor Y, float alpha, float beta) {
        return binaryCrossEntropy_deltaYh(true, Yh, Y, alpha, beta);
    }
    public Tensor binaryCrossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y)  {
        return binaryCrossEntropy_deltaYh(likeYh, Yh, Y, 1.0f, 1.0f);
    }      
    @Passed("CudaFloat32EngieBase")
    public Tensor binaryCrossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y"); 
        }
        
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.binaryCrossEntropy2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address, 
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    public Tensor sigmoid_binaryCrossEntropy(Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy(true, X, Y, 1.0f, 1.0f);
    }
    public Tensor sigmoid_binaryCrossEntropy(Tensor X, Tensor Y, float alpha, float beta) {
        return sigmoid_binaryCrossEntropy(true, X, Y, alpha, beta);
    }
    public Tensor sigmoid_binaryCrossEntropy(boolean likeX, Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy(likeX, X, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_binaryCrossEntropy(boolean likeX, Tensor X, Tensor Y, float alpha, float beta) {
        if(check) {
            require_dataType(X);
            require_dataType(Y);
            equals_valueStructure(X, "X", Y, "Y"); 
        }
        
        Tensor L = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.sigmoid_binaryCrossEntropy2D(L.address, 
                Y.address, X.address, 
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor sigmoid_binaryCrossEntropy_deltaX(Tensor X, Tensor Y)  {
        return sigmoid_binaryCrossEntropy_deltaX(true, X, Y, 1.0f, 1.0f);
    }
    public Tensor sigmoid_binaryCrossEntropy_deltaX(Tensor X, Tensor Y, float alpha, float beta)  {
        return sigmoid_binaryCrossEntropy_deltaX(true, X, Y, alpha, beta);
    }
    public Tensor sigmoid_binaryCrossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y) {
        return sigmoid_binaryCrossEntropy_deltaX(likeX, X, Y, 1.0f, 1.0f);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor sigmoid_binaryCrossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y, float alpha, float beta) {
        if(check) { 
            require_dataType(X);
            require_dataType(Y);
            equals_valueStructure(X, "X", Y, "Y"); 
        }
        
        Tensor deltaX = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.sigmoid_binaryCrossEntropy2D_deltaX(deltaX.address,
                Y.address, X.address,
                alpha, beta,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    public Tensor crossEntropy(Tensor Yh, Tensor Y) { return crossEntropy(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor crossEntropy(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y");
        }
        
        Tensor L = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.crossEntropy2D(L.address,
                Y.address, Yh.address, 
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
    
    public Tensor crossEntropy_deltaYh(Tensor Yh, Tensor Y) { return crossEntropy_deltaYh(true, Yh, Y); }
    @Passed("CudaFloat32EngieBase")
    public Tensor crossEntropy_deltaYh(boolean likeYh, Tensor Yh, Tensor Y) {
        if(check) { 
            require_dataType(Yh);
            require_dataType(Y);
            equals_valueStructure(Yh, "Yh", Y, "Y");
        }
        
        Tensor deltaYh = this.empty(likeYh? Yh.dim : Y.dim).c();
        Syncer sc = core.crossEntropy2D_deltaYh(deltaYh.address, 
                Y.address, Yh.address,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaYh.setSyncer(sc);
        return deltaYh;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax_crossEntropy">
    public Tensor softmax_crossEntropy(Tensor X, Tensor Y, int features) {
        return softmax_crossEntropy(true, X, Y, features);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy(boolean likeX, Tensor X, Tensor Y, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dataType(X);
            require_dataType(Y);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= 2");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(X.length % features != 0) throw new IllegalArgumentException("softmax: Illgal features");
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor L = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.softmax_crossEntropy2D(L.address, 
                Y.address, X.address, features,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else L.setSyncer(sc);
        return L;
    }
  
    public Tensor softmax_crossEntropy_deltaX(Tensor X, Tensor Y, int features)  {
        return softmax_crossEntropy_deltaX(true, X, Y, features);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy_deltaX(boolean likeX, Tensor X, Tensor Y, int features) {
        if(features == -1) features = X.lastDim();
        if(check) { 
            require_dataType(X);
            require_dataType(Y);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= 2");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(X.length % features != 0) throw new IllegalArgumentException("softmax: Illgal features");
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor deltaX = this.empty(likeX? X.dim : Y.dim).c();
        Syncer sc = core.softmax_crossEntropy2D_deltaX(deltaX.address,
                Y.address, X.address, features,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    //<editor-fold defaultstate="collapsed" desc="softmax_crossEntropy_deltaX_naive">
    @Passed("CudaFloat32EngieBase")
    public Tensor softmax_crossEntropy_deltaX_naive(boolean likeX, Tensor X, Tensor Y, int row_length)
    {
        if(check) { 
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(X, "X", Y, "Y");
        }
        
        Tensor deltaX = this.softmax(false, X, row_length).c();
        deltaX = this.add(true, 1.0f, deltaX, -1.0f, Y);
        return deltaX;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine(boolean inplace, Tensor X, Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(X);
            require_dataType(A);
            require_dataType(B);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.affine2D(Y.address,
                X.address, 
                A.address, B.address, A.lengthv,
                Y.lengthv, Y.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(A);
            require_dataType(B);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must >= 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals(Y.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
        }
        
        Tensor deltaA = this.empty(A.dim).c();
        Syncer sc = core.affine2D_deltaA_v1(deltaA.address,
                deltaY.address,
                Y.address, 
                A.address, B.address, A.lengthv,//A.lengthv = row_lengthv
                deltaY.lengthv, deltaY.lastDim());//width = deltaY.lastDim
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(A);
            require_dataType(B);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(Y.ndim() <= 1) throw new IllegalArgumentException("Y.ndim must >= 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals(Y.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        
        Syncer sc = core.affine2D_deltaAB_v1(
                deltaA.c().address,//result0
                deltaB.c().address,//result
                deltaY.address, Y.address,
                A.address, B.address, 
                A.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor affine_deltaA_v2(Tensor deltaY, Tensor X, int row_length)  {
        return field_mulSum(deltaY, X, row_length);
    }

    @Passed("CudaFloat32EngieBase")
    public Tensor[] affine_deltaAB_v2(Tensor deltaY, Tensor X, int row_length) {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
        }
        
        int width = X.lastDim(), height = row_length / width;
        int[] dimAB = (height > 1? //[height, width]
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor deltaA = this.empty(dimAB);
        Tensor deltaB = this.empty(dimAB);
        
        Syncer sc = core.affine2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//tesult1
                deltaY.address, 
                X.address,//deltaA.lengthv = row_lengthv
                deltaA.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
      
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqBatchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            require_dataType(A);
            require_dataType(B);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.sqBatchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps,
                A.address, B.address, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_mean);
            require_dataType(X_sqmean); 
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address, 
                Y.address, 
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(X.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must > 2");
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, 
                Y.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.sqBatchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sqBatchNorm_deltaA_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= than 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1? //[height, width]
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.sqBatchNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    public Tensor[] sqBatchNorm_deltaAB_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= than 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1? //[height, width]
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.sqBatchNorm2D_deltaAB_v2(deltaA.c().address, deltaB.c().address,
                deltaY.address,
                X.address, 
                X_mean.address, X_sqmean.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(deltaY); 
            require_dataType(Y);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            require_dataType(A); 
            require_dataType(B);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must > 2");
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.sqBatchNorm2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                Y.address,
                X_mean.address, X_sqmean.address, eps, 
                A.address, B.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] sqBatchNorm_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_sqmean);
            require_dataType(A);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_sqmean.lastDim(), "X_sqmean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_sqmean.length, "X_sqmean.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.sqBatchNorm2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                X.address, 
                X_mean.address, X_sqmean.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm(boolean inplace, Tensor X,
            Tensor X_mean, Tensor X_var, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            require_dataType(A);
            require_dataType(B);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim");
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
            equals(X_mean.length, "X_mean.length", B.length, "B.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.batchNorm2D(Y.address,
                X.address, 
                X_mean.address, X_var.address, eps,
                A.address, B.address, 
                X_mean.lengthv, X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v1(boolean inplace,
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_var); 
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address,
                Y.address, 
                X_var.address, eps,
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v2(boolean inplace,
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            if(X.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address, 
                X_mean.address, X_var.address, eps, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_var);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must > 2");
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_var.length, "X_var.length", A.length, "A.length");
            equals(X_var.length, "X_var.length", B.length, "B.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, 
                Y.address,
                X_var.address, eps, 
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.batchNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, 
                X.address,
                X_mean.address, X_var.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor batchNorm_deltaA_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= than 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1? //[height, width]
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.batchNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address,
                X.address,
                X_mean.address, X_var.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    public Tensor[] batchNorm_deltaAB_v2(Tensor deltaY, Tensor X, 
            Tensor X_mean, Tensor X_var, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            if(deltaY.ndim() <= 1) throw new IllegalArgumentException("deltaY.ndim must >= than 1");
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must >= than 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X.lastDim(), "X.lastDim", X_mean.lastDim(), "X_mean.lastDim");
            equals(X.lastDim(), "X.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
        }
        
        int width = deltaY.lastDim(), height = X_mean.length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1? //[height, width]
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.batchNorm2D_deltaAB_v2(
                deltaA.c().address,//result0
                deltaB.c().address,//result1
                deltaY.address,
                X.address, 
                X_mean.address, X_var.address, eps,
                X_mean.lengthv, deltaY.lengthv, width);
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[] { deltaA, deltaB };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): { deltaX, deltaA, deltaB }">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_gradients_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_var, float eps, 
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(deltaY); 
            require_dataType(Y);
            require_dataType(X_var);
            require_dataType(A); 
            require_dataType(B);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must > 2");
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must > 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim");
            equals(deltaY.lastDim(), "deltaY.lastDim", B.lastDim(), "B.lastDim");
            equals(X_var.length, "X_var.length", A.length, "A.length");
            equals(X_var.length, "X_var.length", B.length, "B.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(B.dim);
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm2D_gradients_v1(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address, 
                Y.address,
                X_var.address, eps, 
                A.address, B.address, 
                X_var.lengthv, deltaY.lengthv, deltaY.lastDim());

        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] batchNorm_gradients_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_mean, Tensor X_var, float eps, 
            Tensor A)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_mean);
            require_dataType(X_var);
            require_dataType(A);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_mean.lastDim(), "X_mean.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", X_var.lastDim(), "X_var.lastDim()");
            equals(deltaY.lastDim(), "deltaY.lastDim", A.lastDim(), "A.lastDim()");
            equals(X_mean.length, "X_mean.length", X_var.length, "X_var.length");
            equals(X_mean.length, "X_mean.length", A.length, "A.length");
        }
        
        Tensor deltaA = this.empty(A.dim);
        Tensor deltaB = this.empty(A.dim);//A.dim = B.dim
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        
        Syncer sc = core.batchNorm2D_gradients_v2(
                deltaX.address,//result0
                deltaA.c().address,//result1
                deltaB.c().address,//result2
                deltaY.address,
                X.address, 
                X_mean.address, X_var.address, eps, 
                A.address, 
                X_mean.lengthv, deltaY.lengthv, deltaY.lastDim());
        
        if(sync) sc.sync(); else {
            deltaX.setSyncer(sc);
            deltaA.setSyncer(sc);
            deltaB.setSyncer(sc);
        }
        return new Tensor[] { deltaX, deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm(boolean inplace, Tensor X,
            Tensor X_row_mean, Tensor X_row_sqmean, float eps)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);        
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.layerNorm2D(Y.address,
                X.address, 
                X_row_mean.address, 
                X_row_sqmean.address, eps,
                X_row_mean.length,//X_row_mean.length = field_length
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm(boolean inplace, Tensor X,
            Tensor X_row_mean, Tensor X_row_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals(X.lastDim(), "X.lastDim", A.lastDim(), "A.lastDim");
            equals(X.lastDim(), "X.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor Y = (inplace? X : this.empty(X.dim).c());
        Syncer sc = core.layerNorm2D(Y.address,
                X.address, 
                X_row_mean.address, 
                X_row_sqmean.address, eps,
                A.address, B.address, X_row_mean.length,//X_row_mean.length = field_length
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation: deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_row_mean, Tensor X_row_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v1(deltaX.address, 
                deltaY.address, Y.address, 
                X_row_mean.address,
                X_row_sqmean.address, eps, X_row_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_row_mean, Tensor X_row_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, X.address,
                X_row_mean.address, 
                X_row_sqmean.address, eps, X_row_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v1(boolean inplace, 
            Tensor deltaY, Tensor Y,//V1: holdY(), Y is not changed
            Tensor X_row_mean, Tensor X_row_sqmean, float eps,
            Tensor A, Tensor B)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(Y);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            require_dataType(A);
            require_dataType(B);
            if(Y.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals(Y.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals(Y.lastDim(), "Y.lastDim", B.lastDim(), "B.lastDim");
            equals(A.length, "A.length", B.length, "B.length");
            equals_valueStructure(deltaY, "deltaY", Y, "Y");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v1(deltaX.address,
                deltaY.address, Y.address, 
                X_row_mean.address,
                X_row_sqmean.address, eps,
                A.address, B.address, X_row_mean.length, 
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaX_v2(boolean inplace, 
            Tensor deltaY, Tensor X,//V2: holdX(), X is not changed
            Tensor X_row_mean, Tensor X_row_sqmean, float eps, 
            Tensor A)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            require_dataType(A);
            if(X.ndim() < 2) throw new IllegalArgumentException("Y.ndim must >= 2");
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals(X.lastDim(), "Y.lastDim", A.lastDim(), "A.lastDim");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_row_sqmean.length");
        }
        
        Tensor deltaX = (inplace? deltaY : this.empty(deltaY.dim).c());
        Syncer sc = core.layerNorm2D_deltaX_v2(deltaX.address, 
                deltaY.address, X.address,
                X_row_mean.address,
                X_row_sqmean.address, eps,
                A.address, X_row_mean.length,
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaX.setSyncer(sc);
        return deltaX;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaA_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaA_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] layerNorm_deltaAB_v1(Tensor deltaY, Tensor Y, Tensor A, Tensor B) {
        return affine_deltaAB_v1(deltaY, Y, A, B);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor layerNorm_deltaA_v2(Tensor deltaY,
            Tensor X, Tensor X_row_mean, Tensor X_row_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_sqmean.length");
        }
        
        int field_length = X_row_mean.length;
        int row_length = X.length / field_length;
        int width = deltaY.lastDim(), height = row_length / width;//X_mean.memstruc = A.mem_struc
        int[] dimA = (height > 1?
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dimA).c();
        
        Syncer sc = core.layerNorm2D_deltaA_v2(deltaA.address, 
                deltaY.address, X.address,
                X_row_mean.address, 
                X_row_sqmean.address, eps, field_length,//field_length = X_row_mean.length;
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else deltaA.setSyncer(sc);
        return deltaA;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] layerNorm_deltaAB_v2(Tensor deltaY,
            Tensor X, Tensor X_row_mean, Tensor X_row_sqmean, float eps)
    {
        if(check) {
            require_dataType(deltaY);
            require_dataType(X);
            require_dataType(X_row_mean);
            require_dataType(X_row_sqmean);
            if(deltaY.ndim() < 2) throw new IllegalArgumentException("deltaY.ndim must >= 2");
            if(X.ndim() < 2) throw new IllegalArgumentException("X.ndim must >= 2");
            if(X_row_mean.ndim() != 1) throw new IllegalArgumentException("X_mean.ndim != 1");
            if(X_row_sqmean.ndim() != 1) throw new IllegalArgumentException("X_sqmean.ndim != 1");
            equals_valueStructure(deltaY, "deltaY", X, "X");
            equals(X_row_mean.length, "X_mean.length", X_row_sqmean.length, "X_sqmean.length");
        }
        
        int field_length = X_row_mean.length;
        int row_length = X.length / field_length;
        int width = deltaY.lastDim(), height = row_length / width;//X_mean.memstruc = A.mem_struc
        int[] dim = (height > 1?
                new int[]{ height, width }:
                new int[]{ width });
        Tensor deltaA = this.empty(dim);
        Tensor deltaB = this.empty(dim);
        
        Syncer sc = core.layerNorm2D_deltaAB_v2(deltaA.c().address, deltaB.c().address, 
                deltaY.address, X.address, 
                X_row_mean.address,
                X_row_sqmean.address, eps, field_length, //field_length = X_row_mean.length;
                deltaY.lengthv, deltaY.lastDim());
        if(sync) sc.sync(); else { deltaA.setSyncer(sc); deltaB.setSyncer(sc); }
        return new Tensor[]{ deltaA, deltaB };
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot">
    public Tensor onehot(Tensor X, int num_class) { return onehot(X, 1.0f, 0.0f, num_class); }
    public Tensor onehot(Tensor X, float alpha, int num_class) {
        float beta = (1.0f - alpha) / (num_class - 1);
        return onehot(X, alpha, beta, num_class);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor onehot(Tensor X, float alpha, float beta, int num_class) {
        if(check) { if(num_class < 2) throw new IllegalArgumentException("num_class must >= 2"); }
        
        int dimX[] = X.dim, ndimX = dimX.length;//[dimX, num_class]
        int[] dimY = new int[ndimX + 1];
        for(int i=0; i<ndimX; i++) dimY[i] = dimX[i];
        dimY[ndimX] = num_class;
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = null; 
        if(X.dataType.equals(core.dataType_int32())) {
            sc = core.onehot2D_row_int32(Y.address, X.address,
                    alpha, beta, X.length,  //X.length = field_length
                    Y.lengthv, Y.lastDim());
        }
        else if(X.dataType.equals(core.dataType_int8())) {
            sc = core.onehot2D_row_int8(Y.address, X.address,
                    alpha, beta, X.length,
                    Y.lengthv, Y.lastDim());
        }
        else { throw new RuntimeException(String.format(
                "X.dataType[%s] != %s of %s", X.dataType, 
                core.dataType_int32(), 
                core.dataType_int8()));
        }
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="pix2tensor">
    @Passed("CudaFloat32EngieBase")
    public Tensor pix2tensor(boolean inplace, Tensor X)  {
        if(check) { require_int8(X); }
        Tensor Y = empty(X.dim).c();
        Syncer sc = core.pix2tensor2D(Y.address, X.address,
                Y.lengthv, Y.lastDim());
        
        if(!inplace) { if(sync) sc.sync(); else Y.setSyncer(sc); return Y; }
        if(sync) { sc.sync(); delete(X); }
        else Y.setSyncer(Syncer.dual(sc, ()-> { delete(X); }));
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    public Tensor sgd(Tensor W, Tensor deltaW, float lr) { 
        return add(true, 1.0f, W, -lr, deltaW);
    }
    
    public Tensor sgd(Tensor W, Collection<Tensor> gradients, float lr) {
        if(check) { 
            require_dataType(W);
            require_dataType(gradients);
            equals_valueStructure(W, "W", gradients, "W.gradients"); 
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        
        Syncer sc = core.sgd(W.address, gradsAddr, lr, W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Tensor deltaW, float lr)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.sgdmn(W.address, 
                V.address, momentum, dampen, nesterov, 
                deltaW.address, lr,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nestrov, 
            Collection<Tensor>  gradients, float lr)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.sgdmn(W.address,
                V.address, momentum, dampen, nestrov,
                gradsAddr, lr, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Tensor deltaW, float lr, 
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.sgdmn_decay(W.address, 
                V.address, momentum, dampen, nesterov, 
                deltaW.address, lr,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sgdmn(Tensor W, 
            Tensor V, float momentum, float dampen, float nesterov, 
            Collection<Tensor>  gradients, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.sgdmn_decay(W.address,
                V.address, momentum, dampen, nesterov,
                gradsAddr, lr, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Momentum">
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(deltaW);
            require_dataType(V);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.momentum2D(W.address, 
                V.address, a1, a2,
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Collection<Tensor>  gradients, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.momentum2D(W.address, 
                V.address, a1, a2,
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Momentum(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(V);
            require_dataType(W);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.momentum2D_decay(W.address, 
                V.address, a1, a2,
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor momentum(Tensor W, 
            Tensor V, float a1, float a2, 
            Collection<Tensor>  gradients, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.momentum2D_decay(W.address, 
                V.address, a1, a2,
                gradsAddr, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.rmsprop2D(W.address, 
                S.address, a1, a2, eps_t, 
                deltaW.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Collection<Tensor> gradients, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.rmsprop2D(W.address, 
                S.address, a1, a2, eps_t, 
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.rmsprop2D_decay(W.address,
                S.address, a1, a2, eps_t,
                deltaW.address, lr_t, L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor rmsprop(Tensor W, 
            Tensor S, float a1, float a2, float eps_t,
            Collection<Tensor> gradients, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc  = core.rmsprop2D_decay(W.address, 
                S.address, a1, a2, eps_t,
                gradsAddr, lr_t, 
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adam">
    //<editor-fold defaultstate="collapsed" desc="adam_type2">
    public Tensor adam_type2(Tensor W,
            Tensor V, float a1, float a2, float Uv,
            Tensor S, float b1, float b2, float eps, float Us,
            Tensor deltaW, float lr)
    {
         if(check) {
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D_type2(W.address, 
                V.address, a1, a2, Uv,
                S.address, b1, b2, eps, Us,
                deltaW.address, lr,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(deltaW);
            require_dataType(S);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> gradients, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.adam2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                gradsAddr, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adam2D_decay(W.address,
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adam(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> gradients, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.adam2D_decay(W.address,
                V.address, a1, a2,
                S.address, b1, b2, eps_t,
                gradsAddr, lr_t, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adamax2D(W.address,
                V.address, a1, a2,
                S.address, b1, eps, 
                W.address, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Collection<Tensor> gradients, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.adamax2D(W.address, 
                V.address, a1, a2,
                S.address, b1, eps,
                gradsAddr, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
       
        Syncer sc = core.adamax2D_decay(W.address,
                V.address, a1, a2, 
                S.address, b1, eps,
                deltaW.address, lr_t, 
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamax_decay(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float eps,
            Collection<Tensor> gradients, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.adamax2D_decay(W.address,
                V.address, a1, a2,
                S.address, b1, eps,
                gradsAddr, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Tensor deltaW, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(deltaW);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
        
        Syncer sc = core.adamW2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                deltaW.address, lr_t, lr,
                L1coef, L2coef, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); else W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamW(Tensor W,
            Tensor V, float a1, float a2,
            Tensor S, float b1, float b2, float eps_t,
            Collection<Tensor> gradients, float lr_t, float lr,
            float L1coef, float L2coef)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(gradients);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;
        Syncer sc = core.adamW2D(W.address,
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                gradsAddr, lr_t, lr,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Tensor deltaW, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(G);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adamod2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                G.address, c1, c2, 
                deltaW.address, lr_t,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    
     @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Collection<Tensor> gradients, float lr_t)
    {
        if(check) {
            require_dataType(W);
            require_dataType(V);
            require_dataType(S);
            require_dataType(G);
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;

        Syncer sc = core.adamod2D(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                G.address, c1, c2,
                gradsAddr, lr_t, 
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod(L1, L2)">
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Tensor deltaW, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", deltaW, "deltaW");
        }
         
        Syncer sc = core.adamod2D_decay(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t,
                G.address, c1, c2, 
                deltaW.address, lr_t,
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adamod(Tensor W,
            Tensor V, float a1, float a2, 
            Tensor S, float b1, float b2, float eps_t,
            Tensor G, float c1, float c2,
            Collection<Tensor> gradients, float lr_t,
            float L1coef, float L2coef)
    {
        if(check) {
            equals_valueStructure(W, "W", V, "V");
            equals_valueStructure(W, "W", S, "S");
            equals_valueStructure(W, "W", G, "G");
            equals_valueStructure(W, "W", gradients, "W.gradients");
        }
        
        long[] gradsAddr = new long[gradients.size()]; int index = 0;
        for(Tensor grad : gradients) gradsAddr[index++] = grad.address;

        Syncer sc = core.adamod2D_decay(W.address, 
                V.address, a1, a2, 
                S.address, b1, b2, eps_t, 
                G.address, c1, c2,
                gradsAddr, lr_t, 
                L1coef, L2coef,
                W.lengthv, W.lastDim());
        if(sync) sc.sync(); W.setSyncer(sc);
        return W;
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    //<editor-fold defaultstate="collapsed" desc="uniform">
    public Tensor Uniform(int... dim) {
        return uniform(this.empty(dim).c(), 0.0f, 1.0f);
    }
    public Tensor Uniform(Tensor X) {
        return uniform(X, 0.0f, 1.0f);
    }
     
    public Tensor uniform(float vmin, float vmax, int... dim) {
        return uniform(this.empty(dim).c(), vmin, vmax);
    }

    @Passed("CudaFloat32EngieBase")
    public Tensor uniform(Tensor X, float vmin, float vmax) 
    {
        if(check) { require_dataType(X); }
        Syncer sc = core.uniform2D(X.address, vmin, vmax, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sparse uniform">
    public Tensor Sparse_Uniform(Tensor X, float p) {
        return sparse_uniform(X, p, 0.0f, 1.0f);
    }
    public Tensor Sparse_Uniform(float p, int... dim) {
        return sparse_uniform(this.empty(dim).c(), p, 0.0f, 1.0f);
    }
    
    public Tensor sparse_uniform(float p, float vmin, float vmax, int... dim)  {
         return sparse_uniform(this.empty(dim).c(), p, vmin, vmax);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sparse_uniform(Tensor X, float p, float vmin, float vmax) 
    {
        if(check) { require_dataType(X); }
        Syncer sc = core.sparse_uniform2D(X.address, p, vmin, vmax, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="bernouli">
    public Tensor Bernouli(float p, int... dim) {
        return bernouli(this.empty(dim).c(), p, 1.0f , 0.0f);
    }
    public Tensor Bernouli(float p, float v1, float v2, int... dim)  {
        return bernouli(this.empty(dim).c(), p, v1, v2);
    }
    
    public Tensor bernouli(Tensor X, float p) { return bernouli(X, p, 1.0f , 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor bernouli(Tensor X, float p, float v1, float v2)  
    {
        if(check) { require_dataType(X); }
        Syncer sc = core.bernouli2D(X.address, p, v1, v2, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="bernouli_mul">
    public Tensor[] dropout(Tensor X, float nonzero_percent) {
        float p = nonzero_percent;
        float pr = 1.0f / p;
        return bernouli_mul(X, p, pr, 0);//X * bernouli(p, 1/p, 0)
    }
    
    public Tensor[] bernouli_mul(Tensor X, float p) {
        return bernouli_mul(X, p, 1.0f , 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor[] bernouli_mul(Tensor X, float p, float v1, float v2)  
    {
        if(check) { require_dataType(X); }
        
        Tensor Y = this.empty(X.dim);
        Tensor R = this.empty(X.dim);
        Syncer sc = core.bernouli2D_mul(
                Y.c().address,//result0
                R.c().address,//result1
                X.address, 
                p, v1, v2, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); R.setSyncer(sc); }
        return new Tensor[]{ Y, R };
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="gaussian">
    public Tensor Gaussian(int... dim) {
        return gaussian(this.empty(dim).c(), 0.0f, 1.0f);
    }
    public Tensor Gaussian(Tensor X) {
        return gaussian(X, 0.0f, 1.0f);
    }
    
    public Tensor gaussian(float mu, float sigma, int... dim) {
        return gaussian(this.empty(dim).c(), mu, sigma);
    } 
    
    @Passed("CudaFloat32EngieBase")
    public Tensor gaussian(Tensor X, float mu, float sigma) 
    {
        if(check) { require_dataType(X); }
        
        Syncer sc = core.gaussian2D(X.address, mu, sigma, 
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sparse gaussian">
    public Tensor Sparse_Gaussian(float p, int... dim) {
        return sparse_gaussian(this.empty(dim).c(), p, 0.0f, 1.0f);
    }
    
    public Tensor Sparse_Gaussian(Tensor X, float p) {
        return sparse_gaussian(X, p, 0.0f, 1.0f);
    }
    
    public Tensor sparse_gaussian(float p, float mu, float sigma, int... dim)  {
        return sparse_gaussian(this.empty(dim).c(), p, mu, sigma);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor sparse_gaussian(Tensor X, float p, float mu, float sigma) 
    {
        if(check) { require_dataType(X); }
        
        Syncer sc = core.sparse_gaussian2D(X.address, p, mu, sigma,
                X.lengthv, X.lastDim());
        if(sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: FanMode">
    public static class FanMode {
        public static final int fan_in = 0;
        public static final int fan_out = 1;
        public static final int fan_in_out = 2;
        public static final int default_value = fan_in;
    }
    
    public int fan_mode(String type) {
        type = type.toLowerCase();
        if("fan_in".equals(type)) return FanMode.fan_in;
        if("fan_out".equals(type)) return FanMode.fan_out;
        if("fan_in_out".equals(type)) return FanMode.fan_in_out;
        return FanMode.default_value;
    }
    
    public float fan(int fan_mode, int[] fans) {
        if(fan_mode == FanMode.fan_in) return fans[0];
        if(fan_mode == FanMode.fan_out) return fans[1];
        return (fans[0] + fans[1]) * 0.5f; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: Nonlinearity">
    public static class Nonlinearity  {
        public static final int sigmoid = 0;
        public static final int tanh = 1;
        public static final int relu = 2;
        public static final int leaky_relu = 3;
        public static final int elu = 4;
        public static final int default_value = leaky_relu;
    }
    
    public int nonlinearity(String type) {
        type = type.toLowerCase();
        if("sigmoid".equals(type)) return Nonlinearity.sigmoid;
        if("tanh".equals(type)) return Nonlinearity.tanh;
        if("relu".equals(type)) return Nonlinearity.relu;
        if("leaky_relu".equals(type)) return Nonlinearity.leaky_relu;
        if("elu".equals(type)) return Nonlinearity.elu;
        return Nonlinearity.default_value;
    }
    
    protected float gain(int nonlinearity, float... params) {
        if(nonlinearity == Nonlinearity.sigmoid) return 1.0f;
        if(nonlinearity == Nonlinearity.tanh) return 1.666667f;// 5/3
        if(nonlinearity == Nonlinearity.relu) return 1.414214f;//sqrt(2.0)
        if(nonlinearity == Nonlinearity.leaky_relu) {
            float k = 0.01f;
            if(params != null && params.length != 0) k = params[0];
            return (float) Math.sqrt(2.0 / (1.0 + k*k));
        }
        if(nonlinearity == Nonlinearity.elu) return 0.75f;//3.0 / 4
        return 1.0f;
    }
    //</editor-fold>
    
    public int xavier_fan_mode = FanMode.default_value;
    //<editor-fold defaultstate="collapsed" desc="xavier_uniform">
    public Tensor xavier_uniform(Tensor X, int[] fans) {
        return xavier_uniform(X, 1.0f, xavier_fan_mode, fans);
    }
    
    public Tensor xavier_uniform(Tensor X, int fan_mode, int[] fans) {
        return xavier_uniform(X, 1.0f, fan_mode, fans);
    }
    
    public Tensor xavier_uniform(Tensor X, float alpha, int fan_mode, int[] fans) {
        float fan = fan(fan_mode, fans);
        float std = (float) (1.0 / Math.sqrt(fan));
        float bound = alpha * 1.732051f * std;//sqrt(3) = 1.732051 
        return this.uniform(X, -bound, bound);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="xavier_gaussian">
    public Tensor xavier_gaussian(Tensor X, int[] fans) {
        return xavier_gaussian(X, 1.0f, xavier_fan_mode, fans);
    }
    
    public Tensor xavier_gaussian(Tensor X, int fan_mode, int[] fans) {
        return xavier_gaussian(X, 1.0f, fan_mode, fans);
    }
    
    public Tensor xavier_gaussian(Tensor X, float alpha, int fan_mode, int[] fans)
    {
        float fan = fan(fan_mode, fans);
        float std = (float) (1.0 / Math.sqrt(fan));
        float sigma = alpha * std;
        return this.gaussian(X, 0, sigma);
    }
    //</editor-fold>
    
    public int kaiming_fan_mode = FanMode.default_value;
    public int kaiming_no_linearity = Nonlinearity.default_value;
    //<editor-fold defaultstate="collapsed" desc="kaiming_uniform">
    public Tensor kaiming_uniform(Tensor X, int[] fans) {
        return kaiming_uniform(X, 1.0f, 
               kaiming_fan_mode, fans,
               kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha, int[] fans) {
        return kaiming_uniform(X, alpha, 
               kaiming_fan_mode, fans,
               kaiming_no_linearity,  null);
    }
    
    public Tensor kaiming_uniform(Tensor X, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) {
        return kaiming_uniform(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, int[] fans, float... params) {
        return kaiming_uniform(X, 1.0f, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, params);
    }
    
    public Tensor kaiming_uniform(Tensor X, float alpha,
            int fan_mode, int[] fans,
            int nonlinearity, float... params) {
        float fan = fan(fan_mode, fans);
        float gain = gain(nonlinearity, params);
        float std = (float) (gain / Math.sqrt(fan));
        float bound = alpha * 1.732051f * std;//sqrt(3) = 1.732051
        return this.uniform(X, -bound, bound);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="kaiming_gaussian">
    public Tensor kaiming_gaussian(Tensor X, int[] fans) {
        return kaiming_gaussian(X, 1.0f, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, int[] fans) {
        return kaiming_gaussian(X, alpha, 
                kaiming_fan_mode, fans,
                kaiming_no_linearity, null);
    }
    
    public Tensor kaiming_gaussian(Tensor X, 
            int fan_mode, int[] fans, 
            int nonlinearity, float... params) {
        return kaiming_gaussian(X, 1.0f, 
                fan_mode, fans,
                nonlinearity, params);
    }
    
    public Tensor kaiming_gaussian(Tensor X, float alpha, 
            int fan_mode, int[] fans,
            int nonlinearity, float... params) {
        float fan = fan(fan_mode, fans);
        float gain = gain(nonlinearity, params);
        float std = alpha * (float) (gain / Math.sqrt(fan));
        float sigma = alpha*std;
        return this.gaussian(X, 0, sigma);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    //<editor-fold defaultstate="collapsed" desc="straight linear">
    public Result<Boolean> hasNan(Tensor X) {
        return new Result<Boolean>() {
            @Override
            protected Boolean waitResult() {
               return Float.isNaN(straight_sum(X).get());
            }
        };
    }
    
    public Result<Float> straight_mean(Tensor X) {//(1 / N) * sum(X) = sum(X / N)
        float alpha = 1.0f / X.length; 
        return straight_linear(X, alpha, 0.0f);
    }
    
    public Result<Float> straight_sum(Tensor X) {
        return straight_linear(X, 1.0f, 0.0f);
    }
    public Result<Float> straight_sum(Tensor X, float alpha) {
        return straight_linear(X, alpha, 0.0f); 
    }
    
    @Passed("CudaFloat32EngieBase")//sum(alpha*X + beta)
    public Result<Float> straight_linear(Tensor X, float alpha, float beta) {
        if(check) { require_dataType(X); }
        
        Result<Float> result = core.straight_linear(X.address, 
                alpha, beta, X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight quadratic">
    public Result<Float> straight_sqmean(Tensor X) {//(1/ N) * sum(X^2) = sum(X^2 / N)
        float alpha = 1.0f / X.length;
        return straight_quadratic(X, alpha, 0.0f, 0.0f);
    }
    public Result<Float> straight_sqsum(Tensor X) {
        return straight_quadratic(X, 1.0f, 0.0f, 0.0f); 
    }
    public Result<Float> straight_sqsum(Tensor X, float alpha) {//alpha * sum(X^2)
        return straight_quadratic(X, alpha, 0.0f, 0.0f); 
    }
   
    @Passed("CudaFloat32EngieBase")//sum(alpha*X^2 + beta*X + gamma)
    public Result<Float> straight_quadratic(Tensor X, float alpha, float beta, float gamma)  {
        if(check) { require_dataType(X); }
        
        Result<Float> result = core.straight_quadratic(X.address, 
                alpha, beta, gamma, X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight min & max">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_max(Tensor X)  {
        if(check) { require_dataType(X); }
        
        Result<Float> result = core.straight_max(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
      
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_min(Tensor X)  {
        if(check) { require_dataType(X); }
        
        Result<Float> result = core.straight_min(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight min & max indexed">
    @Passed("CudaFloat32EngieBase")
    public IndexedResult<Float> straight_max_indexed(Tensor X)  {
        if(check) { require_dataType(X); }
        
        IndexedResult<Float> result = core.straight_max_indexed(X.address,
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
      
   @Passed("CudaFloat32EngieBase")
    public IndexedResult<Float> straight_min_indexed(Tensor X)  {
        if(check) { require_dataType(X); }
        
        IndexedResult<Float> result = core.straight_min_indexed(X.address, 
                X.lengthv, X.lastDim());
        if(sync) result.get(); 
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight equal">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_equal(Tensor X1, Tensor X2) {
        Tensor Y = equal(X1, X2);
        float alpha = 1.0f / Y.length;
        Result<Float> result = core.straight_linear(Y.c().address, alpha, 0.0f, 
                Y.lengthv, Y.lastDim());
        if(sync) { result.get(); Y.delete(); return result; }
        return Result.dual(result, ()->{ Y.delete(); });
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="straight var(mean & squareMean)">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_var(Tensor X) {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f,
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f,
                X.lengthv, X.lastDim());
        
        return new Result<Float>() {
            @Override
            protected Float waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float var = sm - m*m;//E(X^2) - E(X)*E(X)
                return var;
            }
        };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_var_mean(Tensor X) {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f,
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f,
                X.lengthv, X.lastDim());
        
        return new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float var = sm - m*m;//E(X^2) - E(X)*E(X)
                return new float[] { var, m };
            }
        };
    }
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_var_mean_sqmean(Tensor X) {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f,
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f,
                X.lengthv, X.lastDim());
        
        return new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float var = sm - m*m;//E(X^2) - E(X)*E(X)
                return new float[] { var, m, sm };
            }
        };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="straight stddev(mean & squareMean)">
    @Passed("CudaFloat32EngieBase")
    public Result<Float> straight_std(Tensor X) {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> squmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f, 
                X.lengthv, X.lastDim());
        
        Result<Float> result =  new Result<Float>() {
            @Override
            protected Float waitResult() {
                float m = mean.get();
                float sm = squmean.get();
                float stddev = (float) Math.sqrt(sm - m*m);//E(X^2) - E(X)*E(X)
                return stddev;
            }
        };
        if(sync) result.get();
        return result;
    }   
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_std_mean(Tensor X) 
    {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f, 
                X.lengthv, X.lastDim());
        
        Result<float[]> result =  new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float stddev = (float) Math.sqrt(sm - m*m);//E(X^2) - E(X)*E(X)
                return new float[]{ stddev, m };
            }
        };
        if(sync) result.get();
        return result;
    }   
    
    @Passed("CudaFloat32EngieBase")
    public Result<float[]> straight_std_mean_sqmean(Tensor X) {
        float alpha = 1.0f / X.length; 
        Result<Float> mean = core.straight_linear(X.address, alpha, 0.0f, 
                X.lengthv, X.lastDim());
        Result<Float> sqmean = core.straight_quadratic(X.address, alpha, 0.0f, 0.0f, 
                X.lengthv, X.lastDim());
        
        Result<float[]> result =  new Result<float[]>() {
            @Override
            protected float[] waitResult() {
                float m = mean.get();
                float sm = sqmean.get();
                float stddev = (float) Math.sqrt(sm - m*m);//E(X^2) - E(X)*E(X)
                return new float[]{ stddev, m, sm };
            }
        };
        if(sync) result.get();
        return result;
    }   
    //</editor-fold>  
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    //<editor-fold defaultstate="collapsed" desc="field linear">
    public Tensor field_mean(Tensor X) { return field_mean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_mean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length) 
        return field_linear(X, row_length, alpha, 0.0f);
    }
    
    public Tensor field_sum(Tensor X) { return field_sum(X, -1); } 
    public Tensor field_sum(Tensor X, int row_length) {//sum(X)
        return field_linear(X, row_length, 1.0f, 0.0f);
    }
    public Tensor field_sum(Tensor X, float alpha, int row_length) {//sum(alpha*X)
        return field_linear(X, row_length, alpha, 0.0f);
    }
   
    public Tensor field_linear(Tensor X, float alpha, float beta) {
        return field_linear(X, -1, alpha, beta);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_linear(Tensor X, int row_length,
            float alpha, float beta)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? //this 2D mem_structure can save memory in padding0-cases
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_linear(Y.address, 
                X.address, alpha, beta,
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field linear2">
    public Tensor field_linear2(Tensor X1, Tensor X2, 
            float alpha, float beta, float gamma) {
        return field_linear2(X1, X2, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_linear2(Tensor X1, Tensor X2, int row_length, 
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            if(X1.ndim() <= 1) throw new IllegalArgumentException("X1.ndim must > 1");
            if(X2.ndim() <= 1) throw new IllegalArgumentException("X2.ndim must > 1");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X1.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_linear_dual(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                X1.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic"> 
    public Tensor field_sqmean(Tensor X) { return field_sqmean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_sqmean(Tensor X, int row_length) { 
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length)
        return field_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor field_sqsum(Tensor X) { return field_sqsum(X, -1); }
    public Tensor field_sqsum(Tensor X, int row_length) {//sum(X^2)
        return field_quadratic(X, row_length, 1.0f, 0.0f, 0.0f);
    }
    public Tensor field_sqsum(Tensor X, int row_length, float alpha) {//sum(alpha*X^2)
        return field_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor field_quadratic(Tensor X, float alpha, float beta, float gamma) {
        return field_quadratic(X, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_quadratic(Tensor X, int row_length,
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? //[height, width]
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_quadratic(Y.address,
                X.address, alpha, beta, gamma, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field quadratic2">
    public Tensor field_mulmean(Tensor X1, Tensor X2) { return field_mulmean(X1, X2, -1); }
    public Tensor field_mulmean(Tensor X1, Tensor X2, int row_length) {
        if(row_length == -1) row_length = X1.lastDim();
        float alpha = (float) ((1.0 * row_length) / X1.length);//(1/ field_length)
        return field_quadratic2(X1, X2, row_length, 
                0, alpha, 0,
                0, 0, 0);
    }
    
    public Tensor field_mulsum(Tensor X1, Tensor X2) { return field_mulSum(X1, X2, -1); }
    public Tensor field_mulSum(Tensor X1, Tensor X2, int row_length) {//sum(X1*X2)
        return field_quadratic2(X1, X2, row_length, 
                0, 1.0f, 0,
                0, 0, 0);
    }
    public Tensor field_mulSum(Tensor X1, Tensor X2, int row_length, float alpha) {//sum(alpha * X1*X2)
        return field_quadratic2(X1, X2, row_length, 
                0, alpha, 0, 
                0, 0, 0);
    }
   
    public Tensor field_quadratic2(Tensor X1, Tensor X2,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return field_quadratic2(X1, X2, -1, k11, k12, k22, k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_quadratic2(Tensor X1, Tensor X2, int row_length, 
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) { 
            require_dataType(X1);
            require_dataType(X2);
            if(X1.ndim() <= 1) throw new IllegalArgumentException("X1.ndim must > 1");
            if(X2.ndim() <= 1) throw new IllegalArgumentException("X2.ndim must > 1");
            equals_valueStructure(X1, "X1", X2, "X2");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X1.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? //[height, width]
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_quadratic_dual(Y.address,
                X1.address, X2.address,
                k11, k12, k22, 
                k1, k2, C, 
                X1.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field linear_quadratic">
    public Tensor[] field_mean_sqmean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) ((1.0 * row_length) / X.length);//(1 / field_length)
        return field_linear_quadratic(X, row_length,
                alpha, 0.0f,//mean = Y1 = field_sum: X / field_length
                alpha, 0.0f, 0.0f);//squareMean = Y2 = field_sum: X^2 / field_length
    }
    
    public Tensor[] field_sum_sqsum(Tensor X) { return field_sum_sqsum(X, -1); }
    public Tensor[] field_sum_sqsum(Tensor X, int row_length) {
        return field_linear_quadratic(X, row_length, 
                1.0f, 0.0f,//Y1 = field_sum: X
                1.0f, 0.0f, 0.0f);//Y2 = field_sum: X^2
    }
    
    public Tensor[] field_linear_quadratic(Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) {
        return field_linear_quadratic(X, -1,
                alpha1, beta1, 
                alpha2, beta2, gamma2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_linear_quadratic(Tensor X, int row_length,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? //[height, width]
                new int[]{ height, width }: 
                new int[]{ width });
        Tensor Y1 = this.empty(dimY);
        Tensor Y2 = this.empty(dimY);
        
        Syncer sc = core.field_linear_quadratic(Y1.c().address, Y2.c().address,
                X.address, 
                alpha1, beta1,
                alpha2, beta2, gamma2,
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[]{ Y1, Y2 };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field max\min">
    public Tensor field_max(Tensor X) { return field_max(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_max(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1?
                new int[]{ height, width } : 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_max(Y.address,
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor field_min(Tensor X) { return field_min(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_min(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1?
                new int[]{ height, width } : 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        
        Syncer sc = core.field_min(Y.address,
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field max\min indexed">
    public Tensor[] field_max_indexed(Tensor X) { return field_max_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_max_indexed(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1?
                new int[]{ height, width } : 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        Tensor Index = this.empty_int32(dimY).c();
        
        Syncer sc = core.field_max_indexed(Y.address, Index.address, 
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[]{ Y, Index };
    }
    
    public Tensor[] field_min_indexed(Tensor X) { return field_min_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_min_indexed(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1?
                new int[]{ height, width } : 
                new int[]{ width });
        Tensor Y = this.empty(dimY).c();
        Tensor Index = this.empty_int32(dimY).c();
        
        Syncer sc = core.field_min_indexed(Y.address, Index.address,
                X.address, 
                X.length, row_length, width);
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="field var(mean & squareMean)">  
    public Tensor field_var(Tensor X) { return field_var(X, - 1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_var(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor var = mean;
        
        Syncer sc = core.field_var(
                var.address, //result0
                mean.c().address, //result1
                sqmean.c().address, //result2
                X.address, 
                X.length, row_length, width);
        
        if(sync) { sc.sync(); delete(sqmean); }
        else var.setSyncer(Syncer.dual(sc, ()->{ delete(sqmean); }));
        return var;
    }
    
    public Tensor[] field_var_mean(Tensor X) { return field_var_mean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_var_mean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor var = sqmean;
        
        Syncer sc = core.field_var(
                var.address, //result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { var, mean };
    }
    
    public Tensor[] field_var_mean_sqmean(Tensor X) { return field_var_mean_sqmean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_var_mean_sqmean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");}
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor var = this.empty(dimY);
        
        Syncer sc = core.field_var(
                var.c().address, //result0
                mean.c().address, //result1
                sqmean.c().address, //result2
                X.address, 
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); sqmean.setSyncer(sc); }
        return new Tensor[] { var, mean, sqmean };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field std(mean & squareMean)">  
    public Tensor field_std(Tensor X) { return field_std(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor field_std(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor std = mean;
        
        Syncer sc = core.field_std(
                std.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address,
                X.length, row_length, width);
        
        if(sync) { sc.sync(); delete(sqmean); }
        else std.setSyncer(Syncer.dual(sc, ()->{ delete(sqmean); })); 
        return std;
    }
    
    public Tensor[] field_std_mean(Tensor X) { return field_std_mean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_std_mean(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        }
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor std = sqmean;
        
        Syncer sc = core.field_std(
                std.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address,
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { std.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { std, mean };
    }
    
    public Tensor[] field_std_mean_sqmean(Tensor X)  { return field_std_mean_sqmean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] field_std_mean_sqmean(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");}
        
        //[Y_firstDim = Y.height = height, Y.lastDim = mem_width = Y.width]
        int width = X.lastDim(), height = row_length / width;
        int[] dimY = (height > 1? 
                new int[]{ height, width }: 
                new int[]{ width });
        
        Tensor mean = this.empty(dimY);
        Tensor sqmean = this.empty(dimY);
        Tensor stddev = this.empty(dimY);
        
        Syncer sc = core.field_std(
                stddev.c().address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address,
                X.length, row_length, width);
        
        if(sync) sc.sync(); else { stddev.setSyncer(sc); mean.setSyncer(sc); sqmean.setSyncer(sc); }
        return new Tensor[] { stddev, mean, sqmean };
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    //<editor-fold defaultstate="collapsed" desc="row_reduce_param_check">
    private void row_reduce_param_check(Tensor X, int row_length) {
        if(X.ndim() <= 1) throw new IllegalArgumentException("X.ndim must > 1");
        if(X.length % row_length != 0) throw new IllegalArgumentException("illgal row_length");
    }
    
    private void row_reduce_param_check(Tensor X, String name, int row_length) {
        if(X.ndim() <= 1) throw new IllegalArgumentException(name + ".ndim must > 1");
        if(X.length % row_length != 0) throw new IllegalArgumentException("illgal row_length");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row linear">
    @Passed("CudaFloat32EngieBase")
    public Tensor row_mean(Tensor X) { return row_mean(X, -1); }
    public Tensor row_mean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_linear(X, row_length, alpha, 0.0f);//sum(X / row_length)
    }
    
    public Tensor row_sum(Tensor X) { return row_sum(X, -1); }
    public Tensor row_sum(Tensor X, int row_length) { return row_linear(X, row_length, 1.0f, 0.0f); }
    public Tensor row_sum(Tensor X, int row_length, float alpha) {
        return row_linear(X, row_length, alpha, 0.0f);
    }
    
    public Tensor row_linear(Tensor X, float alpha, float beta) { return row_linear(X, -1, alpha, beta); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_linear(Tensor X, int row_length, 
            float alpha, float beta)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear(Y.address,
                X.address, alpha, beta,
                field_length, row_length,
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear2">
    public Tensor row_linear2(Tensor X1, Tensor X2,
            float alpha, float beta, float gamma) {
        return row_linear2(X1, X2, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_linear2(Tensor X1, Tensor X2, int row_length,
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2");
            row_reduce_param_check(X1, "X1", row_length);
            row_reduce_param_check(X2, "X2", row_length);
        }
        
        int field_length = X1.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear_dual(Y.address, 
                X1.address, X2.address,
                alpha, beta, gamma,
                field_length, row_length,
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic">
    @Passed("CudaFloat32EngieBase")
    public Tensor row_sqmean(Tensor X) { return row_sqmean(X, -1); }
    public Tensor row_sqmean(Tensor X, int row_length) {
        if(row_length == -1) row_length = X.lastDim();
        //sum(X^2)/row_length = sum(X^2 / row_length)
        float alpha = (float) (1.0 / row_length);
        return row_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor row_sqsum(Tensor X) { return row_sqsum(X, -1); }
    public Tensor row_sqsum(Tensor X, int row_length) { return row_quadratic(X, row_length, 1.0f, 0.0f, 0.0f); }
    public Tensor row_sqsum(Tensor X, int row_length, float alpha) {
        return row_quadratic(X, row_length, alpha, 0.0f, 0.0f);
    }
    
    public Tensor row_quadratic(Tensor X, float alpha, float beta, float gamma) {
        return row_quadratic(X, -1, alpha, beta, gamma);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_quadratic(Tensor X, int row_length,
            float alpha, float beta, float gamma)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_quadratic(Y.address,
                X.address, alpha, beta, gamma,
                field_length, row_length,
                X.lastDim()); 
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row quadratic2">
    public Tensor row_mulMean(Tensor X1, Tensor X2) { return row_mulMean(X1, X2, -1); }
    public Tensor row_mulMean(Tensor X1, Tensor X2, int row_length) {//sum(X) / row_length = 
        if(row_length == -1) row_length = X1.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_quadratic2(X1, X2, row_length,
                0, alpha, 0,
                0, 0, 0);
    }
    
    public Tensor row_mulSum(Tensor X1, Tensor X2) { return row_mulSum(X1, X2, -1); }
    public Tensor row_mulSum(Tensor X1, Tensor X2, int row_length) {//sum(X1*X2)
        return row_quadratic2(X1, X2, row_length, 
                0, 1.0f, 0,
                0, 0, 0);
    }
    public Tensor row_mulSum(Tensor X1, Tensor X2, int row_length, float alpha) {//sum(alpha*X1*X2)
        return row_quadratic2(X1, X2, row_length, 
                0, alpha, 0,
                0, 0, 0);
    }
    
    public Tensor row_quadratic2(Tensor X1, Tensor X2,
            float k11, float k12, float k22, 
            float k1, float k2, float C) {
        return row_quadratic2(X1, X2, -1, k11, k12, k22, k1, k2, C);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_quadratic2(Tensor X1, Tensor X2, int row_length,
            float k11, float k12, float k22, 
            float k1, float k2, float C)
    {
        if(row_length == -1) row_length = X1.lastDim();
        if(check) {
            require_dataType(X1);
            require_dataType(X2);
            equals_valueStructure(X1, "X1", X2, "X2");
            row_reduce_param_check(X1, "X1", row_length);
            row_reduce_param_check(X2, "X2", row_length);
        }
        
        int field_length = X1.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_quadratic_dual(Y.address, 
                X1.address, X2.address, 
                k11, k12, k22,
                k1, k2, C, 
                field_length, row_length, 
                X1.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row linear_quadratic">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_mean_sqmean(Tensor X) { return row_mean_sqmean(X, -1); }
    public Tensor[] row_mean_sqmean(Tensor X, int row_length) {//sum(X / row_length)
        if(row_length == -1) row_length = X.lastDim();
        float alpha = (float) (1.0 / row_length);
        return row_linear_quadratic(X, row_length,
                alpha, 0.0f,//mean = Y1 = row_sum: X / row_length
                alpha, 0.0f, 0.0f);//squareMean = Y2 = row_sum: X^2 / row_length
    }
    
    public Tensor[] row_sum_sqsum(Tensor X) { return row_sum_sqsum(X, -1); }
    public Tensor[] row_sum_sqsum(Tensor X, int row_length) {
        return row_linear_quadratic(X, row_length,
                1.0f, 0.0f,//Y1 = row_sum: X
                1.0f, 0.0f, 0.0f);//Y2 = row_sum: X^2
    }
    
    public Tensor[] row_linear_quadratic(Tensor X,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2) {
        return row_linear_quadratic(X, -1, 
                alpha1, beta1, 
                alpha2, beta2, gamma2);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_linear_quadratic(Tensor X, int row_length,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y1 = this.empty(field_length);//Y = Tensor1D[field_length]
        Tensor Y2 = this.empty(field_length);//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_linear_quadratic(Y1.c().address, Y2.c().address,
                X.address, 
                alpha1, beta1,//Y1 = alpha1*X + beta1
                alpha2, beta2, gamma2,//Y2 = alpha2*X^2 + beta2*X + gamma2
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { Y1.setSyncer(sc); Y2.setSyncer(sc); }
        return new Tensor[]{ Y1, Y2 };
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row max\min">
    public Tensor row_max(Tensor X) { return row_max(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_max(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_max(Y.address, 
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor row_min(Tensor X) { return row_min(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_min(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        
        Syncer sc = core.row_min(Y.address, 
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row max\min indexed">
    public Tensor row_max_index(Tensor X) { return row_max_index(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_max_index(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length).c();
        
        Syncer sc = core.row_max_indexed(Y.address, Index.address,
                X.address,
                field_length, row_length, 
                X.lastDim());
        
        if(sync) { sc.sync(); delete(Y); }
        else Index.setSyncer(Syncer.dual(sc, ()-> {delete(Y); }));
        return Index;
    }
    
    public Tensor[] row_max_indexed(Tensor X) { return row_max_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_max_indexed(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length).c();
        
        Syncer sc = core.row_max_indexed(Y.address, Index.address,
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    
    public Tensor row_min_index(Tensor X) { return row_min_index(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_min_index(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length).c();
        
        Syncer sc = core.row_min_indexed(Y.address, Index.address,
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) { sc.sync(); delete(Y); }
        else Y.setSyncer(Syncer.dual(sc, ()-> { delete(Y); }));
        return Index;
    }
    
    public Tensor[] row_min_indexed(Tensor X) { return row_min_indexed(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_min_indexed(Tensor X, int row_length)
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor Y = this.empty(field_length).c();//Y = Tensor1D[field_length]
        Tensor Index = this.empty_int32(field_length).c();
        
        Syncer sc = core.row_min_indexed(Y.address, Index.address,
                X.address,
                field_length, row_length, 
                X.lastDim());
        if(sync) sc.sync(); else { Y.setSyncer(sc); Index.setSyncer(sc); }
        return new Tensor[] { Y, Index };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="row var(mean & squareMean)">
    public Tensor row_var(Tensor X) { return row_var(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_var(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor var = sqmean;
        
        Syncer sc = core.row_var(
                var.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) { sc.sync(); delete(mean); }
        else var.setSyncer(Syncer.dual(sc, ()-> { delete(mean); }));
        return var;
    }    
    
    public Tensor[] row_var_mean(Tensor X)  { return row_var_mean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_var_mean(Tensor X, int row_length)  
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor var = sqmean;
        
        Syncer sc = core.row_var(
                var.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { var, mean };
    }    
    
    public Tensor[] row_var_mean_sqmean(Tensor X) { return row_var_mean_sqmean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_var_mean_sqmean(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor var = this.empty(field_length);
        
        Syncer sc = core.row_var(var.c().address, 
                mean.c().address, sqmean.c().address, 
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { var.setSyncer(sc); mean.setSyncer(sc); sqmean.setSyncer(sc); }
        return new Tensor[] { var, mean, sqmean };
    }    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row stddev(mean & squareMean)">
    public Tensor row_std(Tensor X) { return row_std(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor row_std(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor std = sqmean;
        
        Syncer sc = core.row_stddev(
                std.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) { sc.sync(); delete(mean); }
        else std.setSyncer(Syncer.dual(sc, ()-> { delete(mean); }));
        return std;
    }    
    
    public Tensor[] row_std_mean(Tensor X)  { return row_std_mean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_std_mean(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) { 
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor std = sqmean;
        
        Syncer sc = core.row_stddev(
                std.address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { std.setSyncer(sc); mean.setSyncer(sc); }
        return new Tensor[] { std, mean };
    }    
    
    public Tensor[] row_std_mean_sqmean(Tensor X) { return row_std_mean_sqmean(X, -1); }
    @Passed("CudaFloat32EngieBase")
    public Tensor[] row_std_mean_sqmean(Tensor X, int row_length) 
    {
        if(row_length == -1) row_length = X.lastDim();
        if(check) {
            require_dataType(X);
            row_reduce_param_check(X, row_length); 
        }
        
        int field_length = X.length / row_length;
        Tensor mean = this.empty(field_length);
        Tensor sqmean = this.empty(field_length);
        Tensor std = this.empty(field_length);
        
        Syncer sc = core.row_stddev(
                std.c().address,//result0
                mean.c().address,//result1
                sqmean.c().address,//result2
                X.address, 
                field_length, row_length,
                X.lastDim());
        
        if(sync) sc.sync(); else { std.setSyncer(sc); mean.setSyncer(sc); sqmean.setSyncer(sc); }
        return new Tensor[] { std, mean, sqmean };
    }    
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Neural: Extended">
    public void garbage_collect(Unit unit) { unit.gc(); }
    public void delete(Unit unit) { unit.delete(); }
    //</editor-fold>
}

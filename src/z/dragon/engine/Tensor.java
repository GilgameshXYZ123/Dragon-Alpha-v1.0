/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import z.dragon.nn.unit.Trace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import z.util.math.vector.Vector;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Syncer.DualSyncer;
import z.dragon.engine.Syncer.FollowSyncer;
import z.dragon.engine.Syncer.RemoteSync;
import z.dragon.nn.unit.Unit;

/**
 * @author Gilgamesh
 * {4, 3, 2, 1}, firstDim -> lastDim
 */
@SuppressWarnings("unchecked")
public class Tensor implements StateValue
{
    //<editor-fold defaultstate="collapsed" desc="static class: TensorList">
    public static class TensorList extends ArrayList<Tensor>  
    {
        private static final long serialVersionUID = 615120712446L;
        
        public TensorList(int init_capacity) { super(init_capacity);}
        public TensorList() { super(); }
        
        @Override
        public boolean add(Tensor ts) {
            return ((ts == null || ts.isNull()) ? false : super.add(ts));
        }

        public boolean addAll(Tensor[] arr) {
            boolean flag = false;
            for(Tensor ts : arr) {
                if(ts == null || ts.isNull()) continue;
                flag |= super.add(ts);
            }
            return flag;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: TenorSet">
    public static class TensorSet extends HashSet<Tensor> 
    {
        private static final long serialVersionUID = 1L;
        
        public TensorSet() { super(); }
        public TensorSet(int init_capacity) { super(init_capacity); }

        @Override
        public boolean add(Tensor ts) {
            return ((ts == null || ts.isNull()) ? false : super.add(ts));
        }
        
        public boolean add(Tensor...ts) {
            if(ts == null || ts.length == 0) return false;
            boolean result = true;
            for(Tensor t : ts) {
                if(t == null || t.isNull()) result &= false;
                else result &= super.add(t);
            }
            return result;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: TensorMap">
    public static class TensorMap<K> extends HashMap<K, Tensor>
    {
        private static final long serialVersionUID = 1L;
        
        public TensorMap() { super(); }
        public TensorMap(int initialCapacity) { super(initialCapacity); }

        @Override
        public Tensor put(K key, Tensor value) {
            return ((value == null || value.isNull()) ? null :  super.put(key, value)); 
        }
    }
    //</editor-fold>
    
    public static final int MAX_NDIM = 4;
    public static final int MAX_LENGTH_4X = Integer.MAX_VALUE;

    //<editor-fold defaultstate="collapsed" desc="member params">
    protected final Engine eg;
    protected String dataType;
    protected long address;
    protected int[] dim;//dimensions, from the first to last, the highest dimension is the first
    
    protected long mem_size;//mem_size alloced by Engine
    protected int length_4x;//length_algined <= mem_length, as least used mem_length
    protected int lengthv;//mem_stride * mem_height, mem_stride = (mem_width + 3) >> 2 << 2
    protected int length;//mem_width * mem_height
    //</editor-fold>
    
    Tensor(Engine engine, String dataType, int[] dim) { this(true, engine, dataType, dim); }
    Tensor(boolean check, Engine engine, String dataType, int[] dim) {
        this.eg = engine;
        this.dataType = dataType;
        this.setDim(check, dim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Baisc-Functions: Engine & Address"> 
    public Engine engine() {return eg;}
    
    public String dataType() { return dataType;}
    public boolean is_dtype() { return dataType.equals(eg.dataType()); }
    public boolean is_int32() { return dataType.equals(eg.dataType_int32()); }
    public boolean is_int8() { return dataType.equals(eg.dataType_int8()); }
    
    public long address() {return address;}
    
    protected String msg = null;
    public String message() { return msg; }
    public Tensor message(String message) { this.msg = message; return this;}
    
    public static boolean isNull(Tensor ts) { return ts == null || ts.address == 0L; }
    public boolean isNull() { return address == 0L; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { dataType = ").append(eg.dataType());
        sb.append(", dim = ").append(Arrays.toString(dim));
        sb.append(", [length, mem_size, address] = [")
                .append(length).append(", ")
                .append(mem_size).append(",  ")
                .append(address).append(']');
        sb.append(", need_gards = ").append(need_grad);
        if(msg != null) sb.append(", messge = ").append(msg);
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }
      
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        eg.delete(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Dimension - Memory Structure"> 
    public int ndim() { return dim.length; }
    public int[] dim() { return dim; }
    
    public int dim(int index) {//index = -1: the last dim, dim.length - 1
        if(index < 0) index = dim.length + index;//index = -2: the second last dim, dim.length - 2
        return dim[index];
    }
    
    public int firstDim() {return dim[0];}
    public int lastDim() {return dim[dim.length - 1];}
    
    public int length() { return length; }
    public int lengthv() { return lengthv; }
    public int length_4x() { return length_4x; }
    public long memory_size() { return mem_size; }
    
    protected void bindMemory(long[] block) {
        this.mem_size = block[0];
        this.address = block[1];
    }
      
    protected void copy_memoryMetaData_and_deleteSrc(Tensor X) {
        synchronized(X) {
            this.address = X.address;
            this.dim = X.dim;
            this.mem_size = X.mem_size;
            this.length_4x = X.length_4x;
            this.lengthv = X.lengthv;
            this.length = X.length;
            X.address = 0;//As X.finalize may be called
        }
    }   
    
    protected void copy_memoryMetaData(Tensor X) {
         synchronized(X) {
            this.address = X.address;
            this.dim = X.dim;
            this.mem_size = X.mem_size;
            this.length_4x = X.length_4x;
            this.lengthv = X.lengthv;
            this.length = X.length;
        }
    }
    
    protected final void setDim(boolean check, int... dim) {
        if(check) {
            if(dim == null) throw new NullPointerException();
            if(dim.length == 0) throw new IllegalArgumentException("Tensor.ndim == 0");
            if(dim.length > MAX_NDIM) throw new IllegalArgumentException("Tensor.dim > MAX_NDIM = " + MAX_NDIM);
            for(int d : dim) if(d <= 0) throw new IllegalArgumentException("dim_size must be positive");
        }
        
        int firstDim = dim[0], firstDim_4x = ((firstDim + 3) >> 2) << 2;
        int Length_4x, Lengthv, Length;
        if(dim.length == 1) {
            Length_4x = firstDim_4x;
            Lengthv   = firstDim_4x;
            Length    = firstDim;
        }
        else {
            int lastDim = dim[dim.length-1], lastDim_4x = ((lastDim + 3) >> 2) << 2;
            int midlen = 1;
            for(int i = 1; i< dim.length - 1; i++) midlen *= dim[i];
            Length_4x = firstDim_4x * midlen * lastDim_4x;
            Lengthv   = firstDim    * midlen * lastDim_4x;
            Length    = firstDim    * midlen * lastDim;
        }
        
        if(Length_4x < 0 || Length_4x > MAX_LENGTH_4X) 
            throw new IllegalArgumentException("the maximum length of Tensor is exceed");
        
        this.length_4x = Length_4x;
        this.lengthv = Lengthv;
        this.length = Length;
        this.dim = Vector.arrayCopy(dim);
    }
    
    public boolean dimEquals(int... dim) { return Arrays.equals(this.dim, dim); }
    public boolean dimEquals(Tensor ts) { return Arrays.equals(dim, ts.dim); }
    
    public boolean isMemAligned() {
        return (dim[0] & 3) != 0 || (dim[dim.length - 1] & 3) != 0;
    }
    
    public boolean memSturcEquals(Tensor ts)  {return memStrucEquals(ts.length, ts.dim);}
    boolean memStrucEquals(int length2, int... dim2)
    {
        if(length2 != this.length) return false;//ts.length != this.length
        int ndim1 = dim.length, ndim2 = dim2.length;
        int firstDim1 = dim[0], lastDim1 = dim[dim.length - 1]; 
        int firstDim2 = dim2[0], lastDim2 = dim2[dim2.length - 1];
        
        //if: this and another is not memalgined, and this.length = another.length,
        if((firstDim1 & 3) == 0 && (lastDim1 & 3) == 0 &&
           (firstDim2 & 3) == 0 && (lastDim2 & 3) == 0) return true;
        
        if(ndim1>1 && ndim2>1) {//ND ND
            //the mem alignment only effects the first and the last dim
            return firstDim2 == firstDim1 && lastDim2 == lastDim1;
        }
        if(ndim1>1) {//ND 1D
            return (firstDim1 & 3) == 0 && (lastDim1 & 3) == 0 &&
                   (length2 & 3) == 0;
        }
        if(ndim2>1) {//1D ND
            return (firstDim2 & 3) == 0 && (lastDim2 & 3) == 0 &&
                   (this.length & 3) == 0;
        }
        return true;//1D 1D
    }
    
    boolean valueStrucEquals(int length2, int... dim2)  {
        if(length2 != this.length) return false;//ts.length != this.length
        return dim[dim.length - 1] == dim2[dim2.length - 1];
        //we have: lengthv2 % mem_width2 == 0
        //As: ts.length != this.length
        //    ts.mem_width = this.mem_width
        //So: ts.lengthv = ts.length/ts.mem_width * ts.mem_stride
        //    this.lengthv = this.length/this.mem_width * this.mem_stride
        //we have ts.lengthv = this.lengthv
    }
    
    public boolean valueStrucEquals(Tensor ts) {
        if(ts.lengthv != this.lengthv) return false;
        
        if(ts.lastDim() == this.lastDim()) return true;
        if(!ts.isMemAligned() && !this.isMemAligned()) return true;
        return false;
    }
    
    public void requireValueStrucEquals(Tensor ts, String name1, String name2) {
        if(!valueStrucEquals(ts)) throw new IllegalArgumentException(
                name1 + ".valueStructure is different from that of" + name2);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Syncer: extended">
    protected volatile Syncer syncer;//to get the result of the last computation
    
    public Syncer syncer() { return syncer; }
    public synchronized void setSyncer(Syncer sc) { this.syncer = sc; }
    
    public synchronized Tensor c() {
        if(syncer != null) {
            Syncer sc = syncer;
            syncer = null;
            sc.sync();
        }
        return this;
    }
    
    public Tensor remote_sync() { RemoteSync.sync(this); return this; }
    
    public Tensor dual(Syncer after) {
        synchronized(this) {
            Syncer before = this.syncer;
            this.syncer = new DualSyncer(before, after);
        }
        return this;
    }
    
    public Tensor follow(Tensor ts) {
        synchronized(this) {
            this.syncer = new FollowSyncer(ts);
        }
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Engine: extended">
    //<editor-fold defaultstate="collapsed" desc="value: functions">
    @Override public float[] value() { return eg.valueOf(this); }
    public int[] value_int32() { return eg.valueOf_int32(this); }
    public byte[] value_int8() { return eg.valueOf_int8(this); }
    
    public float[][] value2D()  { return value2D(this.lastDim());}
    public float[][] value2D(int lastDim) {
        if(this.ndim() < 2) throw new IllegalArgumentException("ndim must >= 2");
        float[] value = eg.valueOf(this);
        int width = lastDim; if(width == -1) width =  this.lastDim();
        int height = this.length / width;
        return Vector.to2D(value, height, width);
    }
    
    public Tensor vprintln() { Vector.println(eg.valueOf(this)); return this; }
    public Tensor vprintln(int start, int end) {
        float[] value = eg.valueOf(this);
        Vector.println(value, start, end);
        return this;
    }
    
    public <T> T data() {
        int ndim = ndim();
        if(is_dtype()) {
            float[] value = eg.valueOf(this);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        if(is_int32()) {
            int[] value = eg.valueOf_int32(this);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        if(is_int8()) {
            byte[] value =eg.valueOf_int8(this);
            if(ndim == 1) return (T) value;
            if(ndim == 2) return (T) Vector.to2D(value, dim[0], dim[1]);
            if(ndim == 3) return (T) Vector.to3D(value, dim[0], dim[1], dim[2]);
            else return (T) Vector.to4D(value, dim[0], dim[1], dim[2], dim[3]);
        }
        throw new RuntimeException("Unknown dataType");
    }
    //</editor-fold>
    
    public static void sync(Tensor... arr) { for(Tensor t : arr) if(t != null) t.c(); }
    public static void delete(Tensor... arr) { for(Tensor t: arr) if(t != null) t.delete(); }
    
    public static Tensor[] zero_like(Tensor... arr) {
        Tensor[] zeros = new Tensor[arr.length];
        for(int i=0; i<arr.length; i++) zeros[i] = arr[i].zeros_like();
        return zeros;
    }
    public static Tensor[] zero_like(Parameter... params) {
        Tensor[] zeros = new Tensor[params.length];
        for(int i=0; i<params.length; i++) zeros[i] = params[i].tensor.zeros_like();
        return zeros;
    }
   
    public void delete() { eg.delete(this); }
    
    public Tensor zero() { return eg.zero(this); }
    public Tensor constant(float value) { return eg.constant(this, value); }
    public Tensor zero_nan(){ return eg.zero_nan(true, this); }
    
    public Tensor empty_like() { return eg.empty_like(this); }
    public Tensor zeros_like() { return eg.zeros_like(this); }
    public Tensor ones_like() { return eg.ones_like(this); }
    public Tensor constant_like(float value) { return eg.constant_like(value, this); }
    
    public Tensor set(float[] value) { eg.set(this, value); return this; }
    public Tensor set(byte[]  value) { eg.set(this, value); return this; }
    public Tensor set(String line) { eg.set(this, line); return this; }
    public Tensor set(ArrayList<String> lines) { eg.set(this, lines); return this; }
    public Tensor set(StateValue value, boolean partial, String msg) {
        eg.set(this, value, partial, msg); 
        return this;
    }
    
    public Result<Boolean> hasNan() { return eg.hasNan(this); }
    public Result<Float> max() { return eg.straight_max(this); }
    public Result<Float> min() { return eg.straight_min(this); }
    public Result<Float> sum() { return eg.straight_sum(this); }
    public Result<Float> sqsum() { return eg.straight_sqsum(this); }
    public Result<Float> mean() { return eg.straight_mean(this); }
    public Result<Float> var() { return eg.straight_var(this); }
    public Result<Float> std() { return eg.straight_std(this); }
    public Result<Float> equal(Tensor X) { return eg.straight_equal(X, this); }
    
    public Result<float[]> std_mean() { return eg.straight_std_mean(this); }
    public Result<float[]> std_mean_sqmean() { return eg.straight_std_mean_sqmean(this); }
    public Result<float[]> var_mean() { return eg.straight_var_mean(this); }
    public Result<float[]> var_mean_sqmean() { return eg.straight_var_mean_sqmean(this); }
    
    public Tensor sadd(boolean inplace, float C) { return eg.sadd(inplace, this, C); }
    public Tensor ssub(boolean inplace, float C) { return eg.ssub(inplace, this, C); }
    public Tensor smul(boolean inplace, float C) { return eg.smul(inplace, this, C); }
    public Tensor sdiv(boolean inplace, float C) { return eg.sdiv(inplace, this, C); }
    public Tensor linear(boolean inplace, float alpha, float beta) { 
        return eg.linear(inplace, alpha, this, beta); 
    }
   
    public Tensor square(boolean inplace) { return eg.square(inplace, this); }
    public Tensor quadratic(boolean inplace, float alpha, float beta, float gamma) { 
        return eg.quadratic(inplace, this, alpha, beta, gamma);
    }
   
    public Tensor add(boolean inplace, Tensor X) { return eg.add(inplace, this, X); }
    public Tensor sub(boolean inplace, Tensor X) { return eg.sub(inplace, this, X); }
    public Tensor linear2(boolean inplace, Tensor X, float alpha, float beta, float gamma) { 
        return eg.linear2(inplace, this, X, alpha, beta, gamma); 
    }
  
    public Tensor mul(boolean inplace, Tensor X) { return eg.mul(inplace, this, X); }
    public Tensor mul(boolean inplace, float alpha, Tensor X) { return eg.mul(inplace, alpha, this, X); }
    public Tensor squareAdd(boolean inplace, Tensor X) { return eg.squareAdd(inplace, this, X); } 
    public Tensor squareAdd(boolean inplace, Tensor X, float alpha, float beta) { return eg.squareAdd(inplace, this, X, alpha, beta); }
    public Tensor quadraitic2(boolean inplace, Tensor X,
            float k11, float k12, float k22,
            float k1, float k2, float C) {
        return eg.quadratic2(inplace, this, X,
                k11, k12, k22,
                k1, k2, C);
    }
    
    public Tensor rpl(boolean inplace) { return eg.rpl(inplace, this); }
    public Tensor rpl(boolean inplace, float alpha) { return eg.rpl(inplace, alpha, this); }
    public Tensor div(boolean inplace, Tensor X) { return eg.div(inplace, this, X); }
    public Tensor div(boolean inplace, float alpha, Tensor X) { return eg.div(inplace, alpha, this, X); }
    public Tensor div(boolean inplace, Tensor X, 
            float alpha1, float beta1,
            float alpha2, float beta2, 
            float gamma) {
        return eg.div(inplace, 
                alpha1, this, beta1, 
                alpha2, X, beta2, 
                gamma);
    }
    
    public Tensor min(boolean inplace, float vmin) { return eg.min(inplace, this, vmin); }
    public Tensor max(boolean inplace, float vmax) { return eg.max(inplace, this, vmax); }
    public Tensor min2(boolean inplace, Tensor X) { return eg.min2(inplace, this, X); }
    public Tensor max2(boolean inplace, Tensor X) { return eg.max2(inplace, this, X); }
    public Tensor clip(boolean inplace, float vmin, float vmax) { return eg.clip(inplace, this, vmin, vmax); }
    
    public Tensor sin(boolean inplace) { return eg.sin(inplace, this); }
    public Tensor sin(boolean inplace, float alpha, float beta) { return eg.sin(inplace, alpha, this, beta); }
    public Tensor cos(boolean inplace) { return eg.cos(inplace, this); }
    public Tensor cos(boolean inplace, float alpha, float beta) { return eg.cos(inplace, alpha, this, beta); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="State: extended">
    @Override public Class<?> type() { return float[].class; }

    private static final int element_each_line = 4096;
    
    @Override
    public ArrayList<String> toStringLines() 
    {
        float[] value = eg.valueOf(this);
        ArrayList<String> lines = new ArrayList<>(4);
        int sb_size = Math.min(element_each_line, length) << 4;//length = 4096 * 16, 16K mem
        StringBuilder sb = new StringBuilder(sb_size);
        
        for(int i=0; i<value.length; i++) {
            sb.append(value[i]).append(',');
            if((i + 1) % element_each_line == 0) {
                String line = sb.deleteCharAt(sb.length() - 1).toString();
                lines.add(line);
                sb.setLength(0);//reset the StringBuilder
            }
        }
        
        if(value.length % element_each_line != 0) {
            String line = sb.deleteCharAt(sb.length() - 1).toString();
            lines.add(line);
        }
        return lines;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Neural: extended">
    //<editor-fold defaultstate="collapsed" desc="mod_count: if the data of Tensor is changed">
    public static class Mod_Count { private int value = 0; }
    protected Mod_Count mod_counter = new Mod_Count();
    
    public int mod_count() { synchronized(mod_counter) { return mod_counter.value; } }
    public Tensor modify(boolean flag) { 
        if(flag) synchronized(mod_counter) { mod_counter.value++; }
        return this; 
    }
    
    public boolean isHold(int mod_count) {
        synchronized(mod_counter) { 
            return mod_count == mod_counter.value; 
        } 
    }
    
    public Tensor hold(int mod_count, String msg) {
        synchronized(mod_counter) {
            if(mod_count != mod_counter.value) throw new RuntimeException(
                    msg + ": the tensor.data is changed, it will cause error in compute graph");
        }
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="gradient: find the gradient of a tensor">
    protected boolean is_grad = false;
    public boolean is_grad() { return is_grad; }
    
    protected boolean need_grad = false;
    public boolean need_grad() { return need_grad; }
    public Tensor need_grad(boolean flag) { this.need_grad = flag; return this; }
    
    protected Tensor grad = null;//for variables
    public Tensor grad() { return grad; }
    public synchronized Tensor grad(Tensor grad) { this.grad = grad; return this; }
    
    public synchronized void clear_grad() {
        if(grad == null) return;
        grad.delete(); grad = null;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="carrier: for OneOff Unit">
    //pay attention: the gradients is only used form OneOffScale
    //As sime intermediate variables causes memory leaks and need to be brought out by carriers
    protected boolean need_carry = false;//only the output of OneOffScale needCarry
    public boolean need_carry() { return need_carry; }
    public Tensor need_carry(boolean flag) { this.need_carry = flag; return this; }
    
    protected TensorSet carrier;
    public synchronized TensorSet carrier() {return carrier;}
    
    public void carry(Tensor ts) {
        if(ts == null || ts == this || !ts.need_carry) return;
        if(carrier == null) carrier = new TensorSet(4);
        
        synchronized(this) {
            carrier.add(ts);
            if(ts.carrier != null && !ts.carrier.isEmpty()) {//hitch = union(ts.hitch, ts)
                carrier.addAll(ts.carrier);
                ts.carrier = null;//clear ts.carrier
            }
        }
    }
    
   public synchronized void carry(Tensor... arr) {
       if(arr == null || arr.length == 0) return;
       for(Tensor ts : arr) carry(ts);
   }
    
    public synchronized void clear_carrier() {
        if(carrier == null) return;
        carrier.forEach((fare) -> { eg.delete(fare); });
        carrier.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trace & forward_gc: Used to connect two Units">
    protected Trace trace;
    
    public Trace trace() { return trace; }
    
    public void setTrace(Unit last, int last_out_index, boolean need_grads) { 
        trace = new Trace(last, last_out_index, need_grads);
    }
    //</editor-fold>
    //</editor-fold>
}

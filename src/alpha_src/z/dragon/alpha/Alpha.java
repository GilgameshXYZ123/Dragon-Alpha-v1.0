/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.alpha;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.function.Function;
import z.dragon.common.DragonCV;
import z.dragon.common.DragonFile;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateReader;
import z.dragon.common.state.State.StateWriter;
import z.dragon.data.Buffer;
import z.dragon.data.DataSet;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.Transform;
import z.dragon.data.container.AutoLoadContainer;
import z.dragon.data.container.AutoLoadContainer.Loader;
import z.dragon.data.container.ListContainer;
import z.dragon.engine.Engine;
import z.dragon.engine.EngineCore;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.PinnedMempool;
import z.dragon.nn.loss.dim2.CrossEntropy;
import z.dragon.nn.loss.dim1.L2;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.loss.WeightedSummary;
import z.dragon.nn.loss.dim1.L1;
import z.dragon.nn.loss.dim1.BinaryCrossEntropy;
import z.dragon.nn.loss.dim1.SigmoidBinaryCrossEntropy;
import z.dragon.nn.loss.dim1.SmoothL1;
import z.dragon.nn.loss.dim2.SoftmaxCrossEntropy;
import z.dragon.nn.optim.Adam;
import z.dragon.nn.optim.AdamW;
import z.dragon.nn.optim.Adamax;
import z.dragon.nn.optim.Adamod;
import z.dragon.nn.optim.Momentum;
import z.dragon.nn.optim.RMSprop;
import z.dragon.nn.optim.SGD;
import z.dragon.nn.optim.lr_schedular.CosAnnealingLr;
import z.dragon.nn.optim.lr_schedular.ExponentialLr;
import z.dragon.nn.optim.lr_schedular.LambdaLr;
import z.dragon.nn.unit.complex.Branch;
import z.dragon.nn.unit.complex.Sequence;
import z.dragon.nn.unit.furcation.Chunk;
import z.dragon.nn.unit.furcation.Furcation;
import z.dragon.nn.unit.furcation.Split;
import z.dragon.nn.unit.reducer.Concat;
import z.dragon.nn.unit.reducer.LinearMean;
import z.dragon.nn.unit.reducer.Reducer;
import z.dragon.nn.unit.reducer.LinearSummary;
import z.dragon.nn.unit.dual.DualUnit;
import z.dragon.nn.unit.dual.math2.BatchMatMul;
import z.dragon.nn.unit.dual.math2.BatchMatMulT1;
import z.dragon.nn.unit.dual.math2.BatchMatMulT2;
import z.dragon.nn.unit.dual.math2.Div;
import z.dragon.nn.unit.dual.math1.MatMul;
import z.dragon.nn.unit.dual.math1.MatMulT1;
import z.dragon.nn.unit.dual.math1.MatMulT2;
import z.dragon.nn.unit.dual.math2.Quadratic2;
import z.dragon.nn.unit.simple.pool2d.AdaptiveAvgPool2D;
import z.dragon.nn.unit.simple.pool2d.AdaptiveMaxPool2D;
import z.dragon.nn.unit.simple.pool2d.AdaptiveNaiveMaxPool2D;
import z.dragon.nn.unit.simple.pool2d.AvgPool2D;
import z.dragon.nn.unit.simple.pool2d.AvgUnpool2D;
import z.dragon.nn.unit.simple.Conv3D;
import z.dragon.nn.unit.simple.Deconv3D;
import z.dragon.nn.unit.simple.FullConnect;
import z.dragon.nn.unit.simple.pool2d.MaxPool2D;
import z.dragon.nn.unit.simple.pool2d.NaiveMaxPool2D;
import z.dragon.nn.unit.simple.SimpleUnit;
import z.dragon.nn.unit.simple.math1.Abs;
import z.dragon.nn.unit.simple.affine.Affine;
import z.dragon.nn.unit.simple.math2.Arcsin;
import z.dragon.nn.unit.simple.math2.Arctan;
import z.dragon.nn.unit.simple.affine.GlobalSqBatchNorm;
import z.dragon.nn.unit.simple.math1.Cos;
import z.dragon.nn.unit.simple.math2.Cot;
import z.dragon.nn.unit.simple.math2.Dropout;
import z.dragon.nn.unit.simple.math2.Elu;
import z.dragon.nn.unit.simple.math2.Exp;
import z.dragon.nn.unit.simple.tensor.Flatten;
import z.dragon.nn.unit.simple.math2.HalfSin;
import z.dragon.nn.unit.simple.affine.LayerNorm;
import z.dragon.nn.unit.simple.math2.LeakyRelu;
import z.dragon.nn.unit.simple.math2.Linear;
import z.dragon.nn.unit.simple.math2.Log;
import z.dragon.nn.unit.simple.math1.Quadratic;
import z.dragon.nn.unit.simple.math2.Relu;
import z.dragon.nn.unit.simple.tensor.Rot180;
import z.dragon.nn.unit.simple.math2.Rpl;
import z.dragon.nn.unit.simple.math2.Sigmoid;
import z.dragon.nn.unit.simple.math1.Sin;
import z.dragon.nn.unit.simple.math2.Softplus;
import z.dragon.nn.unit.simple.math2.Sqrt;
import z.dragon.nn.unit.simple.math2.Tan;
import z.dragon.nn.unit.simple.math2.Tanh;
import z.dragon.nn.unit.simple.tensor.Transpose;
import z.dragon.nn.unit.simple.tensor.Reshape;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.dual.math2.Linear2;
import z.dragon.nn.unit.reducer.QuadraticMean;
import z.dragon.nn.unit.reducer.QuadraticSummary;
import z.dragon.nn.unit.simple.affine.SqBatchNorm;
import z.dragon.nn.unit.simple.math2.LogSoftmax;
import z.dragon.nn.unit.simple.math2.Softmax;
import z.dragon.nn.unit.simple.tensor.View;
import z.util.lang.annotation.Passed;
import z.dragon.data.container.AutoLoadContainer.Triger;
import z.dragon.data.container.DataContainer;
import z.dragon.engine.EngineCore_ori;
import z.dragon.engine.memp.Memp1;
import z.dragon.engine.memp.Memp2;
import z.dragon.engine.memp.Memp3;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.simple.math2.BernouliMul;
import z.util.math.vector.Vector;
import z.dragon.common.state.State.Stateful;
import z.dragon.common.state.State.StatefulTransformer;
import z.dragon.common.state.ZipState.ZipStateReader;
import z.dragon.common.state.ZipState.ZipStateWriter;
import z.dragon.engine.Parameter;
import z.dragon.nn.optim.SGDMN;
import z.dragon.nn.unit.simple.affine.BatchNorm;
import z.dragon.nn.unit.simple.affine.GlobalBatchNorm;

/**
 *
 * @author Gilgamesh
 */
public final class Alpha 
{
    public final long MEM_1GB = Engines.MEM_1GB;
    public final long MEM_1MB = Engines.MEM_1MB;
    
    private Alpha() {}
    
    public static final Alpha alpha = new Alpha();
    public final Engines engine = Engines.engine;
    public final UnitBuilder nn = UnitBuilder.nn;
    public final Loss loss = Loss.loss;
    public final Optim optim = Optim.optim;
    public final Datas data = Datas.data;
    public final Stats stat = Stats.stat;
    
    public final DragonCV cv = DragonCV.instance();
    public final DragonFile fl = DragonFile.instance();
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("Dragon - Alpha: ");
        sb.append("\nAuthor: 张智溢 Gilgamesh.CN");
        sb.append("\nVersion: 2022/4/24");
        sb.append("\nAnywhere, try your best!");
        return sb.toString();
    }
    
    //<editor-fold defaultstate="collapsed" desc="alpha: string && print(args)">
    //<editor-fold defaultstate="collapsed" desc="toString: float">
    public String format(float f) { 
        return (f > 1e-4f) || (-f > 1e-4f)? 
                String.format("%6f", f):
                String.format("%6e", f); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, float[])">
    public void append(StringBuilder sb, float[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]); 
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, prefix + " ", X[i]); } 
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string(float[])">
    public String str(float[] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toString: int">
    public String format(int d) { return Integer.toString(d); }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, int[])">
    public void append(StringBuilder sb, int[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]); 
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, prefix + " ", X[i]); } 
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string(int[])">
    public String str(int[] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toString: byte">
    public String format(byte d) { return String.format("%3d", d); }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, byte[])">
    public void append(StringBuilder sb, byte[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]); 
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, prefix + " ", X[i]); } 
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sting(byte[])">
    public String str(byte[] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(byte[][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
      public String str(byte[][][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(byte[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    
    public String string(Object arg) {
        //array 1D--------------------------------------------------------------
        if(arg instanceof   float[]) return str((float[]) arg);
        if(arg instanceof    byte[]) return str((byte[])  arg);
        if(arg instanceof     int[]) return str((int[])   arg);
        
        //array 2D--------------------------------------------------------------
        if(arg instanceof   float[][]) return str((float[][]) arg);
        if(arg instanceof    byte[][]) return str((byte[][])  arg);
        if(arg instanceof     int[][]) return str((int[][])   arg);
        
        //array 3D--------------------------------------------------------------
        if(arg instanceof   float[][][]) return str((float[]) arg);
        if(arg instanceof    byte[][][]) return str((byte[])  arg);
        if(arg instanceof     int[][][]) return str((int[])   arg);
        
        //array 3D--------------------------------------------------------------
        if(arg instanceof   float[][][][]) return str((float[]) arg);
        if(arg instanceof    byte[][][][]) return str((byte[])  arg);
        if(arg instanceof     int[][][][]) return str((int[])   arg);
        
        return Objects.toString(arg);
    }
    
    private boolean __print_array1D(Object arg) {
        if(arg instanceof   float[]) { out.print(str((float[]) arg)); return true; }
        if(arg instanceof    byte[]) { out.print(str((byte[])  arg)); return true; }
        if(arg instanceof     int[]) { out.print(str((int[])   arg)); return true; }
        return false;
    }
    
    private boolean __print_array2D(Object arg) {
        if(arg instanceof   float[][]) { out.print(str((float[][])arg)); return true; }
        if(arg instanceof    byte[][]) { out.print(str( (byte[][])arg)); return true; }
        if(arg instanceof     int[][]) { out.print(str(  (int[][])arg)); return true; }
        return false;
    }
    
    private boolean __print_array3D(Object arg) {
        if(arg instanceof float[][][]) { out.print(str((float[][][])arg)); return true;}
        if(arg instanceof  byte[][][]) { out.print(str(( byte[][][])arg)); return true;}
        if(arg instanceof   int[][][]) { out.print(str(  (int[][][])arg)); return true;}
        return false;
    }
    
    private boolean __print_array4D(Object arg) {
        if(arg instanceof float[][][][]) { out.print(str((float[][][][])arg)); return true;}
        if(arg instanceof  byte[][][][]) { out.print(str( (byte[][][][])arg)); return true;}
        if(arg instanceof   int[][][][]) { out.print(str(  (int[][][][])arg)); return true;}
        return false;
    }
    
    public static PrintStream out = System.out;
    public Alpha print(Object... args) {
        if(args == null) { out.println("null"); return this; }
        if(__print_array2D(args)) { out.println(); return this; }
        if(__print_array3D(args)) { out.println(); return this; }
        if(__print_array4D(args)) { out.println(); return this; }
        
        for(Object arg : args) {
            if(__print_array1D(arg)) continue;
            if(__print_array2D(arg)) continue;
            if(__print_array3D(arg)) continue;
            if(__print_array4D(arg)) continue;
            out.print(Objects.toString(arg));
        }
        
        out.println(); 
        return this;
    }
    //</editor-fold>s
    
    //<editor-fold defaultstate="collapsed" desc="parallel_class: Line">
    private static final ThreadFactory daemonThreadFactory = new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    };
    private static final ExecutorService exec = Executors.newFixedThreadPool(4, daemonThreadFactory); 
    
    public static class Line<T> 
    {
        private final Future<T> ft;
        
        public Line(Future<T> ft) { this.ft = ft; }
        
        public T c() {
            try { return ft.get(); }
            catch(InterruptedException | ExecutionException e) {
                throw new RuntimeException(e); 
            }
        }
    }
    
    public <T> Line<T> line(Callable<T> call) {  return new Line<>(exec.submit(call));  }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ALPHA_HOME">
    static String alpha_home;
    
    public String home() { return alpha_home; }
    
    public synchronized void home(String ALPHA_HOME)
    {
        if(ALPHA_HOME == null || ALPHA_HOME.isEmpty()) 
            throw new NullPointerException("ALPHA_HOME is empty");
        
        alpha_home = ALPHA_HOME;
        
        if(alpha_home.endsWith("\\")) //alpha_home是文件夹位置的标识，后面不能加 \
            alpha_home = alpha_home.substring(0, alpha_home.length() - 1);
        
        //load native lib: CudaFloat32EngineBase--------------------------------
        if (!CudaFloat32EngineBase.__NATIVE_LOAD__()) {
                CudaFloat32EngineBase.load_native_lib(alpha_home);
                CudaFloat32EngineBase.__SET_NATIVE_LOAD__(true);
                System.out.println("[alpha]: CudaFloat32-nativeLib is loaded");
            }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Engines">
    public static class Engines 
    {
        protected Engines() {}
        
        public static final long MEM_1GB = (1L) << 30;
        public static final long MEM_1MB = (1L) << 20;
        public static final Engines engine = new Engines();
        
        public static final int default_cuda_stream_pool_size = 128;
        public static final long default_max_memory_size = 2 * MEM_1GB;
        public static final long default_max_transfer_buf_size = 256 * MEM_1MB;
        
        //<editor-fold defaultstate="collapsed" desc="create: mempool">
        public Memp1 memp1(long maxMemorySize) { return new Memp1(maxMemorySize); }
        public Memp2 memp2(long maxMemorySize) { return new Memp2(maxMemorySize); }
        public Memp3 memp3(long maxMemorySize) { return new Memp3(maxMemorySize); }
        
        public Memp1 memp1() { return new Memp1(default_max_memory_size); }
        public Memp2 memp2() { return new Memp2(default_max_memory_size); }
        public Memp3 memp3() { return new Memp3(default_max_memory_size); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="cufa-float32-old">
        public Engine cuda_float32_old(int deviceId, 
                int streamPool_maxsize,
                long maxMemorySize,
                long maxTransferBufSize,//PinnedMemoryPool
                boolean check) 
        {
            CudaFloat32EngineBase base = new CudaFloat32EngineBase(deviceId, streamPool_maxsize);
            if(maxTransferBufSize > 0) base.buf_mempool(new PinnedMempool(maxTransferBufSize));
        
            EngineCore core = new EngineCore_ori(maxMemorySize, check).engineBase(base);
            return new Engine().engineCore(core);
        }

        public Engine cuda_float32_old(int deviceId, long maxMemorySize, long maxBufSize, boolean check) {
            return cuda_float32_old(deviceId, 
                    default_cuda_stream_pool_size,
                    maxMemorySize, 
                    maxBufSize, 
                    check);
        }
        
        public Engine cuda_float32_old(int deviceId, long maxMemorySize, long maxBufSize) {
            return cuda_float32_old(deviceId,
                    default_cuda_stream_pool_size, 
                    maxMemorySize,
                    maxBufSize,
                    true);
        }
        
         public Engine cuda_float32_old(int deviceId, long maxMemorySize) {
            return cuda_float32_old(deviceId,
                    default_cuda_stream_pool_size, 
                    maxMemorySize,
                    default_max_transfer_buf_size,
                    true);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="cufa-float32">
        public Engine cuda_float32(int deviceId, 
                int streamPool_maxsize,
                Mempool mempool, 
                long max_trans_buf_size,//PinnedMemoryPool
                boolean check) 
        {
            CudaFloat32EngineBase base = new CudaFloat32EngineBase(deviceId, streamPool_maxsize);
            if(max_trans_buf_size > 0) base.buf_mempool(new PinnedMempool(max_trans_buf_size));
            
            EngineCore core = new EngineCore(mempool, check).engineBase(base);
            return new Engine().engineCore(core);
        }
        
        public Engine cuda_float32(int deviceId, Mempool memp, long max_trans_buf_size) {
            return cuda_float32(deviceId, 
                    default_cuda_stream_pool_size,
                    memp,
                    max_trans_buf_size, 
                    true);
        }
        
       public Engine cuda_float32(int deviceId, Mempool memp) {
            return cuda_float32(deviceId, 
                    default_cuda_stream_pool_size,
                    memp, 
                    default_max_transfer_buf_size,
                    true);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: UnitBuilder">
    public static class UnitBuilder 
    {
        public final UnitFunctional F = UnitFunctional.F;
        protected UnitBuilder() {}
        
        public static final UnitBuilder nn = new UnitBuilder();
        
        public static boolean simple_math2_inplace = true;
        public static boolean simple_tensor_inplace = true;
        public static boolean simple_affine_inplace = true;
        public static boolean simple_pool2D_avg_ignore_padding = false;
        public static boolean dual_math2_likeX1 = true;
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.math1">
        public Abs abs() { return new Abs(1.0f, 0.0f); }
        public Abs abs(float alpha, float beta) {
            return new Abs(alpha, beta);
        }
        
        public Sin sin() { return new Sin(1.0f, 0.0f); }
        public Sin sin(float alpha, float beta) {
            return new Sin(alpha, beta);
        }
        public Cos cos() { return new Cos(1.0f, 0.0f);}
        public Cos cos(float alpha, float beta) {
            return new Cos(alpha, beta);
        }
        
        public Quadratic square() {
            return new Quadratic(1.0f, 0.0f, 0.0f);
        }
        public Quadratic square(float alpha) {
            return new Quadratic(alpha, 0.0f, 0.0f);
        }
        public Quadratic quadratic(float alpha, float beta, float gamma){
            return new Quadratic(alpha, beta, gamma);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.math2">
        public Rpl rpl() {
            return new Rpl(simple_math2_inplace, 1.0f, 0.0f, 0.0f);
        }
        public Rpl rpl(float alpha, float beta, float gamma) {
            return new Rpl(simple_math2_inplace, alpha, beta, gamma);
        }
        public Rpl rpl(boolean inplace) {
            return new Rpl(inplace, 1.0f, 0.0f, 0.0f);
        }
        public Rpl rpl(boolean inplace, float alpha, float beta, float gamma) {
            return new Rpl(inplace, alpha, beta, gamma);
        }
        
        //<editor-fold defaultstate="collapsed" desc="linear & scalar: add, sub, div, mul">
        public Linear sadd(float C) { 
            return new Linear(simple_math2_inplace, 1.0f, C); 
        }
        public Linear ssub(float C) {
            return new Linear(simple_math2_inplace, 1.0f, -C); 
        }
        public Linear smul(float C) { 
            return new Linear(simple_math2_inplace, C, 0.0f); 
        }
        public Linear sdiv(float C) {
            return new Linear(simple_math2_inplace, 1.0f / C, 0.0f); 
        }
        public Linear linear(float alpha, float beta) {
            return new Linear(simple_math2_inplace, alpha, beta);
        }
        
        public Linear sadd(boolean inplace, float C) { 
            return new Linear(inplace, 1.0f, C); 
        }
        public Linear ssub(boolean inplace, float C) { 
            return new Linear(inplace, 1.0f, -C); 
        }
        public Linear smul(boolean inplace, float C) { 
            return new Linear(inplace, C, 0.0f); 
        }
        public Linear sdiv(boolean inplace, float C) {
            return new Linear(inplace, 1.0f / C, 0.0f);
        }
        public Linear linear(boolean inplace, float alpha, float beta) {
            return new Linear(inplace, alpha, beta);
        }
        //</editor-fold>

        public Relu relu() { return new Relu(simple_math2_inplace); }
        public Relu relu(boolean inplace) {
            return new Relu(inplace);
        }
        
        public static float leakyRelu_negative_slope = 0.01f;
        public LeakyRelu leakyRelu() {
            return new LeakyRelu(simple_math2_inplace, leakyRelu_negative_slope); 
        }
        public LeakyRelu leakyRelu(float negative_slope) {
            return new LeakyRelu(simple_math2_inplace, negative_slope);
        }
        public LeakyRelu leakyRelu(boolean inplace) {
            return new LeakyRelu(inplace, leakyRelu_negative_slope); 
        }
        public LeakyRelu leakyRelu(boolean inplace, float negative_slope) {
            return new LeakyRelu(inplace, negative_slope);
        }

        
        public Softplus softplus() { return new Softplus(simple_math2_inplace); }
        public Softplus softplus(boolean inplace) {
            return new Softplus(inplace);
        }

        
        public static float elu_negative_slope = 0.01f; 
        public static float elu_alpha = 1.0f;
        public Elu elu() { 
            return elu(simple_math2_inplace, elu_alpha, elu_negative_slope); 
        }
        public Elu elu(float negative_slope) { 
            return elu(simple_math2_inplace, elu_alpha, negative_slope); 
        }
        public Elu elu(float alpha, float negative_slope) {
            return new Elu(simple_math2_inplace, alpha, negative_slope);
        }
        
        public Elu elu(boolean inplace) { 
            return elu(inplace, 1.0f, 1.0f);
        }
        public Elu elu(boolean inplace, float negative_slope) { 
            return elu(inplace, 1.0f, negative_slope); 
        }
        public Elu elu(boolean inplace, float alpha, float negative_slope) {
            return new Elu(inplace, alpha, negative_slope);
        }

        
        public Exp exp() { 
            return new Exp(simple_math2_inplace, 1.0f, 0.0f); 
        }
        public Exp exp(float alpha, float beta) {
            return new Exp(simple_math2_inplace, alpha, beta); 
        }
        
        public Exp exp(boolean inplace) {
            return new Exp(inplace, 1.0f, 0.0f); 
        }
        public Exp exp(boolean inplace, float alpha, float beta) { 
            return new Exp(inplace, alpha, beta); 
        }
        
        
        public Log log() {
            return new Log(simple_math2_inplace, 1.0f, 0.0f); 
        }
        public Log log(float alpha, float beta) { 
            return new Log(simple_math2_inplace, alpha, beta); 
        }
        public Log log(boolean inplace) { 
            return new Log(inplace, 1.0f, 0.0f); 
        }
        public Log log(boolean inplace, float alpha, float beta) { 
            return new Log(inplace, alpha, beta); 
        }
        
        
        public Sqrt sqrt() {
            return new Sqrt(simple_math2_inplace, 1.0f, 0.0f); 
        }
        public Sqrt sqrt(float alpha, float beta) {
            return new Sqrt(simple_math2_inplace, alpha, beta);
        }
        
        public Sqrt sqrt(boolean inplace) { 
            return new Sqrt(inplace, 1.0f, 0.0f); 
        }
        public Sqrt sqrt(boolean inplace, float alpha, float beta) {
            return new Sqrt(inplace, alpha, beta);
        }
        
        
        public Sigmoid sigmoid() { return new Sigmoid(simple_math2_inplace); }
        public Sigmoid sigmoid(boolean inplace) {
            return new Sigmoid(inplace);
        }

        
        public Tanh tanh() { return new Tanh(simple_math2_inplace); }
        public Tanh tanh(boolean inplace) {
            return new Tanh(inplace);
        }

        
        public Softmax softmax() { 
            return new Softmax(simple_math2_inplace, -1); 
        }
        public Softmax softmax(int features) { 
            return new Softmax(simple_math2_inplace, features); 
        }
        public Softmax softmax(boolean inplace) {
            return new Softmax(inplace, -1);
        }
        public Softmax softmax(boolean inplace, int features) {
            return new Softmax(inplace, features);
        }
        
        
        public LogSoftmax log_softmax() { 
            return new LogSoftmax(simple_math2_inplace, -1); 
        }
        public LogSoftmax log_softmax(int features) { 
            return new LogSoftmax(simple_math2_inplace, features); 
        }
        public LogSoftmax log_softmax(boolean inplace) {
            return new LogSoftmax(inplace, -1);
        }
        public LogSoftmax log_softmax(boolean inplace, int features) {
            return new LogSoftmax(inplace, features);
        }

        
        public HalfSin halfSin(float Amp) { 
            return new HalfSin(simple_math2_inplace, Amp, 1.0f, 0.0f); 
        }
        public HalfSin halfSin(float Amp, float alpha, float beta) {
            return new HalfSin(simple_math2_inplace, Amp, alpha, beta);
        }
        
        public HalfSin halfSin(boolean inplace, float Amp) {
            return halfSin(inplace, Amp, 1.0f, 0.0f);
        }
        public HalfSin halfSin(boolean inplace, float Amp, float alpha, float beta) {
            return new HalfSin(inplace, Amp, alpha, beta);
        }
        
        
        public Tan tan() { 
            return new Tan(simple_math2_inplace, 1.0f, 0.0f); 
        }
        public Tan tan(float alpha, float beta) {
            return new Tan(simple_math2_inplace, alpha, beta); 
        }
        
        public Tan tan(boolean inplace) { 
            return new Tan(inplace, 1.0f, 0.0f); 
        }
        public Tan tan(boolean inplace, float alpha, float beta) {
            return new Tan(inplace, alpha, beta);
        }
        
        
        public Cot cot() { 
            return new Cot(simple_math2_inplace, 1.0f, 0.0f); 
        }
        public Cot cot(float alpha, float beta) {
            return new Cot(simple_math2_inplace, alpha, beta); 
        }
        
        public Cot cot(boolean inplace) { 
            return new Cot(inplace, 1.0f, 0.0f); 
        }
        public Cot cot(boolean inplace, float alpha, float beta) {
            return new Cot(inplace, alpha, beta);
        }

        
        public Arcsin arcsin() {
            return new Arcsin(simple_math2_inplace, 1.0f, 0.0f); }
        
        public Arcsin arcsin(float alpha, float beta) {
            return new Arcsin(simple_math2_inplace, alpha, beta); 
        } 
        
        public Arcsin arcsin(boolean inplace) { 
            return new Arcsin(inplace, 1.0f, 0.0f);
        } 
        public Arcsin arcsin(boolean inplace, float alpha, float beta) {
            return new Arcsin(inplace, alpha, beta);
        } 
        
        
        public Arctan arctan() { 
            return new Arctan(simple_math2_inplace, 1.0f, 0.0f); 
        } 
        public Arctan arctan(float alpha, float beta) {
            return new Arctan(simple_math2_inplace, alpha, beta); 
        }
        
        public Arctan arctan(boolean inplace) { 
            return new Arctan(inplace, 1.0f, 0.0f); 
        } 
        public Arctan arctan(boolean inplace, float alpha, float beta) {
            return new Arctan(inplace, alpha, beta);
        } 
        
        public BernouliMul bernouli_mul(float p, float v1, float v2) {
            return new BernouliMul(simple_math2_inplace, p, v1, v2);
        }
        public BernouliMul bernouli_mul(boolean inplace, float p, float v1, float v2) {
            return new BernouliMul(inplace, p, v1, v2);
        }
        
        public Dropout dropout(float nonzero_p) { 
            return new Dropout(simple_math2_inplace, nonzero_p); 
        }
        public Dropout dropout(boolean inplace, float nonzero_prop) {
            return new Dropout(inplace, nonzero_prop);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.tensor">
        public Rot180 rot180() { return new Rot180(simple_tensor_inplace); }
        public Rot180 rot180(boolean inplace) {
            return new Rot180(inplace);
        }
        
        public Flatten flaten() { return new Flatten(simple_tensor_inplace); }
        public Flatten flaten(boolean inplace) {
            return new Flatten(inplace);
        }
        
        public View view(int... outDim) { 
            return new View(simple_tensor_inplace, outDim); 
        }
        public View view(boolean inplace, int... outDim) {
            return new View(inplace, outDim);
        }
        
        public Reshape reshape(int... outDim) { 
            return new Reshape(simple_tensor_inplace, outDim); 
        }
        public Reshape reshape(boolean inplace, int... outDim) {
            return new Reshape(inplace, outDim);
        }
        
        public Transpose transpose(int dimIdx1, int dimIdx2) {
            return new Transpose(simple_tensor_inplace, dimIdx1, dimIdx2);
        }
        public Transpose transpose(boolean inplace, int dimIdx1, int dimIdx2) {
            return new Transpose(inplace, dimIdx1, dimIdx2);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.affine">
        public Affine affine(int... feature_dim) {
            return new Affine(simple_affine_inplace, feature_dim);
        }
        public Affine affine(boolean inplace, int... feature_dim) {
            return new Affine(inplace, feature_dim);
        }

        
        public static float layerNorm_eps = 1e-5f;
        public static boolean layerNorm_affine = true;
        
        public LayerNorm layerNorm(int... feature_dim) {
            return new LayerNorm(simple_affine_inplace, layerNorm_affine, layerNorm_eps, feature_dim);
        }     
        public LayerNorm layerNorm(boolean affine, float eps, int... feature_dim) {
            return new LayerNorm(simple_affine_inplace, affine, eps, feature_dim);
        }
        public LayerNorm layerNorm(boolean inplace, int... feature_dim) {
            return new LayerNorm(inplace, layerNorm_affine, layerNorm_eps, feature_dim);
        }     
        public LayerNorm layerNorm(boolean inplace, boolean affine, float eps, int... feature_dim) {
            return new LayerNorm(inplace, affine, eps, feature_dim);
        }

        
        public static float batchNorm_eps = 1e-5f;
        public static float batchNorm_beta1 = 0.9f;
        public static float batchNorm_beta2 = 0.9f;
        public static boolean batchNorm_affine = true;
        
        public GlobalSqBatchNorm global_sqBatchNorm(int... feature_dim) {
            return new GlobalSqBatchNorm(simple_affine_inplace, batchNorm_affine,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalSqBatchNorm(simple_affine_inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean inplace, int... feature_dim) {
            return new GlobalSqBatchNorm(inplace, true,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalSqBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        
        public SqBatchNorm sqBatchNorm(int... feature_dim) {
            return new SqBatchNorm(simple_affine_inplace, batchNorm_affine,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new SqBatchNorm(simple_affine_inplace, batchNorm_affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean inplace, int... feature_dim) {
            return new SqBatchNorm(inplace, true,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new SqBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        
         public GlobalBatchNorm global_batchNorm(int... feature_dim) {
            return new GlobalBatchNorm(simple_affine_inplace, batchNorm_affine,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalBatchNorm(simple_affine_inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean inplace, int... feature_dim) {
            return new GlobalBatchNorm(inplace, true,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        
        public BatchNorm batchNorm(int... feature_dim) {
            return new BatchNorm(simple_affine_inplace, batchNorm_affine,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public BatchNorm batchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new BatchNorm(simple_affine_inplace, batchNorm_affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public BatchNorm batchNorm(boolean inplace, int... feature_dim) {
            return new BatchNorm(inplace, true,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public BatchNorm batchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new BatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.AvgPool2D">
        public AvgPool2D avgPool2D(int div) {
            return new AvgPool2D(simple_pool2D_avg_ignore_padding,
                    div, div, div, div, 0, 0);
        }
        public AvgPool2D avgPool2D(int kernel_size, int stride, int padding) {
            return new AvgPool2D(simple_pool2D_avg_ignore_padding,
                    kernel_size, kernel_size,
                    stride, stride,
                    padding, padding);
        }
        public AvgPool2D avgPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new AvgPool2D(simple_pool2D_avg_ignore_padding,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width);
        }
        
        public AvgPool2D avgPool2D(boolean ignore_padding, int div) {
            return new AvgPool2D(ignore_padding,
                    div, div, div, div, 0, 0);
        }
        public AvgPool2D avgPool2D(boolean ignore_padding, int kernel_size, int stride, int padding) {
            return new AvgPool2D(ignore_padding,
                    kernel_size, kernel_size,
                    stride, stride,
                    padding, padding);
        }
        public AvgPool2D avgPool2D(boolean ignore_padding,
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new AvgPool2D(ignore_padding,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width);
        }
        //</editor-fold> 
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.NaiveMaxPool2D">
        public NaiveMaxPool2D naive_maxPool2D(int div) {
            return new NaiveMaxPool2D(div, div, div, div, 0, 0);
        }
        public NaiveMaxPool2D naive_maxPool2D(int kernel_size, int stride, int padding) {
            return new NaiveMaxPool2D(
                    kernel_size, kernel_size,
                    stride, stride,
                    padding, padding);
        }
        public NaiveMaxPool2D naive_maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_widt) {
            return new NaiveMaxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_widt);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.MaxPool2D">
        public MaxPool2D maxPool2D(int div) {
            return new MaxPool2D(div, div, div, div, 0, 0);
        }
        public MaxPool2D maxPool2D(int kernel_size, int stride, int padding) {
            return new MaxPool2D(
                    kernel_size, kernel_size,
                    stride, stride,
                    padding, padding);
        }
        public MaxPool2D maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_widt) {
            return new MaxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_widt);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.AdaptiveAvgPool2D">
        public AdaptiveAvgPool2D adaptive_avgPool2D(int out_size) {
            return new AdaptiveAvgPool2D(simple_pool2D_avg_ignore_padding, 
                    out_size, out_size);
        }
        public AdaptiveAvgPool2D adaptive_avgPool2D(int out_height, int out_width) {
            return new AdaptiveAvgPool2D(simple_pool2D_avg_ignore_padding, 
                    out_height, out_width);
        }
        
        public AdaptiveAvgPool2D adaptive_avgPool2D(boolean ignore_padding, int out_size) {
            return new AdaptiveAvgPool2D(ignore_padding,
                    out_size, out_size);
        }
        public AdaptiveAvgPool2D adaptive_avgPool2D(boolean ignore_padding, int out_height, int out_width) {
            return new AdaptiveAvgPool2D(ignore_padding, 
                    out_height, out_width);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.AdaptiveMaxPool2D_Naive">
        public AdaptiveNaiveMaxPool2D adaptive_naive_maxPool2D(int out_size) {
            return new AdaptiveNaiveMaxPool2D(out_size, out_size);
        }
        public AdaptiveNaiveMaxPool2D adaptive_naive_maxPool2D(int out_height, int out_width) {
            return new AdaptiveNaiveMaxPool2D(out_height, out_width);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool2d.AdaptiveMaxPool2D">
        public AdaptiveMaxPool2D adaptive_maxPool2D(int out_size) {
            return new AdaptiveMaxPool2D(out_size, out_size);
        }
        public AdaptiveMaxPool2D adaptive_maxPool2D(int out_height, int out_width) {
            return new AdaptiveMaxPool2D(out_height, out_width);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: AvgUnpool2D">
        //<editor-fold defaultstate="collapsed" desc="output_size = [-1, -1]">
        public AvgUnpool2D avgUnpool2D(int mul) {
            return new AvgUnpool2D(mul, mul, mul, mul, 0, 0, -1, -1);
        }
        
        public AvgUnpool2D avgUnpool2D(int kernel_size, int stride, int padding) {
            return new AvgUnpool2D(kernel_size, kernel_size,
                    stride, stride,
                    padding, padding, -1, -1);
        }
        
        public AvgUnpool2D avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width)
        {
            return new AvgUnpool2D(kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width, -1, -1);
        }
        //</editor-fold>

        public AvgUnpool2D avgUnpool2D(int scaleUp, int out_size) {
            return new AvgUnpool2D(scaleUp, scaleUp, scaleUp, scaleUp, 0, 0, 
                    out_size, out_size);
        }
        
        public AvgUnpool2D avgUnpool2D(int kernel_size, int stride, int padding, int out_size) {
            return new AvgUnpool2D(kernel_size, kernel_size,
                    stride, stride,
                    padding, padding,
                    out_size, out_size);
        }
        
        public AvgUnpool2D avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width,
                int output_height, int output_width) {
            return new AvgUnpool2D(kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    output_height, output_width);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.Conv3D">
        public Conv3D pointConv3D(boolean biased, 
                int in_channel_num, int out_channel_num) {
            return new Conv3D(biased, 
                    in_channel_num, out_channel_num,
                    1, 1, 1, 1, 0, 0);
        }
        
        public Conv3D conv3D(boolean biased,
                int in_channel_num, int out_channel_num,
                int kernel, int stride, int padding) {
            return new Conv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel, kernel,
                    stride, stride,
                    padding, padding);
        }
        public Conv3D conv3D(boolean biased,
                int in_channel_num, int out_channel_num,
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new Conv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.Deconv3D">
        //<editor-fold defaultstate="collapsed" desc="output_size = [-1, -1]">
        public Deconv3D deconv3D(boolean biased,
                int in_channel_num, int out_channel_num,
                int kernel, int stride, int padding) {
            return new Deconv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel, kernel,
                    stride, stride,
                    padding, padding,
                    -1, -1);
        }
        public Deconv3D deconv3D(boolean biased, 
                int in_channel_num, int out_channel_num,
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new Deconv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    -1, -1);
        }
        //</editor-fold>
        
        public Deconv3D deconv3D(boolean biased, 
                int in_channel_num, int out_channel_num,
                int kernel, int stride, int padding,
                int output_size) {
            return new Deconv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel, kernel,
                    stride, stride,
                    padding, padding,
                    output_size, output_size);
        }
        public Deconv3D deconv3D(boolean biased, 
                int in_channel_num, int out_channel_num,
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width,
                int output_height, int output_width) {
            return new Deconv3D(biased, 
                    in_channel_num, out_channel_num,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    output_height, output_width);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.fullconnect">
        public FullConnect fullconnect(boolean biased, int in_features, int out_features) {
            return new FullConnect(biased, in_features, out_features);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: dual.math1">
        public MatMul matMul() { return new MatMul(); }
        public MatMulT1 matMulT1() { return new MatMulT1(); }
        public MatMulT2 matMulT2() { return new MatMulT2(); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: dual.math2">
        public BatchMatMul batchMatMul() { 
            return batchMatMul(dual_math2_likeX1); 
        }
        public BatchMatMul batchMatMul(boolean likeX1) {
            return new BatchMatMul(likeX1);
        }

        
        public BatchMatMulT1 batchMatMulT1() {
            return batchMatMulT1(dual_math2_likeX1); 
        }
        public BatchMatMulT1 batchMatMulT1(boolean likeX1) {
            return new BatchMatMulT1(likeX1);
        }
        
        
        public BatchMatMulT2 batchMatMulT2() { return batchMatMulT2(
                dual_math2_likeX1); 
        }
        public BatchMatMulT2 batchMatMulT2(boolean likeX1) {
            return new BatchMatMulT2(likeX1);
        }
        
        //<editor-fold defaultstate="collapsed" desc="linear2: add, sub">
        public Linear2 add() { 
            return new Linear2(dual_math2_likeX1, 1.0f, 1.0f, 0.0f);
        }
        public Linear2 sub() {
            return new Linear2(dual_math2_likeX1, 1.0f, -1.0f, 0.0f);
        }
        public Linear2 add(float alpha, float beta) { 
            return new Linear2(dual_math2_likeX1, alpha, beta, 0.0f);
        }
        public Linear2 linear2(float alpha, float beta, float gamma) {
            return new Linear2(dual_math2_likeX1, alpha, beta, gamma);
        }
        
        public Linear2 add(boolean likeX1) { 
            return new Linear2(likeX1, 1.0f, 1.0f, 0.0f);
        }
        public Linear2 sub(boolean likeX1) { 
            return new Linear2(likeX1, 1.0f, -1.0f, 0.0f);
        }
        public Linear2 add(boolean likeX1, float alpha, float beta) { 
            return new Linear2(likeX1, alpha, beta, 0.0f);
        }
        public Linear2 linear2(boolean likeX1, float alpha, float beta, float gamma) {
            return new Linear2(likeX1, alpha, beta, gamma);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="quadratic2: mul, squareAdd, squareSub">
        public Quadratic2 mul() {
            return new Quadratic2(dual_math2_likeX1, 0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f);
        }
        public Quadratic2 mul(float alpha) {
            return new Quadratic2(dual_math2_likeX1, 0.0f, alpha, 0.0f, 
                    0.0f, 0.0f, 0.0f);
        }
        public Quadratic2 squareAdd() {
            return new Quadratic2(dual_math2_likeX1, 1.0f, 0.0f, 1.0f, 
                    0.0f, 0.0f, 0.0f);
        }        
        public Quadratic2 squareSub() {
            return new Quadratic2(dual_math2_likeX1, 1.0f, 0.0f, -1.0f, 
                    0.0f, 0.0f, 0.0f);
        }        
        public Quadratic2 squareAdd(float alpha, float beta) {
            return new Quadratic2(dual_math2_likeX1, alpha, 0.0f, beta, 
                    0.0f, 0.0f, 0.0f);
        }        
        public Quadratic2 quadratic2(float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2(dual_math2_likeX1, k11, k12, k22, 
                    k1, k2, C);
        }
        
        public Quadratic2 mul(boolean likeX1) {
            return new Quadratic2(likeX1, 0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f);
        }
        public Quadratic2 mul(boolean likeX1, float alpha) {
            return new Quadratic2(likeX1, 0.0f, alpha, 0.0f, 
                    0.0f, 0.0f, 0.0f);
        }
        public Quadratic2 squareAdd(boolean likeX1) {
            return new Quadratic2(likeX1, 1.0f, 0.0f, 1.0f, 
                    0.0f, 0.0f, 0.0f);
        }       
        public Quadratic2 squareSub(boolean likeX1) {
            return new Quadratic2(likeX1, 1.0f, 0.0f, -1.0f, 
                    0.0f, 0.0f, 0.0f);
        }      
        public Quadratic2 squareAdd(boolean likeX1, float alpha, float beta) {
            return new Quadratic2(likeX1, alpha, 0.0f, beta, 
                    0.0f, 0.0f, 0.0f);
        }        
        public Quadratic2 quadratic2(boolean likeX1, float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2(likeX1, k11, k12, k22, 
                    k1, k2, C);
        }
        //</editor-fold>
        
        public Div div() { 
            return div(dual_math2_likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);  
        }
        public Div div(float alpha1, float beta1, float alpha2, float beta2, float gamma){
            return div(dual_math2_likeX1, alpha1, beta1, alpha2, beta2, gamma);
        }
        
        public Div div(boolean likeX1) { 
            return div(likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); 
        }
        public Div div(boolean likeX1, float alpha1, float beta1, float alpha2, float beta2, float gamma){
            return new Div(likeX1, alpha1, beta1, alpha2, beta2, gamma);
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="create: reducer">
        public LinearSummary sum() { 
            return new LinearSummary(1.0f, 0.0f); 
        }
        public LinearSummary sum(float alpha) { 
            return new LinearSummary(alpha, 0.0f); 
        }
        public LinearSummary linearSum(float alpha, float beta) {
             return new LinearSummary(alpha, beta); 
        }

        
        public LinearMean mean() { 
            return new LinearMean(1.0f, 0.0f); 
        }
        public LinearMean mean(float alpha) { 
            return new LinearMean(alpha, 0.0f); 
        }
        public LinearMean linearMean(float alpha, float beta) { 
            return new LinearMean(alpha, beta); 
        }
        
        public QuadraticSummary squareSum() {
            return new QuadraticSummary(1.0f, 0.0f, 0.0f);
        }
        public QuadraticSummary squareSum(float alpha) {
            return new QuadraticSummary(alpha, 0.0f, 0.0f);
        }
        public QuadraticSummary quadraticSum(float alpha, float beta, float gamma) {
            return new QuadraticSummary(alpha, beta, gamma);
        }
        
        
        public QuadraticMean squareMean() {
            return new QuadraticMean(1.0f, 0.0f, 0.0f);
        }
        public QuadraticMean squareMean(float alpha) {
            return new QuadraticMean(alpha, 0.0f, 0.0f);
        }
        public QuadraticMean quadraticMean(float alpha, float beta, float gamma) {
            return new QuadraticMean(alpha, beta, gamma);
        }
        
        
        public Concat concat() { 
            return new Concat(-1); 
        }
        public Concat concat(int dimIdx) { 
            return new Concat(dimIdx); 
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: furcation">
        public Split split(int dimIdx, int... section){ 
            return new Split(dimIdx, section);
        }
        
        public Chunk chunk(int dimIdx, int n) {
            return new Chunk(dimIdx, n);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: complex">
        public Sequence sequence(Collection<Unit> units) { return new Sequence(units); }
        public Sequence sequence(Unit... units) { return new Sequence(units); }

        public Branch branch(Collection<Unit> units) { return new Branch(units); }
        public Branch branch(Unit... units) {
            return new Branch(units);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="utils">
        public void print_param_stat(Unit unit) { 
            TreeMap<String, Parameter> map = new TreeMap<>(unit.param_map());
            map.forEach((String name, Parameter p) -> {
                System.out.print("name = [" + name + " ]");
                Tensor tensor = p.ts();
                System.out.print("\t, mean = [ " + tensor.mean().get() + " ]");
                System.out.print("\t, std = [" + tensor.std().get() + "]\n");
            });
        }
        
        public void constant_params(Unit unit, float value) {
            for(Parameter p : unit.params()) p.ts().constant(value).c();
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: UnitFunctional">
    public static class UnitFunctional
    {
        private final UnitBuilder sb = new UnitBuilder();
        protected UnitFunctional() {}
        
        public static final UnitFunctional F = new UnitFunctional();
        
        //<editor-fold defaultstate="collapsed" desc="Carriers">
        //<editor-fold defaultstate="collapsed" desc="SimpleFunctionCarraier">
        private static Tensor[] csp(SimpleUnit sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            //if X[0] is output of OneOffScale, X[0].needCarry = true 
            Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return Y;
        }
        
        private static Tensor[] csp(boolean inplace, SimpleUnit sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            if(!inplace){
                //if X[0] is output of OneOffScale, X[0].needCarry = true
                Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
                
                sc.hook_after_backward((self)->{
                    Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                    deltaX.carry(sc.deltaY());
                });
            }
            
            return Y;
        }
        
        private static Tensor[] fsp(SimpleUnit sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            if(!UnitBuilder.simple_math2_inplace){
                //if X[0] is output of OneOffScale, X[0].needCarry = true
                Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
                
                sc.hook_after_backward((self)->{
                    Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                    deltaX.carry(sc.deltaY());
                });
            }
            
            return Y;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="MaxPoolCarraier">
        private static Tensor[] csp_maxPool(MaxPool2D sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            //if X[0] is output of OneOffScale, X[0].needCarry = true 
            Y[0].need_carry(true);
            Tensor Index = sc.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray X[0], Index
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Index(MaxPool2D sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            //if X[0] is output of OneOffScale, X[0].needCarry = true 
            Y[0].need_carry(true);
            Tensor Index = sc.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray X[0], Index
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return new Tensor[] {Y[0], sc.Index()};
        }
      
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="AdaptiveMaxPoolCarraier">
        private static Tensor[] csp_maxPool(AdaptiveMaxPool2D sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            //if X[0] is output of OneOffScale, X[0].needCarry = true 
            Y[0].need_carry(true);
            Tensor Index = sc.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray X[0], Index
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Index(AdaptiveMaxPool2D sc, Tensor[] X) {//simple function carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            //if X[0] is output of OneOffScale, X[0].needCarry = true 
            Y[0].need_carry(true);
            Tensor Index = sc.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray X[0], Index
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return new Tensor[] {Y[0], sc.Index()};
        }
      
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="DualCarrier">
        private static Tensor[] cdu(DualUnit sc, Tensor[] X) {//dual scale carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            Y[0].need_carry(true); Y[0].carry(X);//X1, X2 -> Y
            
            sc.hook_after_backward((self)->{//deltaY -> deltaX1, deltaX2
                Tensor deltaX1 = sc.deltaX1(); deltaX1.need_carry(true);
                Tensor deltaX2 = sc.deltaX2(); deltaX2.need_carry(true);
                deltaX1.carry(sc.deltaY());
            });
            
            return Y;
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="ReducerCarrier">
        private static Tensor[] cre(Reducer sc, Tensor[] X) {//Reducer Carrier
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            Y[0].need_carry(true); Y[0].carry(X);//X[0: N-1] -> Y
            
            sc.hook_after_backward((self)->{
                Tensor[] deltaX = sc.deltaX();
                for(Tensor dX : deltaX) dX.need_carry(true);
                deltaX[0].carry(sc.deltaY());
            });
            
            return Y;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="FurcationCarrier">
        private Tensor[] cfc(Furcation sc, Tensor[] X) {//one input, multi output
            Tensor[] Y = sc.init(X[0].engine()).forward(X);
            
            for (Tensor out : Y) out.need_carry(true);
            Y[0].carry(X[0]);
            
            sc.hook_after_backward((self)->{
                Tensor deltaX = sc.deltaX(); deltaX.need_carry(true);
                deltaX.carry(sc.deltaY());
            });
            
            return Y;
        }
        //</editor-fold>
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="class: EvalOneOff">
        public final class EvalOneOff 
        {
            //<editor-fold defaultstate="collapsed" desc="eval: simple.pool2D.maxPool2D">
            public Tensor[] maxPool2D(int div, Tensor... X) {
                return csp(sb.maxPool2D(div).eval(), X);
            }
            public Tensor[] maxPool2D(int kernel, int stride, int padding, Tensor... X) {
                return csp(sb.maxPool2D(kernel, stride, padding).eval(), X);
            }
            public Tensor[] maxPool2D(
                    int kernel_height, int kernel_width,
                    int stride_height, int stride_width,
                    int padding_height, int padding_width,
                    Tensor... X) {
                return csp(sb.maxPool2D(
                        kernel_height, kernel_width,
                        stride_height, stride_width,
                        padding_height, padding_width).eval(), X);
            }
            //</editor-fold>
            //<editor-fold defaultstate="collapsed" desc="eval: simple.pool2D.AdaptiveMaxPool2D">
            public Tensor[] adaptive_maxPool2D(int out_size, Tensor... X) {
                return csp(sb.adaptive_maxPool2D(out_size).eval(), X);
            }

            public Tensor[] adaptive_maxPool2D(int out_height, int out_width, Tensor... X) {
                return csp(sb.adaptive_maxPool2D(out_width).eval(), X);
            }
            //</editor-fold>
            //<editor-fold defaultstate="collapsed" desc="eval: simple.math2">
            public Tensor[] bernouli_mul(float p, float v1, float v2, Tensor... X) {
                return fsp(sb.bernouli_mul(p, v1, v2).eval(), X);
            }
            public Tensor[] bernouli_mul(boolean inplace, float p, float v1, float v2, Tensor... X) {
                return csp(inplace, sb.bernouli_mul(inplace, p, v1, v2).eval(), X);
            }
            
            public Tensor[] dropout(float nonzero_p, Tensor...X) {
                return fsp(sb.dropout(nonzero_p).eval(), X); 
            }
            public Tensor[] dropout(boolean inplace, float nonzero_p, Tensor...X) {
                return csp(inplace, sb.dropout(inplace, nonzero_p).eval(), X);
            }
            //</editor-fold>
        }
        //</editor-fold>
        private final EvalOneOff eval = new EvalOneOff();
        public EvalOneOff eval() {return eval;}

        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.math1">
        public Tensor[] abs(Tensor... X) {
            return csp(sb.abs(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] abs(float alpha, float beta, Tensor...X) {
            return csp(sb.abs(alpha, beta), X);
        }
        
        
        public Tensor[] sin(Tensor... X) { 
            return csp(sb.sin(), X); 
        }
        public Tensor[] sin(float alpha, float beta, Tensor... X) {
            return csp(sb.sin(alpha, beta), X);
        }
        public Tensor[] cos(Tensor... X) { 
            return csp(sb.cos(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] cos(float alpha, float beta, Tensor... X) {
            return csp(sb.cos(alpha, beta), X);
        }
        
        
        public Tensor[] square(Tensor... X) { 
            return csp(sb.square(), X); 
        }
        public Tensor[] square(float alpha, Tensor... X) {
            return csp(sb.square(alpha), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] quadratic(float alpha, float beta, float gamma, Tensor... X){
            return csp(sb.quadratic(alpha, beta, gamma), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.math2">
        public Tensor[] rpl(Tensor... X) { 
            return fsp(sb.rpl(), X); 
        }
        public Tensor[] rpl(float alpha, float beta, float gamma, Tensor... X) {
            return fsp(sb.rpl(alpha, beta, gamma), X);
        }
        
        public Tensor[] rpl(boolean inplace, Tensor... X) { 
            return csp(inplace, sb.rpl(inplace), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] rpl(boolean inplace, float alpha, float beta, float gamma, Tensor... X) {
            return csp(inplace, sb.rpl(inplace, alpha, beta, gamma), X);
        }
        
        //<editor-fold defaultstate="collapsed" desc="linear & scalar: add, sub, div, mul">
        public Tensor[] sadd(float C, Tensor... X) {
            return fsp(sb.sadd(C), X); 
        }
        public Tensor[] ssub(float C, Tensor... X) {
            return fsp(sb.ssub(C), X); 
        }
        public Tensor[] smul(float C, Tensor... X) {
            return fsp(sb.smul(C), X); 
        }
        public Tensor[] sdiv(float C, Tensor... X) {
            return fsp(sb.sdiv(C), X); 
        }
        public Tensor[] linear(float alpha, float beta, Tensor... X) {
            return fsp(sb.linear(alpha, beta), X);
        }
        
        public Tensor[] sadd(boolean inplace, float C, Tensor... X){
            return csp(inplace, sb.sadd(inplace, C), X); 
        }
        public Tensor[] ssub(boolean inplace, float C, Tensor... X){
            return csp(inplace, sb.ssub(inplace, C), X); 
        }
        public Tensor[] smul(boolean inplace, float C, Tensor... X){
            return csp(inplace, sb.smul(inplace, C), X); 
        }
        public Tensor[] sdiv(boolean inplace, float C, Tensor... X){
            return csp(inplace, sb.sdiv(inplace, C), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] linear(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, sb.linear(inplace, alpha, beta), X);
        }
        //</editor-fold>
        
        public Tensor[] exp(Tensor... X) { 
            return fsp(sb.exp(), X); 
        }
        public Tensor[] exp(float alpha, float beta, Tensor... X) { 
            return fsp(sb.exp(alpha, beta), X); 
        }
        public Tensor[] exp(boolean inplace, Tensor... X) {
            return csp(inplace, sb.exp(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] exp(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, sb.exp(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] log(Tensor... X) { 
            return fsp(sb.log(), X); 
        }
        public Tensor[] log(float alpha, float beta, Tensor... X) { 
            return fsp(sb.log(alpha, beta), X); 
        }
        public Tensor[] log(boolean inplace, Tensor... X) {
            return csp(inplace, sb.log(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] log(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, sb.log(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] sqrt(Tensor... X) {
            return fsp(sb.sqrt(), X); 
        }
        public Tensor[] sqrt(float alpha, float beta, Tensor... X) { 
            return fsp(sb.sqrt(alpha, beta), X); 
        }
        public Tensor[] sqrt(boolean inplace,  Tensor... X) { 
            return csp(inplace, sb.sqrt(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] sqrt(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, sb.sqrt(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] relu(Tensor... X) { 
            return fsp(sb.relu(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] relu(boolean inplace, Tensor... X) {
           return csp(inplace, sb.relu(inplace), X);
        }
        

        public Tensor[] leakyRelu(Tensor... X) { 
            return fsp(sb.leakyRelu(), X); 
        }
        public Tensor[] leakyRelu(float negative_slope, Tensor... X) {
            return fsp(sb.leakyRelu(negative_slope), X);
        }
        public Tensor[] leakyRelu(boolean inplace, Tensor... X) { 
            return csp(inplace, sb.leakyRelu(inplace), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] leakyRelu(boolean inplace, float negative_slope, Tensor... X) {
            return csp(inplace, sb.leakyRelu(inplace, negative_slope), X);
        }

        
        public Tensor[] softplus(Tensor... X) {
            return fsp(sb.softplus(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] softplus(boolean inplace, Tensor... X) {
            return csp(inplace, sb.softplus(inplace), X);
        }

        
        public Tensor[] elu(Tensor... X) {
            return fsp(sb.elu(), X); 
        }
        public Tensor[] elu(float negative_slope, Tensor... X) { 
            return fsp(sb.elu(negative_slope), X); 
        }
        public Tensor[] elu(float alpha, float negative_slope, Tensor... X) {
            return fsp(sb.elu(alpha, negative_slope), X);
        }
        public Tensor[] elu(boolean inplace, Tensor... X) {
            return csp(inplace, sb.elu(inplace), X); 
        }
        public Tensor[] elu(boolean inplace, float negative_slope, Tensor... X) {  
            return csp(inplace, sb.elu(inplace, negative_slope), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] elu(boolean inplace, float alpha, float negative_slope, Tensor... X) {
            return csp(inplace, sb.elu(inplace, alpha, negative_slope), X);
        }

        
        public Tensor[] sigmoid(Tensor... X) { 
            return fsp(sb.sigmoid(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] sigmoid(boolean inplace, Tensor... X) {
            return csp(inplace, sb.sigmoid(inplace), X);
        }


        public Tensor[] tanh(Tensor... X) { 
            return fsp(sb.tanh(), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] tanh(boolean inplace, Tensor... X) {
            return csp(inplace, sb.tanh(inplace), X);
        }

        
        public Tensor[] softmax(Tensor... X) {
            return fsp(sb.softmax(), X);
        }
        public Tensor[] softmax(int features, Tensor... X) {
            return fsp(sb.softmax(features), X);
        }
        public Tensor[] softmax(boolean inplace, Tensor... X) {
            return csp(inplace, sb.softmax(inplace), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] softmax(boolean inplace, int features, Tensor... X) {
            return csp(inplace, sb.softmax(inplace, features), X);
        }
        
        
        public Tensor[] log_softmax(Tensor... X) {
            return fsp(sb.log_softmax(), X);
        }
        public Tensor[] log_softmax(int features, Tensor... X) {
            return fsp(sb.log_softmax(features), X);
        }
        public Tensor[] log_softmax(boolean inplace, Tensor... X) {
            return csp(inplace, sb.log_softmax(inplace), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] log_softmax(boolean inplace, int features, Tensor... X) {
            return csp(inplace, sb.log_softmax(inplace, features), X);
        }

        
        public Tensor[] halfSin(float Amp, Tensor... X) { 
            return fsp(sb.halfSin(Amp), X); 
        }
        public Tensor[] halfSin(float Amp, float alpha, float beta, Tensor... X) {
            return fsp(sb.halfSin(Amp, alpha, beta), X);
        }
        public Tensor[] halfSin(boolean inplace, float Amp, Tensor... X) {
            return csp(inplace, sb.halfSin(inplace, Amp), X);
        }
        public Tensor[] halfSin(boolean inplace, float Amp, float alpha, float beta, Tensor... X) {
            return csp(inplace, sb.halfSin(inplace, Amp, alpha, beta), X);
        }
        
        
        public Tensor[] tan(Tensor...X) { 
            return fsp(sb.tan(), X); 
        }
        public Tensor[] tan(float alpha, float beta, Tensor...X) { 
            return fsp(sb.tan(alpha, beta), X); 
        }
        public Tensor[] tan(boolean inplace, Tensor...X) { 
            return csp(inplace, sb.tan(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] tan(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, sb.tan(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] cot(Tensor...X) { 
            return fsp(sb.cot(), X); 
        }
        public Tensor[] cot(float alpha, float beta, Tensor...X) { 
            return fsp(sb.cot(alpha, beta), X); 
        }
        public Tensor[] cot(boolean inplace, Tensor...X) { 
            return csp(inplace, sb.cot(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] cot(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, sb.cot(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] arcsin(Tensor...X) { 
            return fsp(sb.arcsin(), X); 
        }
        public Tensor[] arcsin(float alpha, float beta, Tensor...X) {
            return fsp(sb.arcsin(alpha, beta), X); 
        }
        public Tensor[] arcsin(boolean inplace, Tensor...X) {
            return csp(inplace, sb.arcsin(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] arcsin(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, sb.arcsin(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] arctan(Tensor...X) { 
            return fsp(sb.arctan(), X); 
        }
        public Tensor[] arctan(float alpha, float beta, Tensor...X) {
            return fsp(sb.arctan(alpha, beta), X); 
        }
        public Tensor[] arctan(boolean inplace, Tensor...X) { 
            return csp(inplace, sb.arctan(inplace), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] arctan(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, sb.arctan(inplace, alpha, beta), X);
        }
        
        
        public Tensor[] bernouli_mul(float p, float v1, float v2, Tensor... X) {
            return fsp(sb.bernouli_mul(p, v1, v2), X);
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] bernouli_mul(boolean inplace, float p, float v1, float v2, Tensor... X) {
            return csp(inplace, sb.bernouli_mul(inplace, p, v1, v2), X);
        }
                
        
        public Tensor[] dropout(float nonzero_p, Tensor...X) {
            return fsp(sb.dropout(nonzero_p), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] dropout(boolean inplace, float nonzero_p, Tensor...X) {
            return csp(inplace, sb.dropout(inplace, nonzero_p), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.tensor">
        public Tensor[] rot180(Tensor... X) { return fsp(sb.rot180(), X); }
        public Tensor[] rot180(boolean inplace, Tensor... X) {
            return fsp(sb.rot180(inplace), X);
        }
        
        
        public Tensor[] flaten(Tensor... X) { return fsp(sb.flaten(), X); }
        public Tensor[] flaten(boolean inplaced, Tensor... X) {
            return csp(inplaced, sb.flaten(inplaced), X);
        }
        
        
        public Tensor[] view(Tensor[] X, int... outDim) { 
            return fsp(sb.view(outDim), X); 
        }
        public Tensor[] view(boolean inplace, Tensor[] X, int... outDim) {
            return csp(inplace, sb.view(inplace, outDim), X);
        }
        
        public Tensor[] view_flaten(Tensor... X) {
            return fsp(sb.view(X[0].dim(0), -1), X);
        }
        public Tensor[] view_flaten(boolean inplace, Tensor... X) {
            return csp(inplace, sb.view(inplace, X[0].dim(0), -1), X);
        }
        
        public Tensor[] reshape(Tensor[] X, int... outDim) { 
            return fsp(sb.reshape(outDim), X); 
        }
        public Tensor[] reshape(boolean inplace, Tensor[] X, int... outDim) {
            return csp(inplace, sb.reshape(inplace, outDim), X);
        }
        
        
        public Tensor[] transpose(int dimIdx1, int dimIdx2, Tensor... X) {
            return fsp(sb.transpose(dimIdx1, dimIdx2), X);
        }
        public Tensor[] transpose(boolean inplace, int dimIdx1, int dimIdx2, Tensor... X) {
            return csp(inplace, sb.transpose(inplace, dimIdx1, dimIdx2), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.AvgPool2D">
        public Tensor[] avgPool2D(int div, Tensor... X) { 
            return csp(sb.avgPool2D(div), X); 
        }
        public Tensor[] avgpool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp(sb.avgPool2D(kernel, stride, padding), X);
        }
        public Tensor[] avgpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp(sb.avgPool2D(kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width), X);
        }
        
        public Tensor[] avgPool2D(boolean ignore_padding, int div, Tensor... X) { 
            return csp(sb.avgPool2D(ignore_padding, div), X); 
        }
        public Tensor[] avgpool2D(boolean ignore_padding, int kernel, int stride, int padding, Tensor... X) {
            return csp(sb.avgPool2D(ignore_padding, kernel, stride, padding), X);
        }
        public Tensor[] avgpool2D(boolean ignore_padding,
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp(sb.avgPool2D(ignore_padding, kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.NaiveMaxPool2D">
        public Tensor[] naive_maxPool2D(int div, Tensor... X) {
            return csp(sb.naive_maxPool2D(div), X);
        }
        public Tensor[] naive_maxPool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp(sb.naive_maxPool2D(kernel, stride, padding), X);
        }
        public Tensor[] naive_maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X)  {
            return csp(sb.naive_maxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.MaxPool2D">
        public Tensor[] maxPool2D(int div, Tensor... X) {
            return csp_maxPool(sb.maxPool2D(div), X);
        }
        public Tensor[] maxPool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp_maxPool(sb.maxPool2D(kernel, stride, padding), X);
        }
        public Tensor[] maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp_maxPool(sb.maxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.MaxPool2D_with_Index{Y, Index}">
        public Tensor[] maxPool2D_with_Index(int div, Tensor... X) {
            return csp_maxPool_with_Index(sb.maxPool2D(div), X);
        }
        public Tensor[] maxPool2D_with_Index(int kernel, int stride, int padding, Tensor... X) {
            return csp_maxPool_with_Index(sb.maxPool2D(kernel, stride, padding), X);
        }
        public Tensor[] maxPool2D_width_Index(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp_maxPool_with_Index(sb.maxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.AdpativeAvgPool2D">
        public Tensor[] adaptive_avgPool2D(int out_size, Tensor... X) {
            return csp(sb.adaptive_avgPool2D(out_size), X);
        }
        public Tensor[] adaptive_avgPool2D(int out_height, int out_width, Tensor... X) {
            return csp(sb.adaptive_avgPool2D(out_height, out_width), X);
        }
        
        public Tensor[] adaptive_avgPool2D(boolean ignore_padding, int out_size, Tensor... X) {
            return csp(sb.adaptive_avgPool2D(ignore_padding, out_size), X);
        }

        public Tensor[] adaptive_avgPool2D(boolean ignore_padding, int out_height, int out_width, Tensor... X) {
            return csp(sb.adaptive_avgPool2D(ignore_padding, out_height, out_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.AdaptiveNaiveMaxPool2D">
        public Tensor[] adaptive_naive_maxPool2D(int out_size, Tensor... X) {
            return csp(sb.adaptive_naive_maxPool2D(out_size), X);
        }
        
        public Tensor[] adaptive_naive_maxPool2D_naive(int out_height, int out_width, Tensor... X) {
            return csp(sb.adaptive_naive_maxPool2D(out_height, out_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.AdaptiveMaxPool2D">
        public Tensor[] adaptive_maxPool2D(int out_size, Tensor... X) {
            return csp_maxPool(sb.adaptive_maxPool2D(out_size), X);
        }

        public Tensor[] adaptive_maxPool2D(int out_height, int out_width, Tensor... X) {
            return csp_maxPool(sb.adaptive_maxPool2D(out_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool2d.AdaptiveMaxPool2D_width_Index{Y, Index}">
        public Tensor[] adaptive_maxPool2D_with_Index(int out_size, Tensor... X) {
            return csp_maxPool_with_Index(sb.adaptive_maxPool2D(out_size), X);
        }

        public Tensor[] adaptive_maxPool2D_with_Index(int out_height, int out_width, Tensor... X) {
            return csp_maxPool_with_Index(sb.adaptive_maxPool2D(out_width), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: avgUpool2d">
        //<editor-fold defaultstate="collapsed" desc="outsize = [-1, -1]">
        public Tensor[] avgUnpool2D(int scaleUp, Tensor...X) { 
            return csp(sb.avgUnpool2D(scaleUp), X); 
        }
        public Tensor[] avgUnpool2D(int kernel_size, int stride, int padding, Tensor... X) {
            return csp(sb.avgUnpool2D(kernel_size, stride, padding), X);
        }
        public Tensor[] avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor...X)
        {
            return csp(sb.avgUnpool2D(kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width), X);
        }
        //</editor-fold>
        
        public Tensor[] avgUnpool2D(int scaleUp, int out_size, Tensor...X) {
            return csp(sb.avgUnpool2D(scaleUp, out_size), X);
        }
        
        public Tensor[] avgUnpool2D(int kernel_size, int stride, int padding, int out_size, Tensor...X) {
            return csp(sb.avgUnpool2D(kernel_size, stride, padding, out_size), X);
        }
        
        public Tensor[] avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width,
                int output_height, int output_width,
                Tensor...X)
        {
            return csp(sb.avgUnpool2D(kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    output_height, output_width), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functioal: dual.math1">
        public Tensor[] matMul(Tensor...X) { return cdu(sb.matMul(), X); }
        public Tensor[] matMulT1(Tensor...X) {  return cdu(sb.matMulT1(), X); }
        public Tensor[] matMulT2(Tensor...X) {  return cdu(sb.matMulT2(), X); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functioal: dual.math2">
        public Tensor[] batchMatMul(Tensor...X) { 
            return cdu(sb.batchMatMul(), X); 
        }
        public Tensor[] batchMatMul(boolean likeX1, Tensor...X) {
            return cdu(sb.batchMatMul(likeX1), X);
        }
        
        
        public Tensor[] batchMatMulT1(Tensor...X) { 
            return cdu(sb.batchMatMulT1(), X); 
        }
        public Tensor[] batchMatMulT1(boolean likeX1, Tensor...X) {
            return cdu(sb.batchMatMulT1(likeX1), X);
        }
       
        
        public Tensor[] batchMatMulT2(Tensor... X) { 
            return cdu(sb.batchMatMulT2(), X); 
        }
        public Tensor[] batchMatMulT2(boolean likeX1, Tensor... X) {
            return cdu(sb.batchMatMulT2(likeX1), X);
        }
        
        //<editor-fold defaultstate="collapsed" desc="linear2: add, sub">
        public Tensor[] add(Tensor... X) { 
            return cdu(sb.add(), X); 
        }
        public Tensor[] sub(Tensor... X) { 
            return cdu(sb.sub(), X); 
        }
        public Tensor[] add(float alpha, float beta, Tensor... X) { 
            return cdu(sb.add(alpha, beta), X); 
        }
        public Tensor[] linear2(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(sb.linear2(alpha, beta, gamma), X);
        }
        
        public Tensor[] add(boolean likeX1, Tensor... X) { 
            return cdu(sb.add(likeX1), X); 
        }
        public Tensor[] sub(boolean likeX1, Tensor... X) { 
            return cdu(sb.sub(likeX1), X); 
        }
        public Tensor[] add(boolean likeX1, float alpha, float beta, Tensor... X) { 
            return cdu(sb.add(likeX1, alpha, beta), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] linear2(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(sb.linear2(likeX1, alpha, beta, gamma), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="quadratic2: mul, squareAdd, squareSub">
        public Tensor[] mul(Tensor... X) { 
            return cdu(sb.mul(), X); 
        }
        public Tensor[] mul(float alpha, Tensor... X) { 
            return cdu(sb.mul(alpha), X); 
        }
        public Tensor[] squareAdd(Tensor... X) { 
            return cdu(sb.squareAdd(), X); 
        }
        public Tensor[] squareSub(Tensor... X) { 
            return cdu(sb.squareSub(), X); 
        }
        public Tensor[] squareAdd(float alpha, float beta, Tensor... X) { 
            return cdu(sb.squareAdd(alpha, beta), X); 
        }
        public Tensor[] quadratic2(float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(sb.quadratic2(k11, k12, k22, k1, k2, C), X);
        }
        
        public Tensor[] mul(boolean likeX1, Tensor... X) { 
            return cdu(sb.mul(likeX1), X); 
        }
        public Tensor[] mul(boolean likeX1, float alpha, Tensor... X) { 
            return cdu(sb.mul(likeX1, alpha), X); 
        }
        public Tensor[] squareAdd(boolean likeX1, Tensor... X) { 
            return cdu(sb.squareAdd(likeX1), X); 
        }
        public Tensor[] squareSub(boolean likeX1, Tensor... X) { 
            return cdu(sb.squareSub(likeX1), X); 
        }
        public Tensor[] squareAdd(boolean likeX1, float alpha, float beta, Tensor... X) { 
            return cdu(sb.squareAdd(likeX1, alpha, beta), X); 
        }
        @Passed("CudaFloat32EngieBase")
        public Tensor[] quadratic2(boolean likeX1, float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(sb.quadratic2(likeX1, k11, k12, k22, k1, k2, C), X);
        }
        //</editor-fold>
        
        public Tensor[] div(Tensor... X) { 
            return cdu(sb.div(), X); 
        }
        public Tensor[] div(float alpha1, float beta1, float alpha2, float beta2, float gamma, Tensor... X) {
            return cdu(sb.div(alpha1, beta1, alpha2, beta2, gamma), X);
        }
        public Tensor[] div(boolean likeX1, Tensor... X) { 
            return cdu(sb.div(likeX1), X); 
        }
        public Tensor[] div(boolean likeX1, float alpha1, float beta1, float alpha2, float beta2, float gamma, Tensor... X){
            return cdu(sb.div(likeX1, alpha1, beta1, alpha2, beta2, gamma), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: reducer">
        public Tensor[] sum(Tensor... X) { 
            return cre(sb.sum(), X); 
        }
        public Tensor[] sum(float alpha, Tensor... X) {
            return cre(sb.sum(alpha), X); 
        }
        public Tensor[] linearSum(float alpha, float beta, Tensor... X) {
            return cre(sb.linearSum(alpha, beta), X); 
        }
        

        public Tensor[] mean(Tensor... X) { 
            return cre(sb.mean(), X); 
        }
        public Tensor[] mean(float alpha, Tensor... X) { 
            return cre(sb.mean(alpha), X); 
        }
        public Tensor[] linearMean(float alpha, float beta, Tensor... X) { 
            return cre(sb.linearMean(alpha, beta), X); 
        }
        
        
        public Tensor[] squareSum(Tensor... X) {
            return cre(sb.squareSum(), X);
        }
        public Tensor[] squareSum(float alpha, Tensor... X) {
            return cre(sb.squareSum(alpha), X);
        }
        public Tensor[] quadraticSum(float alpha, float beta, float gamma, Tensor... X) {
            return cre(sb.quadraticSum(alpha, beta, gamma), X);
        }
        
        
        public Tensor[] squareMean(Tensor... X) {
            return cre(sb.squareMean(), X);
        }
        public Tensor[] squareMean(float alpha, Tensor... X) {
            return cre(sb.squareMean(alpha), X);
        }
        public Tensor[] quadraticMean(float alpha, float beta, float gamma, Tensor... X) {
            return cre(sb.quadraticMean(alpha, beta, gamma), X);
        }
         
        
        public Tensor[] concat(Tensor... X) { 
            return cre(sb.concat(), X); 
        }
        public Tensor[] concat(int dimIdx, Tensor... X) {
            return cre(sb.concat(dimIdx), X); 
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: furcartion">
        public Tensor[] split(Tensor[] X, int...section) { 
            return cfc(sb.split(-1, section), X); 
        }
        public Tensor[] split(int dimIdx, Tensor[] X, int... section) {
            return cfc(sb.split(dimIdx, section), X);
        }
        
        public Tensor[] chunk(int n, Tensor... X) {
            return cfc(sb.chunk(-1, n), X);
        }
        public Tensor[] chunk(int dimIdx, int n, Tensor... X) {
            return cfc(sb.chunk(dimIdx, n), X);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Loss">
    public static class Loss 
    {
        protected Loss() {}
        public static final Loss loss = new Loss();
        
        //<editor-fold defaultstate="collapsed" desc="Norm Loss">
        public L1 L1() { return new L1(); }
        public L2 L2() { return new L2(); }
        public SmoothL1 SmoothL1() { return new SmoothL1(); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="Binary Cross Entropy">
        public BinaryCrossEntropy binaryCrossEntropy() { 
            return new BinaryCrossEntropy(1.0f, 1.0f);
        }
        public BinaryCrossEntropy binaryCrossEntropy(float alpha, float beta) { 
            return new BinaryCrossEntropy(alpha, beta);
        }
        
        public SigmoidBinaryCrossEntropy sigmoid_binaryCrossEntropy() {
            return new SigmoidBinaryCrossEntropy(1.0f, 1.0f);
        }
        public SigmoidBinaryCrossEntropy sigmoid_binaryCrossEntropy(float alpha, float beta) {
            return new SigmoidBinaryCrossEntropy(alpha, beta);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="CrossEntropy">
        public CrossEntropy crossEntropy() { 
            return new CrossEntropy(); 
        }
       
        public SoftmaxCrossEntropy softmax_crossEntropy() {
            return new SoftmaxCrossEntropy(-1);
        }
        public SoftmaxCrossEntropy softmax_crossEntropy(int features) {
            return new SoftmaxCrossEntropy(features);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="class: LossSummaryCreator">
        public static final class WeightedSummaryCreator 
        {
            private final List<Float> weights = new ArrayList<>(4);
            private final List<LossFunction> loss_funcs = new ArrayList<>(4);
            
            private WeightedSummaryCreator() {}
            
            public List<Float> weights() { return weights; }
            public List<LossFunction> loss_funcs() { return loss_funcs; }
            
            public WeightedSummaryCreator append(float weight, LossFunction loss) {
                if(loss == null) throw new NullPointerException("loss is null");
                weights.add(weight);
                loss_funcs.add(loss);
                return this;
            }
            
            public WeightedSummaryCreator clear() {
                weights.clear();
                loss_funcs.clear();
                return this;
            }
            
            public WeightedSummary create() 
            {
                float[] w = new float[weights.size()]; {
                    int index = 0; 
                    for(float weight : weights) w[index++] = weight;
                }
                
                LossFunction[] lf = new LossFunction[loss_funcs.size()]; {
                    int index = 0;
                    for(LossFunction loss : loss_funcs) lf[index++] = loss;
                }
                
                return new WeightedSummary(w, lf);
            }
        }
        //</editor-fold>
        public WeightedSummaryCreator weighted_sum() { 
            return new WeightedSummaryCreator();
        }
        public WeightedSummary weighted_sum(float[] weight, LossFunction[] loss) {
            return new WeightedSummary(weight, loss);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Opim">
    public static final class Optim 
    {
        protected Optim() {}
        public static final Optim optim = new Optim();

        //<editor-fold defaultstate="collpased" desc="create: Optimizer">
        public SGD SGD(Collection<Parameter> params, float lr) {
            return new SGD(params, lr); 
        }
        public SGD SGD(Map<String, Parameter> params, float lr) {
            return new SGD(params, lr); 
        }
        
        public static float Momentum_beta = 0.9f;
        
        public Momentum Momentum(Collection<Parameter> params, float lr) { 
            return new Momentum(params, lr, Momentum_beta); 
        }
        public Momentum Momentum(Collection<Parameter> params, float lr, float beta) {
            return new Momentum(params, lr, beta);
        }
        
        public Momentum Momentum(Map<String, Parameter> param_map, float lr) { 
            return new Momentum(param_map, lr, Momentum_beta); 
        }
        public Momentum Momentum(Map<String, Parameter> param_map, float lr, float beta) {
            return new Momentum(param_map, lr, beta);
        }
        
        
        public static float SGDMN_momentum = 0.9f;
        public static float SGDMN_dampen = 0.0f;
        public static float nestrov = 0.0f;
        
        public SGDMN SGDMN(Collection<Parameter> params, float lr) {
            return new SGDMN(params, lr, SGDMN_momentum, SGDMN_dampen, nestrov);
        }
        public SGDMN SGDMN(Collection<Parameter> params, float lr,
                float momentum, float dampen, float nestrov) {
            return new SGDMN(params, lr, momentum, dampen, nestrov);
        }
        
        public SGDMN SGDMN(Map<String, Parameter> param_map, float lr) {
            return new SGDMN(param_map, lr, SGDMN_momentum, SGDMN_dampen, nestrov);
        }
        public SGDMN SGDMN(Map<String, Parameter> param_map, float lr,
                float momentum, float dampen, float nestrov) {
            return new SGDMN(param_map, lr, momentum, dampen, nestrov);
        }
        
        
        public static float RMSprop_beta = 0.999f;
        public static float RMSprop_eps = 1e-8f;
        
        public RMSprop RMSprop(Collection<Parameter> params, float lr) { 
            return new RMSprop(params, lr, RMSprop_beta, RMSprop_eps );
        }
        public RMSprop RMSprop(Collection<Parameter> params, float lr, float beta, float eps) {
            return new RMSprop(params, lr, beta, eps);
        }
        
        public RMSprop RMSprop(Map<String, Parameter> param_map, float lr) { 
            return new RMSprop(param_map, lr, RMSprop_beta, RMSprop_eps );
        }
        public RMSprop RMSprop(Map<String, Parameter> param_map, float lr, float beta, float eps) {
            return new RMSprop(param_map, lr, beta, eps);
        }
        
        
        public static float Adam_beta1 = 0.9f;
        public static float Adam_beta2 = 0.999f;
        public static float Adam_eps = 1e-8f;
        
        public Adam Adam(Collection<Parameter> params, float lr) { 
            return new Adam(params, lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Collection<Parameter> params, float lr, 
                float beta1, float beta2, float eps) {
            return new Adam(params, lr, beta1, beta2, eps);
        }       
        
        public Adam Adam(Map<String, Parameter> param_map, float lr) { 
            return new Adam(param_map, lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Map<String, Parameter> param_map, float lr, 
                float beta1, float beta2, float eps) {
            return new Adam(param_map, lr, beta1, beta2, eps);
        }       
        
        
        public static float Adamax_beta1 = 0.9f;
        public static float Adamax_beta2 = 0.999f;
        public static float Adamax_eps = 1e-8f;
        
        public Adamax Adamax(Collection<Parameter> params, float lr) { 
            return new Adamax(params, lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Collection<Parameter> params, float lr,
                float beta1, float beta2, float eps) {
            return new Adamax(params, lr, beta1, beta2, eps);
        }
        
        public Adamax Adamax(Map<String, Parameter> param_map, float lr) { 
            return new Adamax(param_map, lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Map<String, Parameter> param_map, float lr,
                float beta1, float beta2, float eps) {
            return new Adamax(param_map, lr, beta1, beta2, eps);
        }
       
        
        public static float AdamW_beta1 = 0.9f;
        public static float AdamW_beta2 = 0.999f;
        public static float AdamW_eps = 1e-8f;
        
        public AdamW AdamW(Collection<Parameter> params, float lr, float L2coef) {
            return new AdamW(params, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, 0, L2coef); 
        }
        public AdamW AdamW(Collection<Parameter> params, float lr, float L1coef, float L2coef) {
            return new AdamW(params, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, L1coef, L2coef); 
        }
        public AdamW AdamW(Collection<Parameter> params, float lr, 
                float beta1, float beta2, float eps,
                float L1coef, float L2coef) {
            return new AdamW(params, lr, beta1, beta2, eps, L1coef, L2coef);
        }
        
        public AdamW AdamW(Map<String, Parameter> param_map, float lr, float L2coef) {
            return new AdamW(param_map, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, 0, L2coef); 
        }
        public AdamW AdamW(Map<String, Parameter> param_map, float lr, float L1coef, float L2coef) {
            return new AdamW(param_map, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, L1coef, L2coef); 
        }
        public AdamW AdamW(Map<String, Parameter> param_map, float lr, 
                float beta1, float beta2, float eps,
                float L1coef, float L2coef) {
            return new AdamW(param_map, lr, beta1, beta2, eps, L1coef, L2coef);
        }
        
        
        public static float Adamod_beta1 = 0.9f;
        public static float Adamod_beta2 = 0.999f;
        public static float Adamod_beta3 = 0.999f;
        public static float Adamod_eps = 1e-8f;
        
        public Adamod Adamod(Collection<Parameter> params, float lr) { 
            return new Adamod(params, lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Collection<Parameter> params, float lr, 
                float beta1, float beta2, float eps, float beta3) {
            return new Adamod(params, lr, beta1, beta2, eps, beta3);
        }
        
        public Adamod Adamod(Map<String, Parameter> param_map, float lr) { 
            return new Adamod(param_map, lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Map<String, Parameter> param_map, float lr, 
                float beta1, float beta2, float eps, float beta3) {
            return new Adamod(param_map, lr, beta1, beta2, eps, beta3);
        }
        //</editor-fold>

        //<editor-fold defaultstate="collpased" desc="creat: LrSchedular">
        public CosAnnealingLr cosAnnealingLr(float Tmax) {
            return new CosAnnealingLr(Tmax, 0f);
        }
        public CosAnnealingLr cosAnnealingLr(float Tmax, float minLr) {
            return new CosAnnealingLr(Tmax, minLr);
        }

        public ExponentialLr exponentialLr(float gamma, float minLr) {
            return new ExponentialLr(gamma, minLr);
        }

        public LambdaLr lambdaLr(Function<Float, Float> updater) {
            return new LambdaLr(updater);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Datas">
    public static class Datas 
    {
        protected Datas() {}
        public static final Datas data = new Datas();
        
        //<editor-fold defaultstate="collapsed" desc="create: AutoLoadContainer">
        public static int autoLoad_capcity = 1024;
        public static int autoLoad_thread_num = 4;
        public static double autoLoad_update_threshold = 0.5;
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls) {
            return new AutoLoadContainer<>(input_cls, label_cls, autoLoad_capcity)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls,
                Loader<K, V> loader) {
            return new AutoLoadContainer<>(input_cls, label_cls, autoLoad_capcity)
                    .loader(loader)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, 
                Loader<K, V> loader, Triger triger) {
            return new AutoLoadContainer<>(input_cls, label_cls, autoLoad_capcity)
                    .loader(loader)
                    .triger(triger)
                    .thread_num(autoLoad_thread_num);
        }
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity)  {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                Loader<K, V> loader) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                Loader<K, V> loader, Triger triger) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(triger)
                    .thread_num(autoLoad_thread_num);
        }
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, int thread_num)  {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                int thread_num, Loader<K, V> loader) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                int thread_num, Loader<K, V> loader, Triger triger) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(triger)
                    .thread_num(thread_num);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: ListContainer">
        public static int list_initCapacity = 2048;
        
        public <K, V> ListContainer<K, V> list(Class<K> input_cls, Class<V> label_cls) {
            return new ListContainer<>(input_cls, label_cls, list_initCapacity);
        }
        
        public <K, V> ListContainer<K, V> list(Class<K> input_cls, Class<V> label_cls, int initCapacity) {
            return new ListContainer<>(input_cls, label_cls, initCapacity);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: DataSet">
        public<K, V> DataSet<K, V> dataset(DataContainer<K, V> conta) {
            return new DataSet<>(conta);
        }
        public<K, V> DataSet<K, V> dataset(DataContainer<K, V> conta, 
                Transform<K[]> input_transform, Transform<V[]> label_transform) {
            return new DataSet<>(conta)
                    .input_transform(input_transform)
                    .label_transform(label_transform);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: Transform">
        //<editor-fold defaultstate="collapsed" desc="class: ImageTransform"> 
        public static class ImageTransform implements Transform<byte[][]> 
        {
            private final int[] dim;
            
            private ImageTransform(int...dim) { this.dim = dim; }

            @Override
            public Tensor transform(Engine eg, byte[][] value) {
                return eg.pix2tensor(value, Vector.append(value.length, dim));
            }
        }
        //</editor-fold>
        public ImageTransform image_transform(int... dim) { 
            return new ImageTransform(dim); 
        }

        //<editor-fold defaultstate="collapsed" desc="class: OnehotTransform"> 
        public static class OnehotTransform implements Transform<Integer[]> 
        {
            private final int num_class;
            
            private OnehotTransform(int num_class) {  this.num_class = num_class; }

            @Override
            public Tensor transform(Engine eg, Integer[] value) {
                int[] labels = new int[value.length];
                for (int i = 0; i < labels.length; i++) labels[i] = value[i];
                return eg.onehot(labels, num_class);
            }
        }
        //</editor-fold>
        public OnehotTransform onehot_transform(int num_class) {
            return new OnehotTransform(num_class); 
        }
        
        //<editor-fold defaultstate="collapsed" desc="class: FloatsTransform"> 
        public static class FloatsTransform implements Transform<float[][]> 
        {
            private final int[] dim;
            
            public FloatsTransform(int... dim) { this.dim = dim; }

            @Override
            public Tensor transform(Engine eg, float[][] value) {
                return eg.tensor(value, Vector.append(value.length, dim));
            }
        }
        //</editor-fold>
        public FloatsTransform floats_transform(int... dim) {
            return new FloatsTransform(dim);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: Buffer">
        public<T> Buffer<T> buffer(Callable<T> getter) { return new Buffer<>(getter); }
        
        public<K, V> Buffer<TensorPair> pair_buffer(Callable<TensorPair> getter) {
            return new Buffer<>(getter);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: States">
    public static class Stats
    {
        protected Stats() {}
        
        public static final Stats stat = new Stats();
        
        public static boolean default_partial = false;
        public static Charset default_charset = Charset.forName("UTF-8");
        
        //<editor-fold defaultstate="collapsed" desc="basic primitives">
        public boolean update(Stateful st, State dic) { return update(st, dic, default_partial); }
        public boolean update(Stateful st, State dic, boolean partial) {
            try { st.update_state(dic, partial); } 
            catch(Exception e) { throw new RuntimeException(e); }
            return true;
        }
        
        public Stateful transform(Stateful st, StatefulTransformer trf) {
            try { return trf.transform(st); }
            catch(Exception e) { throw new RuntimeException(e); }
        }
        
        public State read(StateReader writer) {
           try { return writer.read(); }
           catch(Exception e) { throw new RuntimeException(e); }
        }
        
        public boolean write(Stateful st, StateWriter writer) {
            try { writer.write(st.state()); }
            catch(Exception e) { throw new RuntimeException(e); }
            return true;
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="ZipState: extended">
        public ZipStateWriter zip_writer(String path) { return zip_writer(path, default_charset); }
        public ZipStateWriter zip_writer(String path, Charset cs) {
           try { return new ZipStateWriter(path, cs); }
           catch(IOException e) { throw new RuntimeException(e); }
        }
        
        public ZipStateReader zip_reader(String path) { return zip_reader(path, default_charset); } 
        public ZipStateReader zip_reader(String path, Charset cs) {
            try { return new ZipStateReader(path, cs); }
            catch(IOException e) { throw new RuntimeException(e); }
        }
        
        public boolean save_zip(Stateful st, String path) { return save_zip(st, path, default_charset); }
        public boolean save_zip(Stateful st, String path, Charset cs) {
            return write(st, zip_writer(path, cs));
        }
        
        public boolean load_zip(Stateful st, String path) {
            return load_zip(st, path, default_charset, default_partial);
        }
        public boolean load_zip(Stateful st, String path, boolean partial) { 
            return load_zip(st, path, default_charset, partial);
        }
        public boolean load_zip(Stateful st, String path, Charset cs, boolean partial) {//load and update
            State dic = zip_reader(path, cs).read();
            return update(st, dic, partial);
        }
        //</editor-fold>
    }
    //</editor-fold>
}

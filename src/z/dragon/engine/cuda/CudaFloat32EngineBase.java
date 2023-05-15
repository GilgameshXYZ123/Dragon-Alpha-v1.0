/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda;

import java.io.File;
import java.util.Objects;
import z.dragon.engine.EngineBase;
import z.dragon.engine.EngineCore;
import z.dragon.engine.Result;
import z.dragon.engine.Result.IndexedResult;
import z.dragon.engine.Syncer;
import z.dragon.engine.cuda.impl.math.Cuda_matMul;
import z.dragon.engine.cuda.impl.CudaDevice;
import z.dragon.engine.cuda.impl.Cuda;
import z.dragon.engine.cuda.impl.CudaStreamPool;
import z.dragon.engine.cuda.impl.Cuda_expk2;
import z.dragon.engine.cuda.impl.PinnedMempool;
import z.dragon.engine.cuda.impl.math.Cuda_batchMatMul;
import z.dragon.engine.cuda.impl.math.Cuda_conv3D;
import z.dragon.engine.cuda.impl.math.Cuda_dconv3D_deltaW;
import z.dragon.engine.cuda.impl.math.Cuda_dconv3D_deltaX;
import z.dragon.engine.cuda.impl.math.Cuda_function;
import z.dragon.engine.cuda.impl.math.Cuda_pool2D;
import z.dragon.engine.cuda.impl.math.Cuda_random;
import z.dragon.engine.cuda.impl.math.Cuda_reduce;
import z.dragon.engine.cuda.impl.math.Cuda_upool2D;
import z.util.math.ExMath;
import z.util.math.Num;

/**
 *
 * @author Gilgamesh
 */
public class CudaFloat32EngineBase extends EngineBase
{
    //<editor-fold defaultstate="collapsed" desc="native-lib">
    private static final String PATH = "native-lib\\cuda_float32";
    
    private static boolean NATIVE_LOAD = false;
    private static final boolean TEST_MODE = false;
    
    public static boolean __TEST_MODE__() { return TEST_MODE; }
    public static boolean __NATIVE_LOAD__() { return NATIVE_LOAD; }
    public static void __SET_NATIVE_LOAD__(boolean flag) { NATIVE_LOAD = flag; }
    
    public static synchronized void load_native_lib(String alpha_home) {
        File nativeLib = new File(alpha_home, PATH);
        for(File lib : nativeLib.listFiles((File file) -> { return file.getName().endsWith(".dll");}))
            System.load(lib.getAbsolutePath());
        System.setProperty("ALPHA_HOME", alpha_home);
    }
    //</editor-fold>
     
    protected CudaDevice dev;
    protected CudaStreamPool streamPool;
    protected PinnedMempool bufPool;// = new PinnedMemoryPool(MEM_1MB * 128);
    
    public CudaFloat32EngineBase() {this(new CudaDevice(), 64);}
    public CudaFloat32EngineBase(int deviceId, int streamPool_maxsize) {
       this(new CudaDevice(deviceId), streamPool_maxsize);
    }
    public CudaFloat32EngineBase(CudaDevice device, int streamPool_maxsize) {
        super("cuda_float32", 2, "cuda_int32", "cuda_int8");
        if(device == null) throw new NullPointerException("device");

        this.dev = device;
        this.streamPool = CudaStreamPool.instance(dev, streamPool_maxsize, matMulT1_MaxPartNum);
        
        //param initailze-------------------------------------------------------
        dconv3D_deltaW_SK_MaxPartNum = ExMath.clip(dev.getMultiProcessorCount(), 16, 128);
        matMulT1_MaxPartNum = ExMath.clip(dev.getMultiProcessorCount(), 16, 128);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public CudaStreamPool streamPool() {return streamPool;}
    public synchronized CudaFloat32EngineBase streamPool(CudaStreamPool streamPool) {
        this.streamPool = streamPool;
        return this;
    }
    
    public PinnedMempool buf_mempool() { return bufPool; }
    public synchronized CudaFloat32EngineBase buf_mempool(PinnedMempool mempool) {
        this.bufPool = mempool;
        return this;
    }
    
    public CudaDevice device() {return dev;}
    
    @Override
    public void append(StringBuilder sb) {
        super.append(sb);
        sb.delete(sb.length() - 3, sb.length() - 1);
        
        if(bufPool != null) { //add properties of bufPool
            bufPool.meta_data().forEach((k, v) -> {
                sb.append("\nbufPool.").append(k).append(" = ").append(v);
            });
        }
        
        sb.append(" }");
    }
    
    @Override
    public synchronized void clear() {
        streamPool.clear();
        bufPool.clear();
    }
    
    @Override
    public void finalize() throws Throwable {
        super.finalize();
        this.clear();
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 59 * hash + Objects.hashCode(this.dev);
        return hash;
    }
    
    @Override
    public boolean equals(Object o) {
        if(!(o instanceof CudaFloat32EngineBase)) return false;
        CudaFloat32EngineBase cu32 = (CudaFloat32EngineBase) o;
        return Objects.equals(cu32.dev, dev);
    }
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="extra: int32">
    @Override
    public void get1D_int32(long address, int[] value, int length) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get1D_int(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2_int(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void get2D_int32(long address, int[] value, int height, int width, int stride)
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get2D_int(stream, address, value, height, width, stride); 
            streamPool.returnStream(stream);
            return;
        }
      
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2_int(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set1D_int32(long address, int[] value, int length) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {//no buffer memthod
            Cuda.set1D_int(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2_int(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set2D_int32(long address, int[] value, int height, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.set2D_int(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }

        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set2D_v2_int(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="extra: int8">
    @Override
    public void get1D_int8(long address, byte[] value, int length) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get1D_char(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc( ((length + 3) >> 2) << 2 );//padding length to 4x
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2_char(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void get2D_int8(long address, byte[] value, int height, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get2D_char(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(height * stride);//stride % 4 == 0
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2_char(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set1D_int8(long address, byte[] value, int length) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.set1D_char(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc( ((length + 3) >> 2) << 2 );//padding length to 4x
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2_char(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }

    @Override
    public void set2D_int8(long address, byte[] value, int height, int width, int stride) 
    {
        //[common version]------------------------------------------------------
        if(bufPool == null) {
           long stream = streamPool.getStream();
            Cuda.set2D_char(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        //[width = 3, stride = 4, used for pictures(JPEG) width 3 channels]-----
        if((width == 3) && (stride == 4)  && (height%4 == 0)) {
            long stream = streamPool.getStream();
            
            long[] block1 = bufPool.malloc(height * 3);
            long[] block2 = bufPool.malloc(height * 4);//padding length to 4x
            long buf1_size = block1[0], buf1_address = block1[1];
            long buf2_size = block2[0], buf2_address = block2[1];
            
            Cuda.set2D_v2_char_W3S4(stream, address, value, buf1_address, buf2_address, height);
            
            bufPool.free(buf1_size, buf1_address);
            bufPool.free(buf2_size, buf2_address);
            streamPool.returnStream(stream);
            return;
        }
        
        //[fast version]----------------------------------------------------------
        long stream = streamPool.getStream();
        
        long[] block = bufPool.malloc(height * stride);//padding length to 4x
        long buf_size = block[0], buf_address = block[1];
        
        Cuda.set2D_v2_char(stream, address, value, buf_address, height, width, stride);
        
        bufPool.free(buf_size, buf_address);
        streamPool.returnStream(stream);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Syncer & Result">
    //<editor-fold defaultstate="collapsed" desc="class: StreamSyncer">
    public static final class StreamSyncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        
        public long stream() { return stream; }
        
        StreamSyncer(CudaStreamPool streamPool, long stream) {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
        }
        
        @Override 
        public synchronized void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStream(stream);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Syncer_1">
    public static final class Stream2Syncer_1 implements Syncer //only sync stream1
    {
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        
        public long stream1() { return stream1; }
        public long stream2() { return stream2; }
        
        Stream2Syncer_1(CudaStreamPool streamPool, long stream1, long stream2) {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
        }
        
        @Override 
        public synchronized void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStream(stream1);
            streamPool.returnStream(stream2);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlockSyncer">
    public static final class StreamBlockSyncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        
        private final EngineCore core;
        private final long[] block;
        
        public long stream() { return stream; }
        public long[] block() { return block; }
        
        StreamBlockSyncer(CudaStreamPool streamPool, long stream,
                EngineCore core, long[] block)
        {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            
            this.core = core;
            this.block = block;
        }
        
        @Override 
        public synchronized void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlock2Syncer">
    public static final class StreamBlock2Syncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long stream;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public long stream() { return stream; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
        
        StreamBlock2Syncer(CudaStreamPool streamPool, long stream,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream = stream;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            synchronized(this) {
                if(called) return;
                streamPool.returnStream(stream);
                core.free(block1[0], block1[1]);
                core.free(block2[0], block2[1]);
                called = true;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Block2Syncer_1">
    public static final class Stream2Block2Syncer_1 implements Syncer //only sync stream1
    {
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public long stream1() { return stream1; }
        public long stream2() { return stream2; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
          
        Stream2Block2Syncer_1(CudaStreamPool streamPool, long stream1, long stream2,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            synchronized(this) {
                if(called) return;
                streamPool.returnStream(stream1);
                streamPool.returnStream(stream2);
                core.free(block1[0], block1[1]);
                core.free(block2[0], block2[1]);
                called = true;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: Stream2Block3Syncer_1">
    public static final class Stream2Block3Syncer_1 implements Syncer //only sync stream1
    {
        private final CudaStreamPool streamPool;
        private final long stream1;
        private final long stream2;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        private final long[] block3;
        
        public long stream1() { return stream1; }
        public long stream2() { return stream2; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
        public long[] block3() { return block3; }
        
        Stream2Block3Syncer_1(CudaStreamPool streamPool, long stream1, long stream2,
                EngineCore core, long[] block1, long[] block2, long[] block3)
        {
            this.streamPool = streamPool;
            this.stream1 = stream1;
            this.stream2 = stream2;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream1);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
            this.block3 = block3;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            synchronized(this) {
                if(called) return;
                streamPool.returnStream(stream1);
                streamPool.returnStream(stream2);
                core.free(block1[0], block1[1]);
                core.free(block2[0], block2[1]);
                core.free(block3[0], block3[1]);
                called = true;
            }
        }
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="class: StreamArraySyncer">
    public static final class StreamArraySyncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        
        public long[] streams() { return streams; }
        
        StreamArraySyncer(CudaStreamPool streamPool, long[] streams) 
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams, streams.length);
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlockSyncer">
    public static class StreamArrayBlockSyncer implements Syncer
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        
        private final long event;
        
        private final EngineCore core;
        private final long[] block;
        
        public long[] streams() { return streams; }
        public long[] block() { return block; }
        
        StreamArrayBlockSyncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams, streams.length);
            
            this.core = core;
            this.block = block;
        }
             
        @Override
        public void sync() {
            Cuda.eventSynchronize(event);
            core.free(block[0], block[1]);//release block1.mem_address
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock2Syncer">
    public static class StreamArrayBlock2Syncer implements Syncer
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        
        private final long event;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public long[] streams() { return streams; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
        
        StreamArrayBlock2Syncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams, streams.length);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
             
        @Override
        public void sync() {
            Cuda.eventSynchronize(event);
            core.free(block1[0], block1[1]);//release block1.mem_address
            core.free(block2[0], block2[1]);//release block2.mem_address
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock2Syncer_1">
    public static final class StreamArrayBlock2Syncer_1 implements Syncer //only sync streams[0]
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public long[] streams() { return streams; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
          
        StreamArrayBlock2Syncer_1(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            synchronized(this) {
                if(called) return;
                streamPool.returnStreamArray(streams);
                core.free(block1[0], block1[1]);
                core.free(block2[0], block2[1]);
                called = true;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: StreamArrayBlock4Syncer_1">
    public static final class StreamArrayBlock4Syncer_1 implements Syncer //only sync streams[0]
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        private boolean called = false;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        private final long[] block3;
        private final long[] block4;
        
        public long[] streams() { return streams; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
        public long[] block3() { return block3; }
        public long[] block4() { return block4; }
          
        StreamArrayBlock4Syncer_1(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block1, long[] block2, long[] block3, long[] block4)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
            this.block3 = block3;
            this.block4 = block4;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            synchronized(this) {
                if(called) return;
                streamPool.returnStreamArray(streams);
                core.free(block1[0], block1[1]);
                core.free(block2[0], block2[1]);
                core.free(block3[0], block3[1]);
                core.free(block4[0], block4[1]);
                called = true;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: BiasedForwardSyncer">
    public static final class BiasedForwardSyncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        
        BiasedForwardSyncer(CudaStreamPool streamPool, long[] streams) 
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//streams[0], add bias after the transformation
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: BiasedBlockForwardSyncer">
    public static final class BiasedForwardBlockSyncer implements Syncer 
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        private final long event;
        
        private final EngineCore core;
        private final long[] block;
        
        BiasedForwardBlockSyncer(CudaStreamPool streamPool, long[] streams,
                EngineCore core, long[] block) 
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//streams[0], add bias after the transformation
            
            this.core = core;
            this.block = block;
        }
        
        @Override 
        public void sync() {
            Cuda.eventSynchronize(event);
            streamPool.returnStreamArray(streams);
            core.free(block[0], block[1]);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: SplitKSyncer">
    public static class SplitKSyncer implements Syncer
    {
        private final CudaStreamPool streamPool;
        private final long[] streams;
        
        private final long event;
        
        private final EngineCore core;
        private final long[] block;
        
        SplitKSyncer(EngineCore core, long[] block,
                CudaStreamPool streamPool, long[] streams)
        {
            this.streamPool = streamPool;
            this.streams = streams;
            this.event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streams[0]);//bufSum -> streams[0]
            
            this.core = core;
            this.block = block;
        }
             
        @Override
        public void sync() {
            Cuda.eventSynchronize(event);//wait for BGemm result
            core.free(block[0], block[1]);//release resources
            streamPool.returnStreamArray(streams);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: StreamBlockResult">
    public static final class StreamBlockResult extends Result<Float>
    {
        private final CudaStreamPool streamPool;
        private final long stream;
        
        private final EngineCore core;
        private final long[] block;
        
        public long stream() { return stream; } 
        public long[] block() { return block; }
        
        StreamBlockResult(CudaStreamPool streamPool, long stream_address,
                EngineCore core, long[] block)
        {
            this.streamPool = streamPool;
            this.stream = stream_address;
            
            this.core = core;
            this.block = block;
        }

        @Override
        protected Float waitResult() 
        {
            float[] result = new float[1];
            Cuda.get1D(stream, block[1], result, 1);//V_address = block1
            
            streamPool.returnStream(stream);
            core.free(block[0], block[1]);
            return result[0];
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class: FloatIndexedResult">
    public static final class FloatIndexedResult extends IndexedResult<Float>
    {
        private final CudaStreamPool streamPool;
        private final long stream;
        
        private final EngineCore core;
        private final long[] block1;
        private final long[] block2;
        
        public long stream() { return stream; }
        public long[] block1() { return block1; }
        public long[] block2() { return block2; }
        
        FloatIndexedResult(CudaStreamPool streamPool, long stream_address,
                EngineCore core, long[] block1, long[] block2)
        {
            this.streamPool = streamPool;
            this.stream = stream_address;
            
            this.core = core;
            this.block1 = block1;
            this.block2 = block2;
        }

        @Override
        protected IndexedValue<Float> waitResult() 
        {
            float[] result = new float[1];
            int[] index = new int[1];
            Cuda.get1D(stream, block1[1], result, 1);
            Cuda.get1D_int(stream, block2[1], index, 1);
            
            streamPool.returnStream(stream);
            core.free(block1[0], block1[1]);
            core.free(block2[0], block2[1]);
            
            return new IndexedValue<>(index[0], result[0]);
        }
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Opt">
    @Override 
    public long malloc(long memsize) {
        long address;
        synchronized(Cuda.class) { 
            Cuda.setDevice(dev.getId());
            address = Cuda.malloc(memsize);
        }
        return address;
    }
    
    @Override 
    public void free(long address)  {
        synchronized(Cuda.class) {
            Cuda.setDevice(dev.getId());
            Cuda.free(address);
        }
    }
    
    @Override 
    public Syncer memset(long address, int value, long memsize)  {
        long stream_address = streamPool.getStream();
        Cuda.memsetAsync(stream_address, address, value, memsize);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer memcpy(long dst_address, long src_address, long memsize) {
        long stream_address = streamPool.getStream();
        Cuda.memcpyAsyncDeviceToDevice(stream_address, dst_address, src_address, memsize);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    //<editor-fold defaultstate="collapsed" desc="get tensor value">
    @Override 
    public void get1D(long address, float[] value, int length)
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get1D(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get1D_v2(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    
    @Override 
    public void get2D(long address, float[] value, int height, int width, int stride)
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.get2D(stream, address, value, height, width, stride); 
            streamPool.returnStream(stream);
            return;
        }
      
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.get2D_v2(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = fill(constant)">
    @Override 
    public Syncer set1D(long address, float value, int length)  {
        long stream_address = streamPool.getStream();
        Cuda.set1D(stream_address, address, value, length);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer set2D(long address, float value, int height, int width, int stride) {
        long stream_address = streamPool.getStream();
        Cuda.set2D(stream_address, address, value, height, width, stride);
        return new StreamSyncer(streamPool, stream_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = value<float[]>">
    @Override 
    public void set1D(long address, float[] value, int length) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {//no buffer memthod
            Cuda.set1D(stream, address, value, length);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc(length << 2L); //pinned memory: faster
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set1D_v2(stream, address, value, buf_address, length);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    
    @Override 
    public void set2D(long address, float[] value, int height, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        if(bufPool == null) {
            Cuda.set2D(stream, address, value, height, width, stride);
            streamPool.returnStream(stream);
            return;
        }
        
        long[] bufBlock = bufPool.malloc((height * stride) << 2L);
        long buf_size = bufBlock[0], buf_address = bufBlock[1];
        Cuda.set2D_v2(stream, address, value, buf_address, height, width, stride);
        streamPool.returnStream(stream);
        bufPool.free(buf_size, buf_address);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="tensor = another tensor">
    @Override 
    public Syncer setFrom1Dto2D(long src_address, int src_length, 
            long dst_address, int dst_height, int dst_width, int dst_stride) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom1Dto2D(stream_address,
                src_address, src_length,
                dst_address, dst_height, dst_width, dst_stride);
         return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer setFrom2Dto1D(long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_length) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom2Dto1D(stream_address,
                src_address, src_height, 
                src_width, src_stride, dst_address, dst_length);
         return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override 
    public Syncer setFrom2Dto2D(long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_height, int dst_width, int dst_stride) 
    {
        long stream_address = streamPool.getStream();
        Cuda.setFrom2Dto2D(stream_address,
                src_address, src_height, src_width, src_stride, 
                dst_address, dst_height, dst_width, dst_stride);
        return new StreamSyncer(streamPool, stream_address);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Tensor Trick">
    @Override
    public Syncer gappedMemcpy2D(
            long dX_address, int Xstart, int strideX, 
            long dY_address, int Ystart, int strideY,
            int width, int length)
    {
        long stream_address = streamPool.getStream();
        Cuda_expk2.gappedMemcpy2D(stream_address,
                dX_address, Xstart, strideX,
                dY_address, Ystart, strideY,
                width, length);
        return new StreamSyncer(streamPool, stream_address);
    }
    
    @Override
    public Syncer transpose(
            long Y_address, int[] Ydim,
            long X_address, int[] Xdim, 
            int dimIndex1, int dimIndex2, 
            int strideX, int strideY, 
            int length) 
    {
        long stream = streamPool.getStream();
        if(Xdim.length == 4) {
            Cuda_expk2.transpose4D(stream, 
                    X_address, Y_address,
                    Xdim[1], Xdim[2], Xdim[3], 
                    Ydim[1], Ydim[2], Ydim[3], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else if(Xdim.length == 3) {
            Cuda_expk2.transpose3D(stream,
                    X_address, Y_address, 
                    Xdim[1], Xdim[2], 
                    Ydim[1], Ydim[2], 
                    dimIndex1, dimIndex2,
                    strideX, strideY, length);
        }
        else {
            Cuda_expk2.transpose2D(stream,
                    X_address, Y_address,
                    Xdim[1], Ydim[1],
                    strideX, strideY,
                    length);
        }
        
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer rot180(long Y_address,
            long X_address,
            int IH, int IW, int IC, 
            int length) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.rot180(stream,
                X_address, Y_address, 
                IH, IW, IC, 
                length);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer srcIndexedMemcpy(long Y_address, 
            long X_address, long Index_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.srcIndexedMemcpy(stream,
                X_address, Index_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer dstIndexedMemcpy(long Y_address,
            long X_address, long Index_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_expk2.dstIndexedMemcpy(stream,
                X_address, Index_address, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Matrix Multiply">
    //<editor-fold defaultstate="collapsed" desc="Normal matMul">
    //<editor-fold defaultstate="collapsed" desc="matMul">
    @Override
    public Syncer matMul(long C_address, 
            long A_address, long B_address, 
            int N, int M, int K) 
    {
        int length = Cuda_matMul.streamSize(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_matMul.matMul(streamArray, length, 
                    A_address, B_address, C_address,
                    N, M, K);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer matMul_biased(long C_address, 
            long A_address, long B_address,
            int N, int M, int K,
            long Bias_address,//lengthv = C.lengthv = N*M, stride = C.mem_stride = M
            int lengthv, int width) 
    {
        int length = Cuda_matMul.streamSize(N, M);
        //====stage1: Matrix Multiply===========================================
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_matMul.matMul(streamArray, length, 
                    A_address, B_address, C_address, N, M, K);
        
        //====stage2: add Bias==================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, length);
        Cuda.streamWaitEvent_default(streamArray[0], event);
        Cuda_function.linear_dual2D_row(streamArray[0], C_address,
                Bias_address, M, 
                1.0f, 1.0f, 0.0f, 
                C_address, 
                lengthv, width, M);
        
        return new BiasedForwardSyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT1">
    private int matMulT1_SplitKThreshold = 512;//Always generalized batch_size
    public int matMulT1_SplitKThreshold() { return matMulT1_SplitKThreshold; }
    public CudaFloat32EngineBase matMulT1_SplitKThreshold(int threshold) {
        matMulT1_SplitKThreshold = threshold;
        return this;
    }
    
    private int matMulT1_MaxPartNum = 16;
    public int matMulT1_MaxPartNum() { return matMulT1_MaxPartNum; }
    public CudaFloat32EngineBase matMulT1_MaxPartNum(int maxPartNum) {
        if(maxPartNum < 0) throw new IllegalArgumentException("maxPartNum must >= 0");
        if(maxPartNum < 4) throw new IllegalArgumentException("maxPartNum must >= 4");
        matMulT1_MaxPartNum = maxPartNum;
        return this;
    }
    
    @Override
    public Syncer matMulT1(long C_address,
            long A_address, long B_address, 
            int N, int M, int K) 
    {
        int length = Cuda_matMul.streamSize(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        if(K < matMulT1_SplitKThreshold) {
            Cuda_matMul.matMulT1(streamArray, length,
                        A_address, B_address, C_address,
                        N, M, K);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        else {
            int GridZ;
            if(K < 1024) GridZ = K >> 8;//div 256
            else if(K < 2048) GridZ = K >> 9;//div 512
            else GridZ = K >> 10;//K_slice = K div 1024
            if(GridZ > matMulT1_MaxPartNum) GridZ = matMulT1_MaxPartNum;
            
            int part = GridZ - 1;
            int sizeC = N*M;
            long[] block = core.malloc(sizeC * part);
            long Cbuf_address = block[1];
            
            //stage1------------------------------------------------------------
            Cuda_matMul.matMulT1SK(streamArray, length, GridZ,
                    A_address, 
                    B_address,
                    C_address, Cbuf_address, 
                    N, M, K);
            
            //stage2------------------------------------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
            
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_matMul.SKbuf_summary(stream, 
                    Cbuf_address, C_address, 
                    part, sizeC);
            
            Cuda.deleteEvent(event);
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matMulT2">
    @Override
    public Syncer matMulT2(long C_address,
            long A_address, long B_address, 
            int N, int M, int K)
    {
        int length = Cuda_matMul.streamSize(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_matMul.matMulT2(streamArray, length, 
                    A_address, B_address, C_address,
                    N, M, K);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Batch matMul">
    //<editor-fold defaultstate="collapsed" desc="batchMatMul">
    private boolean batchMatMul_useTexture = true;
    public boolean batchMatMul_useTexture() { return batchMatMul_useTexture; }
    public CudaFloat32EngineBase batchMatMul_useTexture(boolean flag) {
        this.batchMatMul_useTexture = flag;
        return this;
    }

    @Override
    public Syncer batchMatMul(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int M, int BK, int AK) 
    {
        int length = Cuda_batchMatMul.streamSize(N, M);
        long[] streamArray = streamPool.getStreamArray(length);
        if(batchMatMul_useTexture && (N > 47) && (M > 47)) {
            Cuda_batchMatMul.batchMatMul_texture(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, M, BK, AK);
        }
        else { 
            Cuda_batchMatMul.batchMatMul(streamArray, length, 
                    A_address, B_address, C_address,
                    Batch, N, M, BK, AK);
        }
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT1">
    @Override
    public Syncer batchMatMulT1(long C_address, 
            long A_address, long B_address, 
            int Batch, int CN, int AN, int M, int K)
    {
        int length = Cuda_batchMatMul.streamSize(CN, M);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_batchMatMul.batchMatMulT1(streamArray, length,
                A_address, B_address, C_address,
                Batch, CN, AN, M, K);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="batchMatMulT2">
    private boolean batchMatMulT2_useTexture = true;
    public boolean batchMatMulT2_useTexture() { return batchMatMulT2_useTexture; }
    public synchronized CudaFloat32EngineBase batchMatMulT2_useTexture(boolean flag) {
        this.batchMatMulT2_useTexture = flag;
        return this;
    }
    
    @Override
    public Syncer batchMatMulT2(long C_address, 
            long A_address, long B_address,
            int Batch, int N, int CM, int BM, int K)
    {
        int length = Cuda_batchMatMul.streamSize(N, CM);
        long[] streamArray = streamPool.getStreamArray(length);
        boolean useTexture = batchMatMulT2_useTexture && (N > 47) && (CM > 47);
        
        if(useTexture) {
            Cuda_batchMatMul.batchMatMulT2_texture(streamArray, length,
                    A_address, B_address, C_address,
                    Batch, N, CM, BM, K);
        }
        else {
            Cuda_batchMatMul.batchMatMulT2(streamArray, length,
                    A_address, B_address, C_address, 
                    Batch, N, CM, BM, K);
        }
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Convolution3D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation: conv3D">
    private boolean conv3D_useTexture = true;
    public boolean conv3D_useTexture() { return conv3D_useTexture; }
    public CudaFloat32EngineBase conv3D_useTexture(boolean flag) {
        conv3D_useTexture = flag;
        return this;
    }
    
    boolean conv3D_remode = true;
    public boolean conv3D_remode() { return conv3D_remode; }
    public CudaFloat32EngineBase conv3D_remode(boolean flag) {
        conv3D_remode = flag;
        return this;
    }
    
    private float conv3D_remode_V2_Q = 1.05f;
    public float conv3D_remode_V2_Q() { return conv3D_remode_V2_Q; }
    public CudaFloat32EngineBase conv3D_remode_V2_Q(float Q) {
        if(Q < 1.04999f) throw new IllegalArgumentException("Q must >= 1.05");
        conv3D_remode_V2_Q = Q;
        return this;
    }
    
    private float conv3D_V2_Q = 1.25f;
    public float conv3D_V2_Q() { return conv3D_V2_Q; }
    public CudaFloat32EngineBase conv3D_V2_Q(float Q) {
        if(Q < 1.09999) throw new IllegalArgumentException("Q must >= 1.1");
        conv3D_V2_Q = Q;
        return this;
    }
    
    @Override 
    public Syncer conv3D(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW, 
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {
        //====[remode kernel to speed up when computation is heavy]=============
        if(conv3D_remode && (OC > 31) && ((N*OH*OW) > 31)) {
            boolean V2 = (N > 63) && (OC > 63) && //No padding and [FH = FW = 1], V2 = false
                    Cuda_conv3D.paddingScaleUp(IH, IW, OH, OW, FH, FW, sh, sw) > conv3D_remode_V2_Q;//default 1.1
            
            int length = (V2? 
                    Cuda_conv3D.streamSizeV2(N, OC)://V2
                    Cuda_conv3D.streamSize(OH, OW, N, OC));//V1
            long[] streamArray  = streamPool.getStreamArray(length);
            
            //stage1: remode W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]============
            int CW_size = FH * FW * IC * OC;//CW[FH, FW, IC, OC]
            long block[] = core.malloc(CW_size);
            long CW_address = block[1];
            
            long stream = streamArray[0];
            Cuda_conv3D.kernel_remode(stream, 
                    W_address, CW_address, 
                    FH, FW, OC, IC);
            
            //stage2: conv3D====================================================
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);
            Cuda.streamsWaitEvent_default(streamArray, length, event);
            
            if((FH == 1) && (FW == 1)) { 
                Cuda_conv3D.conv3D_GemmR_W1(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address,
                        Y_address,
                        N, IC, OC);
            } 
            else if(V2) {
                Cuda_conv3D.conv3D_GemmV2R(conv3D_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            else if(conv3D_useTexture) { 
                Cuda_conv3D.conv3D_GemmR_texture(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            }
            else { 
                Cuda_conv3D.conv3D_GemmR(streamArray, length,
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            }
            
            Cuda.deleteEvent(event);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);
        }
        //default conv3D method=================================================
        else {
            boolean V2 = (N > 127) && (OC > 127) && //No padding and [FH = FW = 1], V2 = false
                    Cuda_conv3D.paddingScaleUp(IH, IW, OH, OW, FH, FW, sh, sw) >= conv3D_V2_Q;//default 1.25
            int length = (V2 ? 
                    Cuda_conv3D.streamSizeV2(N, OC)://V2
                    Cuda_conv3D.streamSize(OH, OW, N, OC));//V1
            long[] streamArray  = streamPool.getStreamArray(length);
            
            boolean useTexture = conv3D_useTexture && (OC > 15);
            
            if((FH == 1) && (FW == 1)) {
                Cuda_conv3D.conv3D_W1(streamArray, length,
                        X_address, IH, IW,
                        W_address,
                        Y_address,
                        N, IC, OC);
            } 
            else if((ph == 0) && (pw == 0)) {
                Cuda_conv3D.conv3D_np(streamArray, length, 
                    X_address, IH, IW,
                    W_address, FH, FW, 
                    Y_address, OH, OW,
                    N, IC, OC, 
                    sh, sw);
            }
            else if(V2) { 
                Cuda_conv3D.conv3DV2(useTexture, streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);
            }
            else if(useTexture) {  
                Cuda_conv3D.conv3D_texture(streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);
            }
            else { 
                Cuda_conv3D.conv3D(streamArray, length,
                       X_address, IH, IW,
                       W_address, FH, FW,
                       Y_address, OH, OW,
                       N, IC, OC,
                       sh, sw, ph, pw);
            }
            return new StreamArraySyncer(streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="forward propagation: conv3D width bias">
    @Override 
    public Syncer conv3D_biased(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            long W_address, int FH, int FW,         
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw,
            long Bias_address, //stride = OC, lengthv = N*OH*OW*OC
            int lengthv, int width)    
    {
        int GN = OC, GM = N*OH*OW;
        //remode kernel to speed up when computation is heavy===================
         if(conv3D_remode && (GN > 31) && (GM > 31)) {//(GN, GM) >= 32
            boolean V2 = (N > 63) && (OC > 63) && 
                    Cuda_conv3D.paddingScaleUp(IH, IW, OH, OW, FH, FW, sh, sw) > conv3D_remode_V2_Q;//default 1.1;
            
            int length = (V2 ? 
                    Cuda_conv3D.streamSizeV2(N, OC)://V2
                    Cuda_conv3D.streamSize(OH, OW, N, OC));//V1
            
            long[] streamArray = streamPool.getStreamArray(length);
            
            //stage1: remode W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]============
            int CW_size = FH * FW * IC * OC;//CW[FH, FW, IC, OC]
            long block[] = core.malloc(CW_size);
            long CW_address = block[1];
            
            long stream = streamArray[0];
            Cuda_conv3D.kernel_remode(stream,
                    W_address, CW_address, 
                    FH, FW, OC, IC);
            
            //stage2: conv3D====================================================
            long event1 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event1, stream);
            Cuda.streamsWaitEvent_default(streamArray, length, event1);
            
            if((FH == 1) && (FW == 1)) {
                Cuda_conv3D.conv3D_GemmR_W1(streamArray, length,
                        X_address, IH, IW, 
                        W_address, 
                        CW_address,
                        Y_address, 
                        N, IC, OC);
            }
            else if(V2) {
                Cuda_conv3D.conv3D_GemmV2R(conv3D_useTexture, streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            else if(conv3D_useTexture) {
                Cuda_conv3D.conv3D_GemmR_texture(streamArray, length, 
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            }
            else {
                Cuda_conv3D.conv3D_GemmR(streamArray, length,
                        X_address, IH, IW, 
                        W_address, CW_address, FH, FW,
                        Y_address, OH, OW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            }
            Cuda.deleteEvent(event1);
            
            //stage3: add bias==================================================
            long event2 = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event2, streamArray, length);
            Cuda.streamWaitEvent_default(stream, event2);
            
            Cuda_function.linear_dual2D_row(stream, Y_address,
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f, 
                    Y_address,
                    lengthv, width, OC);
            
            Cuda.deleteEvent(event2);
            return new BiasedForwardBlockSyncer(streamPool, streamArray, core, block);
        }
        //default conv3D method=================================================
        else {
            boolean V2 = (N > 127) && (OC > 127) &&
                    (Cuda_conv3D.paddingScaleUp(IH, IW, OH, OW, FH, FW, sh, sw)) >= conv3D_V2_Q;//default 1.25
            
            int length = (V2?
                    Cuda_conv3D.streamSizeV2(N, OC)://V2
                    Cuda_conv3D.streamSize(OH, OW, N, OC));//V1
            
            long[] streamArray = streamPool.getStreamArray(length);
            boolean useTexture = conv3D_useTexture && (GN > 15);
            
            if((FH == 1) && (FW == 1)) {
                Cuda_conv3D.conv3D_W1(streamArray, length,
                        X_address, IH, IW,
                        W_address,
                        Y_address,
                        N, IC, OC);
            } 
            else if((ph == 0) && (pw == 0)) {//no padding
                Cuda_conv3D.conv3D_np(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw);
            }
            else if(V2) {
                Cuda_conv3D.conv3DV2(useTexture, streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            else if(useTexture) { 
                Cuda_conv3D.conv3D_texture(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            else {
                Cuda_conv3D.conv3D(streamArray, length,
                        X_address, IH, IW,
                        W_address, FH, FW,
                        Y_address, OH, OW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            //====Stage2: add Bias==============================================
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, length);
            Cuda.streamWaitEvent_default(streamArray[0], event);

            Cuda_function.linear_dual2D_row(streamArray[0], Y_address,
                    Bias_address, OC,
                    1.0f, 1.0f, 0.0f,
                    Y_address,
                    lengthv, width, OC);

            Cuda.deleteEvent(event);
            return new BiasedForwardSyncer(streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: dconv3D_deltaW"> 
    private int dconv3D_deltaW_SplitKThreshold = 512;
    public int dconv3D_deltaW_SplitKThreshold() { return dconv3D_deltaW_SplitKThreshold; }
    public CudaFloat32EngineBase dconv3D_deltaW_SplitKThreshold(int threshold) {
        if(!Num.isPowerOf2(threshold)) throw new IllegalArgumentException("threshold must be a power of 2");
        if(threshold < 256) throw new IllegalArgumentException("threshold must >= 256");
        
        dconv3D_deltaW_SplitKThreshold = threshold;
        return this;
    }

    /**
     * <pre>
     * The old Version for GridZ: {@code 
     * if(GK < 1024) GridZ = GK >> 8;//div 256
     * else if(GK < 2048) GridZ = GK >> 9;//div 512
     * else GridZ = GK >> 10;//div 1024 }
     * </pre>
     */
    private int dconv3D_deltaW_SK_MaxPartNum = 16;
    public int dconv3D_deltaW_SK_MaxPartNum() { return dconv3D_deltaW_SK_MaxPartNum; } 
    public CudaFloat32EngineBase dconv3D_deltaW_SK_MaxPartNum(int maxPartNum) {
        if(maxPartNum < 0) throw new IllegalArgumentException("maxPartNum must >= 0");
        if(maxPartNum < 4) throw new IllegalArgumentException("maxPartNum must >= 4");
        dconv3D_deltaW_SK_MaxPartNum = maxPartNum;
        return this;
    }
    
    private float dconv3DdeltaW_SK_V2_Q = 1.15f;
    public float dconv3D_deltaW_SK_V2_Q() { return dconv3DdeltaW_SK_V2_Q; }
    public CudaFloat32EngineBase dconv3DdeltaW_SK_V2_Q(float Q) {
        if(Q < 1.1) throw new IllegalArgumentException("Q >= 1.1");
        this.dconv3DdeltaW_SK_V2_Q = Q;
        return this;
    }
    
    @Override
    public Syncer dconv3D_deltaW(
            long deltaW_address, int FH, int FW,
            long X_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int N, int IC, int OC, 
            int sh, int sw, int ph, int pw) 
    {     
        int GK = N * OH * OW;
        //====[GK < deconv3DdeltaW_SplitKThreshold]=============================
        if(GK < dconv3D_deltaW_SplitKThreshold) 
        {
            int length = Cuda_dconv3D_deltaW.streamSize(FH, FW, OC, IC);
            long[] streamArray = streamPool.getStreamArray(length);
            
            if((FH == 1) && (FW == 1)) {
                Cuda_dconv3D_deltaW.dconv3D_deltaW_W1(streamArray, length,
                        X_address, IH, IW,
                        deltaY_address,
                        deltaW_address, 
                        N, IC, OC);
            } 
            else {
                Cuda_dconv3D_deltaW.dconv3D_deltaW(streamArray, length,
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address, FH, FW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            return new StreamArraySyncer(streamPool, streamArray);        
        } 
        //====[GK >= deconv3DdeltaW_SpkitKThreshold]============================
        else { 
            boolean V2 = (OC > 63) && (IC > 63) &&
                    Cuda_dconv3D_deltaW.paddingScaleUp(IH, IW, OH, OW, FH, FW, sh, sw) >= dconv3DdeltaW_SK_V2_Q;
            
            int length = (V2?
                    Cuda_dconv3D_deltaW.streamSizeV2(OC, IC)://V2
                    Cuda_dconv3D_deltaW.streamSize(FH, FW, OC, IC));//V1
            long[] streamArray = streamPool.getStreamArray(length);
                         
            int b1 = (V2? Cuda_conv3D.blockNumV2(OH, OW, N, OC) : Cuda_conv3D.blockNum(OH, OW, N, OC));
            int b2 = (V2? Cuda_dconv3D_deltaX.blockNumV2(IH, IW, N, IC) : Cuda_dconv3D_deltaX.blockNum(IH, IW, N, IC));
            int b3 = (V2? Cuda_dconv3D_deltaW.blockNumV2(FH, FW, OC, IC) : Cuda_dconv3D_deltaW.blockNum(FH, FW, OC, IC));
            int GridZ = (b1 + b2) / (b3 << 1);
            
            if(GridZ < 2) GridZ = 2;//512 / 64 = 8
            if(GridZ > dconv3D_deltaW_SK_MaxPartNum) GridZ = dconv3D_deltaW_SK_MaxPartNum;
            int GridZ2 = GK >> 8; //512 / 256 = 2
            if(GridZ > GridZ2) GridZ = GridZ2;
            
            int part = GridZ - 1;
            int sizeW = OC * FH * FW * IC;
            long[] block = core.malloc(sizeW * part);
            long deltaW_buf_address = block[1];
            
            //stage1: find gradient of W----------------------------------------
            if((FH == 1) && (FW == 1)) {
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmSK_W1(streamArray, length, GridZ, 
                        X_address, IH, IW,
                        deltaY_address,
                        deltaW_address, 
                        deltaW_buf_address,
                        N, IC, OC);
            }
            else if(V2) { 
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmV2SK(streamArray, length, GridZ,
                        X_address, IH, IW, 
                        deltaY_address, OH, OW,
                        deltaW_address, 
                        deltaW_buf_address, FH, FW, 
                        N, IC, OC, 
                        sh, sw, ph, pw);
            }
            else {
                Cuda_dconv3D_deltaW.dconv3D_deltaW_GemmSK(streamArray, length, GridZ,
                        X_address, IH, IW,
                        deltaY_address, OH, OW,
                        deltaW_address,
                        deltaW_buf_address, FH, FW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            //stage2: sum up deltaW from each part------------------------------
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, streamArray, streamArray.length);
             
            long stream = streamArray[0];
            Cuda.streamWaitEvent_default(stream, event);
            Cuda_dconv3D_deltaW.buf_summary(stream, 
                    deltaW_buf_address, deltaW_address, 
                    part, sizeW);
            
            Cuda.deleteEvent(event);
            return new SplitKSyncer(core, block, streamPool, streamArray);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: dconv3D_deltaX">
    private boolean dconv3D_deltaX_s1_useTexture = true;
    public boolean dconv3D_deltaX_s1_useTexture() { return dconv3D_deltaX_s1_useTexture; }
    public CudaFloat32EngineBase dconv3D_deltaX_s1_useTexture(boolean flag) {
        dconv3D_deltaX_s1_useTexture = flag;
        return this;
    }
    
    private boolean dconv3D_deltaX_ks_useTexture = false;
    public boolean dconv3D_deltaX_ks_useTexture() { return dconv3D_deltaX_ks_useTexture; }
    public CudaFloat32EngineBase dconv3D_deltaX_ks_useTexture(boolean flag) {
        dconv3D_deltaX_ks_useTexture = flag;
        return this;
    } 
    
    private float dconv3D_deltaX_ks_V2_Q = 1.1f;
    public float dconv3D_deltaX_ks_V2_Q() { return dconv3D_deltaX_ks_V2_Q; }
    public CudaFloat32EngineBase dconv3D_deltaX_ks_V2_Q(float Q) {
        if(Q < 1.05f) throw new IllegalArgumentException("Q must >= 1.05");
        dconv3D_deltaX_ks_V2_Q = Q;
        return this;
    }
    
    private float dconv3D_deltaX_s1_V2_Q = 1.1f;
    public float dconv3D_deltaX_s1_V2_Q() { return dconv3D_deltaX_s1_V2_Q; }
    public CudaFloat32EngineBase dconv3D_deltaX_s1_V2_Q(float Q) {
        if(Q < 1.05f) throw new IllegalArgumentException("Q must >= 1.05");
        dconv3D_deltaX_s1_V2_Q = Q;
        return this;
    }
    
    @Override
    public Syncer dconv3D_deltaX(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            long W_address, int FH, int FW,
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw) 
    {   
        //crossAdd kernel: when IC is small=====================================
        if((IC <= 8) || (IC <= 16) && (sh * sw >= 4)) {
            int length = Cuda_dconv3D_deltaX.streamSize_CrossAdd(OH, OW, N, OC);
            long[] streamArray = streamPool.getStreamArray(length);
            
            Cuda_dconv3D_deltaX.dconv3D_deltaX_crossAdd(streamArray, length, 
                    deltaY_address, OH, OW,
                    W_address, FH, FW,
                    deltaX_address, IH, IW,
                    N, IC, OC,
                    sh, sw, ph, pw);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //dense kernel: [FH = FW = 1]===========================================
        else if((FH == 1) && (FW == 1)) { 
            int length = Cuda_dconv3D_deltaX.streamSize_ZeroPadding_dense(IH, IW, N, IC);
            long [] streamArray = streamPool.getStreamArray(length);
            
            Cuda_dconv3D_deltaX.dconv3D_deltaX_W1(streamArray, length, 
                    deltaY_address, 
                    W_address, 
                    deltaX_address, IH, IW,
                    N, IC, OC);
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //dense kernel: [sh = sw = 1]===========================================
        else if((sh == 1) && (sw == 1))
        {        
            boolean V2 = (N > 63) && (IC > 63) && 
                    Cuda_dconv3D_deltaX.s1_paddingScaleUp(IH, IW, FH, FW, OH, OW) > dconv3D_deltaX_s1_V2_Q;//default 1.1
            
            int length = (V2 ?
                    Cuda_dconv3D_deltaX.streamSizeV2(N, IC)://V2
                    Cuda_dconv3D_deltaX.streamSize_ZeroPadding_dense(IH, IW, N, IC));//V1
            long[] streamArray = streamPool.getStreamArray(length);
            
            boolean useTexture = dconv3D_deltaX_s1_useTexture && (IC > 15);
            
            if(V2) {
                Cuda_dconv3D_deltaX.dconv3D_deltaX_V2_s1(useTexture, streamArray, length, 
                        deltaY_address, OH, OW,
                        W_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw);
            }
            else if(useTexture) {
                Cuda_dconv3D_deltaX.dconv3D_deltaX_s1_texture(streamArray, length,
                        deltaY_address, OH, OW,
                        W_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw);
            }
            else { 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_s1(streamArray, length,
                        deltaY_address, OH, OW,
                        W_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw);
            }
            return new StreamArraySyncer(streamPool, streamArray);
        }
        //sparse kernel: [sh == sw == 2, (IH, IW)%2 == 0]=======================
        else if((sh == 2) && (sw == 2) && (IH & 1) == 0 && (IW & 1) == 0)
        {
            boolean V2 = (N > 63) && (IC > 63) && 
                    Cuda_dconv3D_deltaX.Ims2_paddingScaleUp(IH, IW, FH, FW, OH, OW) > dconv3D_deltaX_ks_V2_Q;//default 1.1
            
            int length = (V2 ? 
                    Cuda_dconv3D_deltaX.streamSizeV2(N, IC)://V2
                    Cuda_dconv3D_deltaX.streamSize_KernelSplit(IH, IW, N, IC, sh, sw));//V1
            long[] streamArray = streamPool.getStreamArray(length);
            
            //stage1: remode W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]==
            int CFH = (FH + 1) >> 1;
            int CFW = (FW + 1) >> 1;
            int CW_size = (OC * CFH * CFW * IC) << 2;//OC * CFH * CFW * sh * sw
            long block[] = core.malloc(CW_size);
            long CW_address = block[1];
            
            long stream = streamArray[0];
            Cuda_dconv3D_deltaX.ks_remodev2(stream, 
                    W_address, FH, FW,
                    CW_address, CFH, CFW,
                    OC, IC, sh, sw);
            
            //stage2: deconv3D==================================================
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);//streamArray wait stream
            Cuda.streamsWaitEvent_default(streamArray, length, event);
            
            boolean useTexture = dconv3D_deltaX_ks_useTexture && (IC > 15);
            
            if(V2) { 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksV2_Ims2R(useTexture, streamArray, length, 
                        deltaY_address, OH, OW, 
                        CW_address, FH, FW,
                        deltaX_address, IH, IW, 
                        N, IC, OC,
                        ph, pw);
            }
            else if(useTexture) {
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksIms2R_texture(streamArray, length,
                        deltaY_address, OH, OW,
                        CW_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        ph, pw);
            }
            else { 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksIms2R(streamArray, length, 
                        deltaY_address, OH, OW,
                        CW_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC, 
                        ph, pw);
            }
            
            Cuda.deleteEvent(event);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);
        }
        //sparse kernel: [(IH, IW)% (sh, sw)== 0] ==============================
        else if((IH % sh) == 0 && (IW % sw) == 0)
        {
            int length = Cuda_dconv3D_deltaX.streamSize_KernelSplit(IH, IW, N, IC, sh, sw);
            long[] streamArray = streamPool.getStreamArray(length);
            
            //stage1 remode W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]===
            int CFH = (FH + sh - 1) / sh;
            int CFW = (FW + sw - 1) / sw;
            int CW_size = (OC * CFH * CFW * IC * sh * sw);
            long block[] = core.malloc(CW_size);
            long CW_address = block[1];
            
            long stream = streamArray[0];
            Cuda_dconv3D_deltaX.ks_remodev2(stream, 
                    W_address, FH, FW,
                    CW_address, CFH, CFW,
                    OC, IC, sh, sw);
            
            //stage2: deconv3D==================================================
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);//streamArray wait stream
            Cuda.streamsWaitEvent_default(streamArray, length, event);
            
            boolean useTexture = dconv3D_deltaX_ks_useTexture && (IC > 15);
            
            if(useTexture) { 
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksImsR_texture(streamArray, length,
                        deltaY_address, OH, OW,
                        CW_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            else {
                Cuda_dconv3D_deltaX.dconv3D_deltaX_ksImsR(streamArray, length, 
                        deltaY_address, OH, OW,
                        CW_address, FH, FW, 
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        sh, sw, ph, pw);
            }
            
            Cuda.deleteEvent(event);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);
        }
        //sparse kernel(KernelSplitR)===========================================
        else {
            int length = Cuda_dconv3D_deltaX.streamSize_KernelSplit(IH, IW, N, IC, sh, sw);
            long[] streamArray = streamPool.getStreamArray(length);
            
            //stage1 remode W[OC, FH, FW, IC] -> CW[sh, sw, CFH, CFW, OC, IC]===
            int CFH = (FH + sh - 1) / sh;
            int CFW = (FW + sw - 1) / sw;
            int CW_size = (OC * CFH * CFW * IC * sh * sw);
            long block[] = core.malloc(CW_size);
            long CW_address = block[1];
            
            long stream = streamArray[0];
            Cuda_dconv3D_deltaX.ks_remodev2(stream, 
                    W_address, FH, FW,
                    CW_address, CFH, CFW,
                    OC, IC, sh, sw);
          
            //stage2: deconv3D==================================================
            long event = Cuda.newEvent_DisableTiming();
            Cuda.eventRecord(event, stream);//streamArray wait stream
            Cuda.streamsWaitEvent_default(streamArray, length, event);
            
            Cuda_dconv3D_deltaX.dconv3D_deltaX_kernelSplit(streamArray, length,
                        deltaY_address, OH, OW,
                        CW_address, FH, FW,
                        deltaX_address, IH, IW,
                        N, IC, OC,
                        sh, sw, ph, pw);
               
            Cuda.deleteEvent(event);
            return new StreamArrayBlockSyncer(streamPool, streamArray, core, block);     
        }
    }
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Pooling 2D">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    @Override
    public Syncer pool2D_max(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW, 
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_max(streamArray, length, 
                X_address, IH, IW, 
                FH, FW,
                Y_address, OH, OW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }

    @Override
    public Syncer pool2D_max_indexed(
            long Y_address, long Index_address, int OH, int OW,
            long X_address, int IH, int IW, 
            int FH, int FW, 
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_max_indexed(streamArray, length, 
                X_address, IH, IW,
                FH, FW, 
                Y_address, Index_address, OH, OW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer pool2D_avg(
            long Y_address, int OH, int OW,
            long X_address, int IH, int IW,
            int FH, int FW,
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_avg(streamArray, length, 
                X_address, IH, IW, 
                FH, FW,
                Y_address, OH, OW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    @Override
    public Syncer pool2D_avg_ignore_padding(
            long Y_address, int OH, int OW, 
            long X_address, int IH, int IW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw) 
    {
        int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_pool2D.pool2D_avg_ignore_padding(streamArray, length, 
                X_address, IH, IW,
                FH, FW, 
                Y_address, OH, OW,
                N, IC,
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation">
    @Override
    public Syncer upool2D_max(
            long deltaX_address, long X_address, int IH, int IW,
            long deltaY_address, long Y_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        int length = Cuda_upool2D.streamSize(IH, IW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_upool2D.upool2D_max(streamArray, length,
                deltaY_address, Y_address, OH, OW, 
                FH, FW, 
                deltaX_address, X_address, IH, IW,
                N, IC, 
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Xeno">
    //        if(FH == sh && FW == sw) {
//            long stream = streamPool.getStream();
//            int Xsize = N*IH*IW*IC;
//            
//            //stage1: deltaX = 0------------------------------------------------
//            long event = Cuda.newEvent_DisableTiming();
//            Cuda.memsetAsync(stream, deltaX_address, 0, Xsize << 2L);
//           
//            //stage2: deltaX[index] = deltaY------------------------------------
//            int Ysize = N * OH * OW * IC;
//            Cuda.streamWaitEvent_default(stream, event);
//            Cuda_expk2.dstIndexedMemcpy(stream, 
//                    deltaY_address, Index_address, 
//                    deltaX_address, 
//                    Ysize, IC, IC);
//            
//            return new StreamSyncer(streamPool, stream);
//        }
    //</editor-fold>
    @Override
    public Syncer upool2D_max_Indexed(
            long deltaX_address, int IH, int IW, 
            long deltaY_address, long Index_address, int OH, int OW, 
            int FH, int FW, 
            int N, int IC,
            int sh, int sw, int ph, int pw) 
    {

        int length = Cuda_upool2D.streamSize(IH, IW, N, IC);
        long[] streamArray = streamPool.getStreamArray(length);
        Cuda_upool2D.upool2D_max_Indexed(streamArray, length,
                deltaY_address, Index_address, OH, OW,
                FH, FW,
                deltaX_address, IH, IW,
                N, IC,
                sh, sw, ph, pw);
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer upool2D_avg(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW, 
            int FH, int FW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
        long[] streamArray;
        if(FH == sh && FW == sw) {//tiled average pool2D
            int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
            streamArray = streamPool.getStreamArray(length);
            Cuda_upool2D.upool2D_avg_tiled(streamArray, length, 
                    deltaY_address, OH, OW,
                    FH, FW, 
                    deltaX_address, IH, IW, 
                    N, IC, 
                    sh, sw, ph, pw);
        }
        else 
        {
            int length = Cuda_upool2D.streamSize(IH, IW, N, IC);
            streamArray = streamPool.getStreamArray(length);          
            Cuda_upool2D.upool2D_avg(streamArray, length,
                deltaY_address, OH, OW, 
                FH, FW, 
                deltaX_address, IH, IW, 
                N, IC,
                sh, sw, ph, pw);
        }
        
        return new StreamArraySyncer(streamPool, streamArray);
    }
    
    @Override
    public Syncer upool2D_avg_ignore_padding(
            long deltaX_address, int IH, int IW,
            long deltaY_address, int OH, int OW,
            int FH, int FW, 
            int N, int IC, 
            int sh, int sw, int ph, int pw) 
    {
        long[] streamArray;
        if(FH == sh && FW == sw) {//tiled average pool2D
            int length = Cuda_pool2D.streamSize(OH, OW, N, IC);
            streamArray = streamPool.getStreamArray(length);    
            Cuda_upool2D.upool2D_avg_ignore_padding_tiled(streamArray, length, 
                    deltaY_address, OH, OW,
                    FH, FW, 
                    deltaX_address, IH, IW, 
                    N, IC, 
                    sh, sw, ph, pw);
        }
        else {
            int length = Cuda_upool2D.streamSize(IH, IW, N, IC);
            streamArray = streamPool.getStreamArray(length);
            
            Cuda_upool2D.upool2D_avg_ignore_padding(streamArray, length,
                deltaY_address, OH, OW, 
                FH, FW, 
                deltaX_address, IH, IW, 
                N, IC,
                sh, sw, ph, pw);
        }
        
        return new StreamArraySyncer(streamPool, streamArray);
    }

    //</editor-fold>
    //</editor-fold> 
    
    //<editor-fold defaultstate="collapsed" desc="Math Function">
    //<editor-fold defaultstate="collapsed" desc="equal, linear, quadratic, rpl, div, add_div"> 
    //<editor-fold defaultstate="collapsed" desc="equal_abs">
    @Override
    public Syncer equal_abs2D(long Y_address, 
            long X1_address, long X2_address,
            float min, float max, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D(stream,
                X1_address, X2_address,
                min, max, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer equal_abs2D_int8(long Y_address, 
            long X1_address, long X2_address, 
            byte min, byte max,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D_char(stream, 
                X1_address, X2_address, 
                min, max,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer equal_abs2D_int32(long Y_address, 
            long X1_address, long X2_address,
            int min, int max, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.equal_abs2D_int(stream, 
                X1_address, X2_address,
                min, max, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear_greater">   
    @Override
    public Syncer linear_greater2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater2D(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear_greater_dual2D(long Y_address,
            long X1_address, long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_greater_dual2D(stream, 
                X1_address, X2_address,
                alpha, beta, gamma,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear">
    @Override
    public Syncer linear2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D(stream, 
                alpha, X_address, beta, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
   
    @Override
    public Syncer linear_dual_out2D(long Y1_address, long Y2_address, 
            long X_address,
            float alpha1, float beta1, 
            float alpha2, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual_out2D(stream, 
                X_address,
                alpha1, beta1, 
                alpha2, beta2,
                Y1_address, Y2_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int8 to float">
    @Override
    public Syncer linear2D_int8_to_dtype(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_char2float(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2D_float_to_int8(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_float2char(stream,
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear: int32 to float">
    @Override
    public Syncer linear2D_int32_to_dtype(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_int2float(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear2D_dtype_to_int32(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear2D_float2int(stream,
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="linear_dual">
    @Override
    public Syncer linear_dual2D(long Y_address, 
            long X1_address, 
            long X2_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D(stream, 
                X1_address, X2_address,
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override //Y = sum(alpha*Xs[i] + beta)
    public Syncer linear_summary2D(long Y_address,
            long[] Xs, float alpha, float beta, //Xs.length >= 2
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        
        Cuda_function.linear_dual2D(stream,
                Xs[0], Xs[1], 
                alpha, alpha, 2*beta,
                Y_address, //Y = X0*alpha + X1*alpha + 2*beta
                lengthv, width, stride);
        
        for(int i=2; i<Xs.length; i++) {
            Cuda_function.linear_dual2D(stream, 
                    Y_address, Xs[i], 
                    1.0f, alpha, beta, 
                    Y_address, //Y = Y + Xs[i]*alpha + beta
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="linear, linear_dual: row/field">
    @Override
    public Syncer linear_dual2D_row(long Y_address, 
            long X1_address,
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D_row(stream, 
                X1_address, X2_address, row_lengthv,
                alpha, beta, gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer linear_dual2D_field(long Y_address,
            long X1_address, 
            long X2_address, int row_lengthv,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.linear_dual2D_field(stream, 
                X1_address, X2_address, row_lengthv,
                alpha, beta, gamma, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);   
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic">
    @Override
    public Syncer quadratic2D( long Y_address, 
            long X_address, float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic2D(stream, 
                X_address, alpha, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer quadratic2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: quadratic_dual">
    @Override
    public Syncer quadratic_dual2D(long Y_address,
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D(stream, 
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2,
                C,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer quadratic_dual2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D_deltaX(stream, 
                deltaX1_address, deltaX2_address,
                deltaY_address, 
                X1_address, X2_address,
                k11, k12, k22,
                k1, k2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override //Y = sum(alpha*Xs[i]^2 + beta*Xs[i] + gamma)
    public Syncer quadratic_summary2D(long Y_address,
            long[] Xs, float alpha, float beta, float gamma, //Xs.length >= 2
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
       
        Cuda_function.quadratic_dual2D(stream,
                    Xs[0], Xs[1], 
                    alpha, 0, alpha, 
                    beta, beta, 2*gamma,
                    Y_address, //Y = alpha*X0^2 + alpha*X1^2 + beta*X1 + beta*X2 +2*gamma
                    lengthv, width, stride);
        
        for(int i=2; i<Xs.length; i++) {
            Cuda_function.quadratic_dual2D(stream, 
                    Y_address, Xs[i],
                    0, 0, alpha, 
                    1.0f, beta, gamma,
                    Y_address, //Y = Y + alpha*Xs[i]^2 + beta*Xs[i] +gamma
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic_dual: row/field">
    @Override
    public Syncer quadratic_dual2D_row(long Y_address, 
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.quadratic_dual2D_row(stream, 
                X1_address, 
                X2_address, row_lengthv, 
                k11, k12, k22,
                k1, k2, 
                C,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer quadratic_dual2D_field(long Y_address, 
            long X1_address, 
            long X2_address, int row_lengthv,
            float k11, float k12, float k22,
            float k1, float k2, 
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.quadratic_dual2D_field(stream, 
                X1_address, 
                X2_address, row_lengthv,
                k11, k12, k22, 
                k1, k2, 
                C,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: rpl">
    @Override
    public Syncer rpl2D(long Y_address, 
            float alpha, long X_address, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rpl2D(stream,
                alpha, X_address, beta, gamma,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer rpl2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rpl2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, alpha, gamma,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: div">
    @Override
    public Syncer div2D(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2,
            float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D(stream,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                gamma, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer div2D_deltaX(long deltaX1_address, long deltaX2_address, 
            long deltaY_address, 
            long X1_address, float alpha1, float beta1,
            long X2_address, float alpha2, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_deltaX(stream,
                deltaX1_address, deltaX2_address, 
                deltaY_address, 
                X1_address, alpha1, beta1, 
                X2_address, alpha2, beta2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="div2D: row/field">
    @Override
    public Syncer div2D_row(long Y_address, 
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2, 
            float gamma, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_row(stream,
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2, 
                gamma, row_lengthv,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer div2D_field(long Y_address,
            float alpha1, long X1_address, float beta1,
            float alpha2, long X2_address, float beta2, 
            float gamma, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.div2D_field(stream, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                gamma, row_lengthv,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: div2D, (alpha*X1 + beta1) / (alpha2*X2 + beta2)">
  
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="add_div: row\field">
    @Override
    public Syncer add_div2D_row(long Y_address,
            long X1_address,
            long X2_address,
            long X3_address, int row_lengthv,
            float alpha, float beta, float gamma, float delta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.add_div2D_row(stream,
                X1_address,
                X2_address,
                X3_address, row_lengthv,
                alpha, beta, gamma, delta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer add_div2D_field(long Y_address,
            long X1_address, 
            long X2_address, 
            long X3_address, int row_lengthv, 
            float alpha, float beta, float gamma, float delta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.add_div2D_field(stream, 
                X1_address, 
                X2_address,
                X3_address, row_lengthv,
                alpha, beta, gamma, delta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="sign, ceil, floor, abs, sqrt">
    @Override
    public Syncer sign2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sign2D(stream,
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer ceil2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.ceil2D(stream,
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer floor2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.floor2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    //<editor-fold defaultstate="collapsed" desc="BP: abs(Absolute)">
    @Override
    public Syncer abs2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.abs2D(stream,
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer abs2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long X_address, float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.abs2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    @Override
    public Syncer zero_nan2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.zero_nan2D(stream, 
                X_address, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqrt2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqrt2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqrt_quadratic_dual2D(long Y_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2,
            float C,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqrt_quadratic_dual2D(stream, 
                X1_address, X2_address, 
                k11, k12, k22, 
                k1, k2, C,
                Y_address,
                lengthv, width, stride);
         return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="min, max, clip"> 
    //<editor-fold defaultstate="collapsed" desc="min, min_dual">
    @Override
    public Syncer min2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmin,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.min2D(stream, 
                alpha, X_address, beta, vmin,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer min_dual2D(long Y_address, 
            float alpha1, long X1_address, float beta1, 
            float alpha2, long X2_address, float beta2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.min_dual2D(stream,
                alpha1, X1_address, beta1, 
                alpha2, X2_address, beta2, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="max, max_dual">
    @Override
    public Syncer max2D(long Y_address, 
            float alpha, long X_address, float beta, 
            float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.max2D(stream, 
                alpha, X_address, beta, vmax,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer max_dual2D(long Y_address,
            float alpha1, long X1_address, float beta1, 
            float alpha2, long X2_address, float beta2, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.max_dual2D(stream, 
                alpha1, X1_address, beta1,
                alpha2, X2_address, beta2,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>

    @Override
    public Syncer clip2D(long Y_address, 
            float alpha, long X_address, float beta,
            float vmin, float vmax,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.clip2D(stream,
                alpha, X_address, beta,
                vmin, vmax,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="semi-linear unit functions">
    @Override
    public Syncer exp2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.exp2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    //<editor-fold defaultstate="collapsed" desc="BP: log">
    @Override
    public Syncer log2D(long Y_address, 
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.log2D(stream, 
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer log2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.log2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: relu">
    @Override
    public Syncer relu2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D(stream,
                X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer relu2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer relu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.relu2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: leakyRelu">
    @Override
    public Syncer leakyRelu2D(long Y_address,
            long X_address, float k, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D(stream,
                X_address, k, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer leakyRelu2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D_deltaX_v1(stream,
                deltaX_address,
                deltaY_address, 
                Y_address, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer leakyRelu2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address, float k,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.leakyRelu2D_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address,
                X_address, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: elu">
    @Override
    public Syncer elu2D(long Y_address,
            long X_address, float alpha, float k,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D(stream,
                X_address, alpha, k, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer elu2D_deltaX_v1(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha, float k,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address, 
                Y_address, alpha, k,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer elu2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address, float alpha, float k, //V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.elu2D_deltaX_v2(stream,
                deltaX_address,
                deltaY_address,
                X_address, alpha, k, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softplus">
    @Override
    public Syncer softPlus2D(long Y_address,
            long X_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D(stream, 
                X_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer softPlus2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer softPlus2D_deltaX_v2(long deltaX_address, 
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.softPlus2D_deltaX_v2(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="hypherbolic functions">
    //<editor-fold defaultstate="collapsed" desc="BP: tanh">
    @Override
    public Syncer tanh2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D(stream, 
                X_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer tanh2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D_deltaX_v1(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tanh2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tanh2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address, 
                X_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid">
    @Override
    public Syncer sigmoid2D(long Y_address,
            long X_address,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D(stream, 
                X_address,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sigmoid2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D_deltaX_v1(stream, 
                deltaX_address,
                deltaY_address, 
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sigmoid2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid2D_deltaX_v2(stream, 
                deltaX_address,
                deltaY_address,
                X_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmax">
    @Override
    public Syncer softmax2D(long Y_address, 
            long X_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV   = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1], maxX = blockMax[1];//expXm_max = sumEachRow: exp(X - maxX)
      
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmax(stream, //Y = exp(X - M), V = sumEachRow(Y) = sum(exp(X-M))
                X_address, maxX,
                Y_address, 
                field_length, row_lengthv, 
                expXm_max_rowSum,//result: expXm_max = sumEachRow: exp(X - maxX)
                width, stride, 1);
        
        Cuda_function.div2D_field(stream, //final value: Y -> Y/V = expX / sumOfEachRow(Y)
                1.0f, Y_address       , 0.0f, 
                1.0f, expXm_max_rowSum, 0.0f, 
                0.0f, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
       
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }
    
    @Override
    public Syncer softmax2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long deltaY_Y_rowSum = blockV[1];//deltaY_Y_rowSum = sumEachRow: deltaY * Y
        
        //deltaY_Y_rowSum is the result and buffer for this reduction
        Cuda_reduce.row_quadratic_dual(stream, 
                deltaY_address, Y_address,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 
                field_length, row_lengthv,
                deltaY_Y_rowSum,//buffer
                deltaY_Y_rowSum,//result
                width, stride, 1);
        
        Cuda_function.softmax2D_deltaX(stream, 
                deltaX_address,
                deltaY_address, Y_address, 
                deltaY_Y_rowSum, row_lengthv, 
                lengthv, width, stride);
        
        return new StreamBlockSyncer(streamPool, stream, core, blockV);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: logsoftmax">  
    @Override
    public Syncer logsoftmax2D(long Y_address, 
            long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
       
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length :  field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1],  maxX = blockMax[1];//expXm_max_rowSum = sumEachRow: exp(X - maxX)
        
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
         
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream, 
                X_address, maxX,
                field_length, row_lengthv, 
                expXm_max_rowSum,//expXm_max_rowSum = sumEachRow: exp(X - maxX)
                width, stride, 1);
         
        Cuda_function.logsoftmax2D(stream,
                X_address, maxX, 
                expXm_max_rowSum, 
                row_lengthv, 
                Y_address,
                lengthv, width, stride);
        
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }

    @Override
    public Syncer logsoftmax2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV = core.malloc(V_lengthv);
        long deltaY_rowSum = blockV[1];//Y_rowSum = sumEachRow: Y[i]
        
        //Y_rowSum is the result and buffer for this reduction
        Cuda_reduce.row_linear(stream, 
                deltaY_address, 1.0f, 0.0f, 
                field_length, row_lengthv, 
                deltaY_rowSum, //buffer
                deltaY_rowSum, //result
                width, stride, 1);
        
        Cuda_function.logsoftmax2D_deltaX(stream,
                deltaX_address, 
                deltaY_address, 
                Y_address, 
                deltaY_rowSum, row_lengthv,
                lengthv, width, stride);
        
        return new StreamBlockSyncer(streamPool, stream, core, blockV);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="trigonometric functions">
    //<editor-fold defaultstate="collapsed" desc="BP: sin">
    @Override
    public Syncer sin2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sin2D(stream, 
                alpha, X_address, beta, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sin2D_deltaX(long deltaX_address, long deltaY_address, long X_address, float alpha, float beta, int lengthv, int width, int stride) {
        long stream = streamPool.getStream();
        Cuda_function.sin2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address, 
                X_address, alpha, beta, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: tan">
    @Override
    public Syncer tan2D(long Y_address, 
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tan2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer tan2D_deltaX(long deltaX_address,
            long deltaY_address,
            long Y_address, float alpha,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.tan2D_deltaX(stream,
                deltaX_address, 
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: halfSin2D">
    @Override
    public Syncer halfSin2D(long dY_address, 
            float Amp, float alpha, long dX_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.halfSin2D(stream,
                Amp, alpha, dX_address, beta,
                dY_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer halfSin2D_deltaX(long d_deltaX_address,
            long d_deltaY_address, 
            long dY_address, float Amp, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.halfSin2D_deltaX(stream,
                d_deltaX_address, 
                d_deltaY_address,
                dY_address, Amp, alpha, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arcsin2D">
    @Override
    public Syncer arcsin2D(long Y_address,
            float alpha, long X_address, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arcsin2D(stream, 
                alpha, X_address, beta,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer arcsin2D_deltaX(long deltaX_address,
            long deltaY_address, 
            long Y_address, float alpha, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arcsin2D_deltaX(stream, 
                deltaX_address, 
                deltaY_address,
                Y_address, alpha,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: arctan2D">
    @Override
    public Syncer arctan2D(long Y_address,
            float alpha, long X_address, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arctan2D(stream, 
                alpha, X_address, beta,
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer arctan2D_deltaX(long deltaX_address, 
            long deltaY_address, 
            long Y_address, float alpha,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.arctan2D_deltaX(stream,
                deltaX_address,
                deltaY_address,
                Y_address, alpha, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="distance & loss functions">
    //<editor-fold defaultstate="collapsed" desc="BP: L1">
    @Override
    public Syncer L1_2D(long L_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L1_2D(stream, 
                Y_address, Yh_address,
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer L1_2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L1_2D_deltaYh(stream, 
                Y_address, Yh_address, 
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: L2">
    @Override
    public Syncer L2_2D(long L_address, 
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.L2_2D(stream,
                Y_address, Yh_address,
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer L2_2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.L2_2D_deltaYh(stream, 
                Y_address, Yh_address, 
                deltaYh_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: smoothL1">
    @Override
    public Syncer smoothL1_2D(long L_address,
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.smoothL1_2D(stream, 
                Y_address, Yh_address, 
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer smoothL1_2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.smoothL1_2D_deltaYh(stream,
                Y_address, Yh_address,
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: binaryCrossEntropy">
    @Override
    public Syncer binaryCrossEntropy2D(long L_address, 
            long Y_address, long Yh_address,
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.binaryCrossEntropy2D(stream, 
                Y_address, Yh_address,
                alpha, beta,
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer binaryCrossEntropy2D_deltaYh(long deltaYh_address, 
            long Y_address, long Yh_address, 
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.binaryCrossEntropy2D_deltaYh(stream,
                Y_address, Yh_address,
                alpha, beta,
                deltaYh_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: sigmoid_binaryCrossEntropy">
    @Override
    public Syncer sigmoid_binaryCrossEntropy2D(long L_address, 
            long Y_address, long X_address,
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid_binaryCrossEntropy2D(stream,
                Y_address, X_address,
                alpha, beta,
                L_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sigmoid_binaryCrossEntropy_deltaX(long deltaX_address,
            long Y_address, long X_address, 
            float alpha, float beta,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sigmoid_binaryCrossEntropy2D_deltaX(stream, 
                Y_address, X_address, 
                alpha, beta,
                deltaX_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: crossEntropy">
    @Override
    public Syncer crossEntropy2D(long L_address, 
            long Y_address, long Yh_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.crossEntropy2D(stream, 
                Y_address, Yh_address, 
                L_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer crossEntropy2D_deltaYh(long deltaYh_address,
            long Y_address, long Yh_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.crossEntropy2D_deltaYh(stream, 
                Y_address, Yh_address,
                deltaYh_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="BP: softmaxCrossEntropy">
    //<editor-fold defaultstate="collapsed" desc="forward prop">
    @Override
    public Syncer softmax_crossEntropy2D(long L_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV   = core.malloc(V_lengthv);
        long[] blockMax = core.malloc(field_lengthv);
        long expXm_max_rowSum = blockV[1], maxX = blockMax[1];
        
        Cuda_reduce.row_max(stream,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream, 
                X_address, maxX,
                field_length, row_lengthv, 
                expXm_max_rowSum,
                width, stride, 1);
        
        //L = -Y * (X - maxX) + U + (Y - 1)*log(U - exp(X - M))
        Cuda_function.softmax_crossEntropy2D(stream,
                Y_address, X_address, 
                maxX, expXm_max_rowSum, row_lengthv,
                L_address, 
                lengthv, width, stride);
        
        return new StreamBlock2Syncer(streamPool, stream, core, blockV, blockMax);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward prop">
    @Override//Yh - Y = softmax(X) - Y
    public Syncer softmax_crossEntropy2D_deltaX(long deltaX_address, 
            long Y_address, long X_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream1 = streamPool.getStream();
        long stream2 = streamPool.getStream();
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int field_lengthv = ((field_length + 3) >> 2) << 2;
        int SV = (nextM == 1? field_length : field_lengthv);
        int V_lengthv = nextM * SV;//V[nextM, N = field_length]
        long[] blockV        = core.malloc(V_lengthv);
        long[] blockMax      = core.malloc(field_lengthv);
        long[] blockY_rowSum = core.malloc(V_lengthv);
        long expXm_max_rowSum = blockV[1];
        long maxX             = blockMax[1];
        long Y_rowSum         = blockY_rowSum[1];
        
        //Stage1: compute maxX, expXm_max_rowSum, Y_rowSum======================
        Cuda_reduce.row_max(stream1,//find maxX of each row: M = maxRachRow(X)
                X_address, 
                field_length, row_lengthv,
                expXm_max_rowSum, maxX,//buffer: expXsmax_rowSum
                width, stride, 1);
       
        Cuda_reduce.row_softmaxCrossEntropy_stage1(stream1, 
                X_address, maxX,
                field_length, row_lengthv,//expXm_max_rowSum = sumEachRow: exp(X - maxX)
                expXm_max_rowSum,
                width, stride, 1);
        
        Cuda_reduce.row_linear(stream2, 
                Y_address, 1.0f, 0.0f, 
                field_length, row_lengthv,//Y_rowSum = sumEachRow: Y
                Y_rowSum, Y_rowSum,
                width, stride, 1);
        
        //Stage2: deltaX = YrowSum * [exp(X - maxX) / expXm_max_rowSum] - Y=====
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, stream1);
        Cuda.eventRecord(event, stream2);
        Cuda.streamWaitEvent_default(stream1, event);
        
        Cuda_function.softmax_crossEntropy2D_deltaX(stream1,
                Y_address, X_address,
                maxX, expXm_max_rowSum, 
                Y_rowSum, row_lengthv, 
                deltaX_address, //deltaX = YrowSum * [exp(X - maxX) / expXm_max_rowSum] - Y
                lengthv, width, stride);
        
        return new Stream2Block3Syncer_1(streamPool, stream1, stream2,
                core, blockV, blockMax, blockY_rowSum);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="Affine">
    //<editor-fold defaultstate="collapsed" desc="BP: affine_row">
    @Override
    public Syncer affine2D(long Y_address,
            long X_address,
            long A_address, long B_address, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long stream = streamPool.getStream();
        Cuda_function.affine2D_row(stream, 
                X_address, 
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer affine2D_deltaA_v1(long deltaA_address, 
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long A_address, long B_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long block[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1];//V[HV: nextN, M: row_lengthv]
        }
        
        Cuda_reduce.field_affine_deltaA_v1(stream,
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer affine2D_deltaAB_v1(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address,//V1: holdY(), Y is not changed
            long Y_address,
            long A_address, long B_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    
    @Override
    public Syncer affine2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1 
            long deltaY_address,
            long X_address,//V2 holdX(), X is not changed
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                X_address, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: sqBatchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer sqBatchNorm2D(long Y_address, 
            long X_address, 
            long X_mean_address, long X_sqmean_address, float eps, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqBatchNorm2D_row(stream, 
                X_address, 
                X_mean_address, 
                X_sqmean_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer sqBatchNorm2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sqBatchNorm_affined2D_row(stream, 
                X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                Y_address,//Y = X_norm
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Override
    public Syncer sqBatchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address, //V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is end
        
        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                deltaXp1, 
                deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //=====[return Syncer]==================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer sqBatchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //=====[return Syncer]==================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer sqBatchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1]; 
        }

        Cuda_reduce.field_sqBatchNorm_deltaA_v2(stream, 
                deltaY_address, 
                dX_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer sqBatchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length); deltaA_buf_address = blockA[1];
            blockB = core.malloc(mem_length); deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2,
                deltaY_address,
                dX_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB)); 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer sqBatchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }

        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    
    @Override
    public Syncer sqBatchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
       
        Cuda_reduce.field_sqBatchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.sqBatchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        
        //======[syncer]========================================================
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: batchNorm">
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    @Override
    public Syncer batchNorm2D(long Y_address, 
            long X_address, 
            long X_mean_address, long X_var_address, float eps, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm2D_row(stream, 
                X_address, 
                X_mean_address, 
                X_var_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer batchNorm2D(long Y_address,
            long X_address, 
            long X_mean_address, long X_var_address, float eps,
            long A_address, long B_address, 
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.batchNorm_affined2D_row(stream, 
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-progataion: deltaX">
    @Override
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v2(stream1, stream2, 
                deltaY_address, 
                Y_address,//Y = X_norm
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * Y
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_deltaX_v1(stream1, 
                deltaY_address, Y_address, 
                X_var_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    { 
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA
                deltaXp1, deltaXp1,//deltaXp1 = deltaB
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_var_address, eps,
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): deltaX">
    @Override
    public Syncer batchNorm2D_deltaX_v1(long deltaX_address,
            long deltaY_address, 
            long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2,
                deltaY_address,
                Y_address,
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * (Y - B) / A
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is end
        
        Cuda_function.batchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                X_var_address, eps,
                A_address, B_address, 
                deltaXp1, 
                deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //=====[return Syncer]==================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer batchNorm2D_deltaX_v2(long deltaX_address,
            long deltaY_address,
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find deltaXp1, deltaXp2]===============================
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]s
        long[] block1 = core.malloc(mem_length); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(mem_length); long deltaXp2 = block2[1];
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address,
                X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv,
                deltaXp2, deltaXp2,//deltaXp2 = deltaA: deltaY * (X - X_mean) * X_rstd
                deltaXp1, deltaXp1,//deltaXp1 = deltaB: deltaY
                width, stride, 1);
        
        //======[stage2: find deltaX]===========================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.batchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, 
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, 
                deltaXp1, 
                deltaXp2, row_lengthv,
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //=====[return Syncer]==================================================
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer batchNorm2D_deltaA_v2(long deltaA_address,
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1]; 
        }

        Cuda_reduce.field_batchNorm_deltaA_v2(stream, 
                deltaY_address, 
                dX_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address,//deltaY * (X - X_mean) * X_rstd
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer batchNorm2D_deltaAB_v2(
            long deltaA_address,//result0
            long deltaB_address,//result1
            long deltaY_address, 
            long dX_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length); deltaA_buf_address = blockA[1];
            blockB = core.malloc(mem_length); deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2,
                deltaY_address,
                dX_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaA_buf_address, deltaA_address, 
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB)); 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward-propagation(affined): {deltaA, deltaB, deltaX}">
    @Override
    public Syncer batchNorm2D_gradients_v1(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, long Y_address,//V1: holdY(), Y is not changed
            long X_var_address, float eps,
            long A_address, long B_address,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }

        Cuda_reduce.field_affine_deltaAB_v1(stream1, stream2, 
                deltaY_address, Y_address, 
                A_address, B_address, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_deltaX_v1(stream1, 
                deltaY_address, 
                Y_address,
                X_var_address, eps,
                A_address, B_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        
        //======[return Syncer]=================================================
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    
    @Override
    public Syncer batchNorm2D_gradients_v2(
            long deltaX_address,//result0
            long deltaA_address,//result1
            long deltaB_address,//result2
            long deltaY_address, long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_var_address, float eps,
            long A_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride)
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //======[stage1: find { deltaXp1, deltaXp2, deltaA, deltaB }]===========
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        long[] block1 = null; long deltaXp1 = 0L;//deltaXp1 = deltaB_buf
        long[] block2 = null; long deltaXp2 = 0L;//deltaXp2 = deltaA_buf
        
        if(nextN > 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); deltaXp1 = block1[1];
            block2 = core.malloc(mem_length); deltaXp2 = block2[1];
        }
       
        Cuda_reduce.field_batchNorm_deltaAB_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_var_address, eps, 
                field_length, row_lengthv, 
                deltaXp2, deltaA_address,//deltaXp2 = deltaA_buf
                deltaXp1, deltaB_address,//deltaXp1 = deltaB_buf
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended

        Cuda_function.batchNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address,
                X_address,
                X_mean_address, X_var_address, eps,
                A_address, 
                deltaB_address,//deltaXp1 = deltaB
                deltaA_address,//deltaXp2 = deltaA
                row_lengthv,
                deltaX_address, 
                lengthv, width, stride);

        Cuda.deleteEvent(event);
        
        //======[syncer]========================================================
        return (nextN == 1?
                new Stream2Syncer_1(streamPool, stream1, stream2)://block1, block are null
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="BP: layerNorm">
    //<editor-fold defaultstate="collapsed" desc="forward propagation">
    @Override
    public Syncer layerNorm2D(long Y_address, 
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.layerNorm2D_row(stream,
                X_address,
                X_mean_address, X_sqmean_address, eps, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer layerNorm2D(long Y_address,
            long X_address,
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, long B_address,
            int row_lengthv, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.layerNorm_affined2D_row(stream,
                X_address,
                X_mean_address, X_sqmean_address, eps,
                A_address, B_address, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation: deltaX">
    @Override
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_deltaXp_v1(stream1, stream2,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm2D_row_deltaX_v1(stream1,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_deltaXp_v2(stream1, stream2, 
                deltaY_address, X_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm2D_row_deltaX_v2(stream1,
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): deltaX">
    @Override
    public Syncer layerNorm2D_deltaX_v1(long deltaX_address, 
            long deltaY_address,
            long Y_address,//V1: holdY(), Y is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            long A_address, long B_address, 
            int field_length, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;//[nextM, SV]
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_affined_deltaXp_v1(stream1, stream2,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm_affined2D_row_deltaX_v1(stream1,
                deltaY_address, Y_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address, B_address,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address,
                lengthv, width, stride);
         
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }

    @Override
    public Syncer layerNorm2D_deltaX_v2(long deltaX_address, 
            long deltaY_address, 
            long X_address,//V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            long A_address, 
            int field_length, int row_lengthv, 
            int lengthv, int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        //stage1: find deltaXp1, deltaXp2=======================================
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        int SV = ((field_length + 3) >> 2) << 2;// SV % 4 == 0
        int V_lengthv = nextM * SV;
        long[] block1 = core.malloc(V_lengthv); long deltaXp1 = block1[1];
        long[] block2 = core.malloc(V_lengthv); long deltaXp2 = block2[1];
        
        Cuda_reduce.row_layernorm_affined_deltaXp_v2(stream1, stream2, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps,
                A_address, 
                field_length, row_lengthv, 
                deltaXp1, deltaXp2,
                width, stride, 1);
        
        //stage2: find deltaX===================================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait stage1 is ended
        
        Cuda_function.layerNorm_affined2D_row_deltaX_v2(stream1, 
                deltaY_address, X_address, 
                X_mean_address, X_sqmean_address, eps, 
                A_address,
                deltaXp1, deltaXp2, row_lengthv, 
                deltaX_address, 
                lengthv, width, stride);
        
        Cuda.deleteEvent(event);
        return new Stream2Block2Syncer_1(streamPool, stream1, stream2,//only sync stream1
                core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="backward propagation(affined): {deltaA, deltaB}">
    @Override
    public Syncer layerNorm2D_deltaA_v2(long deltaA_address, 
            long deltaY_address, long dX_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mem stride)
        long stream = streamPool.getStream();
        long deltaA_buf_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);
            deltaA_buf_address = block[1];//V[HV: nextN, M: row_lengthv]
        }

        Cuda_reduce.field_layerNorm_deltaA_v2(stream,
                deltaY_address, dX_address,
                X_mean_address, 
                X_sqmean_address, eps,
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer layerNorm2D_deltaAB_v2(long deltaA_address, long deltaB_address,
            long deltaY_address, long dX_address, //V2: holdX(), X is not changed
            long X_mean_address, long X_sqmean_address, float eps, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long deltaA_buf_address = 0L; long blockA[] = null;
        long deltaB_buf_address = 0L; long blockB[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            blockA = core.malloc(mem_length);
            blockB = core.malloc(mem_length);
            deltaA_buf_address = blockA[1];
            deltaB_buf_address = blockB[1];
        }
        
        Cuda_reduce.field_layerNorm_deltaAB_v2(stream1, stream2,
                deltaY_address, dX_address,
                X_mean_address, X_sqmean_address, eps, 
                field_length, row_lengthv,
                deltaA_buf_address, deltaA_address,
                deltaB_buf_address, deltaB_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, blockA, blockB));
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="onehot, pix2tensor">
    @Override
    public Syncer onehot2D_row_int8(long Y_address, 
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.onehot2D_row_char(stream, X_address, 
                alpha, beta, row_lengthv,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer onehot2D_row_int32(long Y_address, 
            long X_address, 
            float alpha, float beta, int row_lengthv,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.onehot2D_row_int(stream, X_address,
                alpha, beta, row_lengthv, 
                Y_address, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer pix2tensor2D(long Y_address, 
            long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.pix2tensor2D(stream, X_address,
                Y_address,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Optimizer">
    //<editor-fold defaultstate="collapsed" desc="SGD">
    @Override
    public Syncer sgd2D(long W_address, 
            long[] gradients, float lr, 
            int lengthv, int width, int stride)  
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.quadratic_dual2D(stream, 
                    W_address, gradient,
                    0, 0, 0, 
                    1.0f, -lr, 0,
                    W_address, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SGDMN">
    @Override
    public Syncer sgdmn2D(long W_address, 
            long V_address, float momentum, float dampen, float nesterov, 
            long deltaW_address, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sgdmn2D(stream, W_address, 
                V_address, momentum, dampen, nesterov,
                deltaW_address, lr, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sgdmn2D(long W_address,
            long V_address, float momentum, float dampen, float nesterov, 
            long[] gradients, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.sgdmn2D(stream, W_address,
                    V_address, momentum, dampen, nesterov, 
                    gradient, lr, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SGDMN_decay">
    @Override
    public Syncer sgdmn2D_decay(long W_address,
            long V_address, float momentum, float dampen, float nesterov,
            long deltaW_address, float lr, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.sgdmn2D_decay(stream, W_address, 
                V_address, momentum, dampen, nesterov,
                deltaW_address, lr, 
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sgdmn2D_decay(long W_address, 
            long V_address, float momentum, float dampen, float nesterov,
            long[] gradients, float lr, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.sgdmn2D_decay(stream, W_address,
                    V_address, momentum, dampen, nesterov, 
                    gradient, lr, 
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Moumentum">
    @Override
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.momentum2D(stream,
                W_address, 
                V_address, a1, a2, 
                deltaW_address, lr_t, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer momentum2D(long W_address, 
            long V_address, float a1, float a2, 
            long[] gradients, float lr_t,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.momentum2D(stream,
                    W_address,
                    V_address, a1, a2,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Moumentum_decay">
    @Override
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.momentum2D_decay(stream, 
                W_address, 
                V_address, a1, a2, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer momentum2D_decay(long W_address, 
            long V_address, float a1, float a2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.momentum2D_decay(stream,
                    W_address,
                    V_address, a1, a2,
                    gradient, lr_t,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="RMSprop">
    @Override
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rmsprop2D(stream, 
                W_address, 
                S_address, a1, a2, eps_t, 
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer rmsprop2D(long W_address, 
            long S_address, float a1, float a2, float eps_t,
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.rmsprop2D(stream,
                    W_address,
                    S_address, a1, a2, eps_t,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="RMSprop-decay">
    @Override
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t, 
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.rmsprop2D_decay(stream, 
                W_address, 
                S_address, a1, a2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer rmsprop2D_decay(long W_address, 
            long S_address, float a1, float a2, float eps_t, 
            long[] gradients, float lr_t,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.rmsprop2D_decay(stream,
                    W_address,
                    S_address, a1, a2, eps_t,
                    gradient, lr_t, 
                    L1coef, L2coef, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="Adam">
    @Override
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adam2D(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                deltaW_address, lr_t, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D_type2(long W_address, 
            long V_address, float a1, float a2, float Uv,
            long S_address, float b1, float b2, float eps, float Us,
            long deltaW_address, float lr, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream(); 
        Cuda_function.adam2D_type2(stream, W_address,
                V_address, a1, a2, Uv,
                S_address, b1, b2, eps, Us,
                deltaW_address, lr, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adam2D(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adam-decay">
    @Override
    public Syncer adam2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adam2D_decay(stream, 
                W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                deltaW_address, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adam2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long[] gradients, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.adam2D_decay(stream, 
                W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                gradient, lr_t,
                L1coef, L2coef,
                lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax">
    @Override
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float eps,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamax2D(stream,
                W_address,
                V_address, a1, a2,
                S_address, b1, eps,
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamax2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for(long gradient : gradients) {
            Cuda_function.adamax2D(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, eps,
                    gradient, lr_t,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamax-decay">
    @Override
    public Syncer adamax2D_decay(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps, 
            long deltaW_address, float lr_t,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamax2D_decay(stream,
                W_address, 
                V_address, a1, a2, 
                S_address, b1, eps,
                deltaW_address, lr_t,
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamax2D_decay(long W_address,
            long V_address, float a1, float a2, 
            long S_address, float b1, float eps,
            long[] gradients, float lr_t,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamax2D_decay(stream,
                    W_address,
                    V_address, a1, a2,
                    S_address, b1, eps,
                    gradient, lr_t,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="AdamW">
    @Override
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps, 
            long deltaW_address, float lr_t, float lr,
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_function.adamW2D(stream, W_address, 
                V_address, a1, a2,
                S_address, b1, b2, eps,
                deltaW_address, lr_t, lr,
                L1coef, L2coef,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adamW2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps, 
            long[] gradients, float lr_t, float lr,
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamW2D(stream, W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps,
                    gradient, lr_t, lr,
                    L1coef, L2coef,
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod">
    @Override
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adamod2D(stream, W_address, 
                V_address, a1, a2, 
                S_address, b1, b2, eps_t,
                G_address, c1, c2, 
                deltaW_address, lr_t,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer adamod2D(long W_address, 
            long V_address, float a1, float a2, 
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2, 
            long[] gradients, float lr_t, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        for (long gradient : gradients) {
            Cuda_function.adamod2D(stream, W_address, 
                    V_address, a1, a2, 
                    S_address, b1, b2, eps_t,
                    G_address, c1, c2,
                    gradient, lr_t, 
                    lengthv, width, stride);
        }
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Adamod-decay">
     @Override
    public Syncer adamod2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t,
            long G_address, float c1, float c2,
            long deltaW_address, float lr_t, 
            float L1coef, float L2coef, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        Cuda_function.adamod2D_decay(stream, W_address,
                V_address, a1, a2, 
                S_address, b1, b2, eps_t, 
                G_address, c1, c2, 
                deltaW_address, lr_t, 
                L1coef, L2coef, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer adamod2D_decay(long W_address,
            long V_address, float a1, float a2,
            long S_address, float b1, float b2, float eps_t, 
            long G_address, float c1, float c2,
            long[] gradients, float lr_t, 
            float L1coef, float L2coef,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();      
        for(long gradient : gradients) {
            Cuda_function.adamod2D_decay(stream, W_address,
                    V_address, a1, a2,
                    S_address, b1, b2, eps_t, 
                    G_address, c1, c2, 
                    gradient, lr_t, 
                    L1coef, L2coef, 
                    lengthv, width, stride);
        } 
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Random Function">
    @Override
    public Syncer bernouli2D(long X_address, 
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.bernouli2D(stream,
                X_address, 
                seed,
                p, v1, v2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer bernouli_mul2D(long Y_address, long R_address, 
            long X_address,
            int seed,
            float p, float v1, float v2,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.bernouli_mul2D(stream,
                X_address, R_address,
                Y_address,
                seed,
                p, v1, v2, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    
    @Override
    public Syncer uniform2D(long X_address,
            int seed, 
            float vmin, float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.uniform2D(stream,
                X_address,
                seed, 
                vmin, vmax,
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sparse_uniform2D(long X_address, 
            int seed1, int seed2,
            float p, float vmin, float vmax,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.sparse_uniform2D(stream,
                X_address, 
                seed1, seed2, 
                p, vmin, vmax, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer gaussian2D(long X_address, 
            int seed1, int seed2,
            float mu, float sigma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.gaussian2D(stream, 
                X_address, 
                seed1, seed2, 
                mu, sigma, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }

    @Override
    public Syncer sparse_gaussian2D(long X_address,
            int seed1, int seed2, int seed3,
            float p, float mu, float sigma, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        Cuda_random.sparse_gaussian2D(stream, 
                X_address, 
                seed1, seed2, seed3, 
                p, mu, sigma, 
                lengthv, width, stride);
        return new StreamSyncer(streamPool, stream);
    }
    //</editor-fold>
 
    //<editor-fold defaultstate="collapsed" desc="Reduce Function">
    //<editor-fold defaultstate="collapsed" desc="straight reduce function">
    @Override
    public Result<Float> straight_linear(long X_address, 
            float alpha, float beta, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_linear(stream, 
                X_address, alpha, beta, lengthv,
                block[1],//V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
    @Override
    public Result<Float> straight_quadratic(long X_address,
            float alpha, float beta, float gamma,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_quadratic(stream, 
                X_address, alpha, beta, gamma, lengthv, 
                block[1],//V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
     @Override
    public Result<Float> straight_max(long X_address, 
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_max(stream,
                X_address, lengthv,
                block[1], //V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }

    @Override
    public Result<Float> straight_min(long X_address,
            int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block = core.malloc(nextLengthV);
        
        Cuda_reduce.straight_min(stream,
                X_address, lengthv,
                block[1], //V_address = block[1]
                width, stride, 1);
        return new StreamBlockResult(streamPool, stream, core, block);
    }
    
    @Override
    public IndexedResult<Float> straight_max_indexed(long X_address, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block1 = core.malloc(nextLengthV);
        long[] block2 = core.malloc_int32(nextLengthV);
        
        Cuda_reduce.straight_max_indexed(stream, 
                X_address, lengthv, 
                block1[1], //V_address = block1[1]
                block2[1], //Index_address = block2[1]
                width, stride, 1);
        
        return new FloatIndexedResult(streamPool, stream, core, block1, block2);
    }

    @Override
    public IndexedResult<Float> straight_min_indexed(long X_address, int lengthv, int width, int stride) 
    {
        long stream = streamPool.getStream();
        int nextLengthV = Cuda_reduce.straight_nextLengthV(lengthv);
        long[] block1 = core.malloc(nextLengthV);
        long[] block2 = core.malloc_int32(nextLengthV);
        
        Cuda_reduce.straight_min_indexed(stream, 
                X_address, lengthv,
                block1[1], //V_address = block1[1]
                block2[1], //Index_address = block2[1]
                width, stride, 1);
        
        return new FloatIndexedResult(streamPool, stream, core, block1, block2);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field reduce function">
    //<editor-fold defaultstate="collapsed" desc="field_linear">
    @Override
    public Syncer field_linear(long Y_address,
            long X_address,
            float alpha, float beta,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_linear(stream, 
                X_address, alpha, beta,
                field_length, row_lengthv,
                V_address, Y_address, //V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_linear_dual(long Y_address, 
            long X1_address, long X2_address,
            float alpha, float beta, float gamma, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_linear_dual(stream, 
                X1_address, X2_address, 
                alpha, beta, gamma,
                field_length, row_lengthv, 
                V_address, Y_address,
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_quadratic">
    @Override
    public Syncer field_quadratic(long Y_address,
            long X_address,
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_quadratic(stream, 
                X_address, alpha, beta, gamma, 
                field_length, row_lengthv,
                V_address, Y_address, //V_address = block[1]
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_quadratic_dual(long Y_address, 
            long X1_address, long X2_address,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_quadratic_dual(stream,
                X1_address, X2_address,
                k11, k12, k22, 
                k1, k2, C, 
                field_length, row_lengthv,
                V_address, Y_address,
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_linear_quadratic & var & std">
    @Override
    public Syncer field_linear_quadratic(long Y1_address, long Y2_address,
            long X_address, 
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];  
        long V1_address = 0L; long block1[] = null;
        long V2_address = 0L; long block2[] = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length);
            block2 = core.malloc(mem_length);
            V1_address = block1[1];
            V2_address = block2[1];
        }
        
        Cuda_reduce.field_linear_quadratic(stream1, stream2,
                X_address,
                alpha1, beta1, 
                alpha2, beta2, gamma2,
                field_length, row_lengthv,
                V1_address, Y1_address,//V1 = Y1.buf
                V2_address, Y2_address,//V2 = Y2.buf
                width, stride, 1);
        
        return (nextN == 1?
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, block1, block2));
    }
    
    private boolean field_var_f64 = false;
    public boolean field_var_f64() { return field_var_f64; }
    public CudaFloat32EngineBase field_var_f64(boolean flag) { field_var_f64 = flag; return this; }
    
    @Override
    public Syncer field_var(long X_var_address, 
            long X_mean_address, 
            long X_sqmean_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
                
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); V1_address = block1[1];
            block2 = core.malloc(mem_length); V2_address = block2[1];
        }
        
        //staget1: compute mean = E(X) and squareMean = E(X^2)==================
        float alpha = (float) (1.0 / field_length);
        Cuda_reduce.field_linear_quadratic(stream1, stream2, 
                X_address,
                alpha, 0.0f, //mean = sum_each_field: X / field_length
                alpha, 0.0f, 0.0f, //squareMean = sum_each_field: X^2 / field_length
                field_length, row_lengthv, 
                V1_address, X_mean_address,//V1 = mean.buf
                V2_address, X_sqmean_address,//V2 = squareMean.buf
                width, stride, 1);
        
        //stage2 compute var = E(X^2) - E(X)^2==================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
       
        //mean[row_lengthv], squareMean[row_lengthv], var[row_lengthv]
        if(field_var_f64) {//var = sqmean -mean^2
            Cuda_function.variance2D_f64(stream1,
                    X_mean_address,//(double)
                    X_sqmean_address,
                    X_var_address,
                    row_lengthv, width, stride);
        }
        else {//var = -mean^2 + sqmean
            Cuda_function.quadratic_dual2D(stream1, 
                    X_mean_address, X_sqmean_address, 
                    -1.0f, 0.0f, 0.0f,  
                    0.0f, 1.0f, 0.0f,
                    X_var_address, 
                    row_lengthv, width, stride);
        }
        return (nextN == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer field_std(long X_std_address, 
            long X_mean_address, 
            long X_sqmean_address,
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
                
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;//V[HV: nextN, M: row_lengthv]
            block1 = core.malloc(mem_length); V1_address = block1[1];
            block2 = core.malloc(mem_length); V2_address = block2[1];
        }

        //staget1: compute mean = E(X) and squareMean = E(X^2)==================
        float alpha = (float) (1.0 / field_length);
        Cuda_reduce.field_linear_quadratic(stream1, stream2, 
                X_address,
                alpha, 0.0f,//mean = sum_each_field: X / field_length
                alpha, 0.0f, 0.0f,//squareMean = sum_each_field: X^2 / field_length
                field_length, row_lengthv,
                V1_address, X_mean_address, 
                V2_address, X_sqmean_address, 
                width, stride, 1);
        
        //stage2 compute stddev = sqrt(E(X^2) - E(X)^2)=========================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, streamArray, 2);
        Cuda.streamWaitEvent_default(stream1, event);//wait for stream1 and stream2
       
        //mean[row_lengthv], squareMean[row_lengthv], var[row_lengthv]
        Cuda_function.sqrt_quadratic_dual2D(stream1, 
                X_mean_address, X_sqmean_address, //std = sqrt(-mean^2 + squareMean)
                -1.0f, 0.0f, 0.0f,  
                0.0f, 1.0f, 0.0f,
                X_std_address, 
                row_lengthv, width, stride);
        
        return (nextN == 1? 
                new Stream2Syncer_1(streamPool, stream1, stream2):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="field_max, min">
    @Override
    public Syncer field_max(long Y_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride)
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_max(stream, 
                X_address,
                field_length, row_lengthv, 
                V_address, Y_address,
                width, stride, 1);
        
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer field_min(long Y_address,
            long X_address,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            block = core.malloc(nextN * row_lengthv);//V[HV: nextN, M: row_lengthv]
            V_address = block[1];
        }
        
        Cuda_reduce.field_min(stream, 
                X_address,
                field_length, row_lengthv, 
                V_address, Y_address,
                width, stride, 1);
       
        return (nextN == 1? 
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer field_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L; 
        long[] block1 = null, block2 = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;
            block1 = core.malloc(mem_length);//V[HV: nextN, M: row_lengthv]
            block2 = core.malloc_int32(mem_length);
            V_address      = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.field_max_indexed(stream,
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address,
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }

    @Override
    public Syncer field_min_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L; 
        long[] block1 = null, block2 = null;
        
        int nextN = Cuda_reduce.field_nextN(field_length, row_lengthv);
        if(nextN != 1) {
            int mem_length = nextN * row_lengthv;
            block1 = core.malloc(mem_length);//V[HV: nextN, M: row_lengthv]
            block2 = core.malloc_int32(mem_length);
            V_address      = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.field_min_indexed(stream,
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address, 
                Y_address, Index_address,
                width, stride, 1);
     
        return (nextN == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row reduce function">
    //<editor-fold defaultstate="collapsed" desc="row_linear">
    @Override
    public Syncer row_linear(long Y_address,
            long X_address,
            float alpha, float beta,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_linear(stream,
                X_address, alpha, beta,
                field_length, row_lengthv, 
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer row_linear_dual(long Y_address,
            long X1_address, long X2_address, 
            float alpha, float beta, float gamma,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }

        Cuda_reduce.row_linear_dual(stream, 
                X1_address, X2_address, 
                alpha, beta, gamma, 
                field_length, row_lengthv,
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_quadratic">
    @Override
    public Syncer row_quadratic(long Y_address, 
            long X_address,
            float alpha, float beta, float gamma, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
       //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_quadratic(stream, 
                X_address, alpha, beta, gamma,
                field_length, row_lengthv,
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer row_quadratic_dual(long Y_address, 
            long X1_address, long X2_address, 
            float k11, float k12, float k22, 
            float k1, float k2, float C, 
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_quadratic_dual(stream, 
                X1_address, X2_address, 
                k11, k12, k22, 
                k1, k2, C,
                field_length, row_lengthv,
                V_address, Y_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_linear_quadratic & var & stddev">
    @Override
    public Syncer row_linear_quadratic(long Y1_address, long Y2_address, 
            long X_address,
            float alpha1, float beta1,
            float alpha2, float beta2, float gamma2,
            int field_length, int row_lengthv, 
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long[] streamArray = streamPool.getStreamArray(2);
        long stream1 = streamArray[0];
        long stream2 = streamArray[1];
        
        long V1_address = 0L; long[] block1 = null;
        long V2_address = 0L; long[] block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc(V_lengthv);
            V1_address = block1[1];
            V2_address = block2[1];
        }
         
        Cuda_reduce.row_linear_quadratic(stream1, stream2, 
                X_address, 
                alpha1, beta1,//Y1 = alpha1*X + beta1
                alpha2, beta2, gamma2,//Y2 = alpha2*X^2 + beta2*X + gamma2
                field_length, row_lengthv,
                V1_address, Y1_address,
                V2_address, Y2_address, 
                width, stride, 1);
        
        return (nextM == 1? 
                new StreamArraySyncer(streamPool, streamArray):
                new StreamArrayBlock2Syncer(streamPool, streamArray, core, block1, block2));
    }

    @Override
    public Syncer row_var(long var_address, 
            long mean_address, long squareMean_address, 
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream1 = streamPool.getStream();
        long stream2 = streamPool.getStream();
        long V1_address = 0L, V2_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc(V_lengthv);
            V1_address = block1[1];//V[nextM, N = field_length]
            V2_address = block2[1];//V[nextM, N = field_length]
        }
        
        //staget1: compute mean = E(X) and squareMean = E(X^2)==================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear(stream1, 
                X_address, alpha, 0.0f,//mean = sum_each_row(X)/row_length
                field_length, row_lengthv, 
                V1_address, mean_address, 
                width, stride, 1);
        
        Cuda_reduce.row_quadratic(stream2, 
                X_address, alpha, 0.0f, 0.0f,
                field_length, row_lengthv, 
                V2_address, squareMean_address,//squareMean = sum_each_row(X^2)/row_length
                width, stride, 1);
        
        //stage2 compute var = E(X^2) - E(X)^2==================================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, stream1);
        Cuda.eventRecord(event, stream2);
        Cuda.streamWaitEvent_default(stream1, event);
        
        //mean[field_lengthv], squareMean[field_lengthv], var[field_lengthv]
        Cuda_function.quadratic_dual2D(stream1, 
                mean_address, squareMean_address, //-mean^2 + squareMean
                -1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 
                var_address,
                field_lengthv, field_length, field_lengthv);
        
        return (nextM == 1? 
                new StreamSyncer(streamPool, stream1):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }

    @Override
    public Syncer row_stddev(long stddev_address,
            long mean_address, long squareMean_address,
            long X_address, 
            int field_length, int field_lengthv,
            int row_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream1 = streamPool.getStream();
        long stream2 = streamPool.getStream();
        long V1_address = 0L, V2_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
         if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc(V_lengthv);
            V1_address = block1[1];//V[nextM, N = field_length]
            V2_address = block2[1];//V[nextM, N = field_length]
        }
        
        //staget1: compute mean = E(X) and squareMean = E(X^2)==================
        float alpha = 1.0f / row_length;
        Cuda_reduce.row_linear(stream1, 
                X_address, alpha, 0.0f,//mean = sum_each_row(X)/row_length
                field_length, row_lengthv, 
                V1_address, mean_address, 
                width, stride, 1);
        
        Cuda_reduce.row_quadratic(stream2, 
                X_address, alpha, 0.0f, 0.0f,
                field_length, row_lengthv, 
                V2_address, squareMean_address,//squareMean = sum_each_row(X^2)/row_length
                width, stride, 1);
        
        //stage2 compute stddev = sqrt(E(X^2) - E(X)^2)=========================
        long event = Cuda.newEvent_DisableTiming();
        Cuda.eventRecord(event, stream1);
        Cuda.eventRecord(event, stream2);
        Cuda.streamWaitEvent_default(stream1, event);
        
        //mean[field_lengthv], squareMean[field_lengthv], var[field_lengthv]
        Cuda_function.sqrt_quadratic_dual2D(stream1,
                mean_address, squareMean_address,
                -1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 
                stddev_address,
                field_lengthv, field_length, field_lengthv);
        
        return (nextM == 1? 
                new StreamSyncer(streamPool, stream1):
                new Stream2Block2Syncer_1(streamPool, stream1, stream2, core, block1, block2));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="row_max, max">
    @Override
    public Syncer row_max(long Y_address, 
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_max(stream, 
                X_address, 
                field_length, row_lengthv,
                V_address, Y_address, 
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }

    @Override
    public Syncer row_min(long Y_address, 
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L; long[] block = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block = core.malloc(V_lengthv);
            V_address = block[1];
        }
        
        Cuda_reduce.row_min(stream, 
                X_address, 
                field_length, row_lengthv,
                V_address, Y_address, 
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):
                new StreamBlockSyncer(streamPool, stream, core, block));
    }
    
    @Override
    public Syncer row_max_indexed(long Y_address, long Index_address,
            long X_address, 
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc_int32(V_lengthv);
            V_address     = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.row_max_indexed(stream,
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address,
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):   
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }

    @Override
    public Syncer row_min_indexed(long Y_address, long Index_address,
            long X_address,
            int field_length, int row_lengthv,
            int width, int stride) 
    {
        //N = field_lenth, M = row_lengthv(mod stride)
        long stream = streamPool.getStream();
        long V_address = 0L, VIndex_address = 0L;
        long[] block1 = null, block2 = null;
        
        int nextM = Cuda_reduce.row_nextM(row_lengthv);
        if(nextM != 1) {//at least 2 stages: tansposed: //X[N, M] -> V[nextM, SV]
            int SV = ((field_length + 3) >> 2) << 2;
            int V_lengthv = nextM * SV;//V[nextM, N = field_length]
            block1 = core.malloc(V_lengthv);
            block2 = core.malloc_int32(V_lengthv);
            V_address     = block1[1];
            VIndex_address = block2[1];
        }
        
        Cuda_reduce.row_min_indexed(stream, 
                X_address, 
                field_length, row_lengthv, 
                V_address, VIndex_address, 
                Y_address, Index_address,
                width, stride, 1);
        
        return (nextM == 1?
                new StreamSyncer(streamPool, stream):   
                new StreamBlock2Syncer(streamPool, stream, core, block1, block2));
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
}

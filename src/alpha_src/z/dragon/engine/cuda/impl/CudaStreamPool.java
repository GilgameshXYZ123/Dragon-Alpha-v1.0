/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

/**
 *
 * @author Gilgamesh
 */
public abstract class CudaStreamPool 
{
    protected final CudaDevice dev;
    protected int max_streamsize;
    protected int max_getsize_oneturn;
    
    public CudaStreamPool(CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn) {
        if(dev == null) throw new NullPointerException("CudaDevice is null");
        if(maxStreamSize <= 0)throw new IllegalArgumentException("maxStreamSize must > 0");
        if(maxGetSizeOneTurn <= 0) throw new IllegalArgumentException("maxGetSizeOneTurn > 0");
        if(maxGetSizeOneTurn > maxStreamSize) throw new IllegalArgumentException("maxGetSizeOneTurn < maxStreamSize");
        
        this.dev = dev;
        this.max_streamsize = maxStreamSize;
        this.max_getsize_oneturn = maxGetSizeOneTurn;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public CudaDevice device() {  return dev; }
    
    public final int max_stream_size() { return max_streamsize; }
    public abstract int total_stream_size();
    
    public abstract long[] stream_addrs();
    
    public synchronized void setMaxPoolSize(int maxStreamSize) {
        if(maxStreamSize <= 0)throw new IllegalArgumentException(
                "maxStreamSize must be positive");
        if(this.max_getsize_oneturn > maxStreamSize) throw new IllegalArgumentException(
                "maxGetSizeOneTurn is less than maxStreamSize");
        this.max_streamsize = maxStreamSize;
    }
    
    public synchronized void setMaxGetSizeOneTurn(int maxGetSizeOneTurn) {
        if(maxGetSizeOneTurn <= 0) throw new IllegalArgumentException(
                "maxGetSizeOneTurn must be positive");
        if(maxGetSizeOneTurn > this.max_streamsize) throw new IllegalArgumentException(
                "maxGetSizeOneTurn is less than maxStreamSize");
        this.max_getsize_oneturn = maxGetSizeOneTurn;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        this.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    public abstract long getStream() ;
    public abstract long[] getStreamArray(int length);
    
    public abstract void returnStream(long stream);
    public abstract void returnStreamArray(long[] streams);
    
    protected abstract void __clear__();
    public synchronized void clear() {
        long[] streams = stream_addrs();
        Cuda.deleteStream(streams, streams.length);
        __clear__();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="interface: CudaStreamPoolFactory">
    public static interface CudaStreamPoolFactory {
        public abstract CudaStreamPool create(CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn);
    }
    
    private static CudaStreamPoolFactory default_factory = (CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn)
            -> { return new CudaStreamp1(dev, maxStreamSize, maxGetSizeOneTurn); };
    
    
    public static void default_factory(CudaStreamPoolFactory factory) { 
        if(factory == null) throw new RuntimeException();
        synchronized(CudaStreamPool.class) { 
            default_factory = factory; 
        }
    }
    
    public static CudaStreamPool instance(CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn) { 
        synchronized(CudaStreamPool.class) {
            return default_factory.create(dev, maxStreamSize, maxGetSizeOneTurn);
        }
    }
    //</editor-fold>
}

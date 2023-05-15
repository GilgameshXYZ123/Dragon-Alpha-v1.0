/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import z.dragon.common.int64.ArrayList_int64;

/**
 * Hashed Simple StreamPool.
 * @author Gilgamesh
 */
public class CudaStreamp1 extends CudaStreamPool
{
    protected ArrayList_int64 gc;//size = gc.size() //how much stream has been created
    protected int gc_index = 0;
    private boolean inited = false;
    
    public CudaStreamp1(CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn) {
        super(dev, maxStreamSize,  maxGetSizeOneTurn);
        this.gc = new ArrayList_int64();
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    @Override public long[] stream_addrs() { return gc.toArray_int64(); }
    @Override public int total_stream_size() { return gc.size(); }
    
    public ArrayList_int64 gc() { return gc; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    private void init() {
        synchronized(Cuda.class) {
            Cuda.setDevice(dev.id);
            for(int i=0; i < max_streamsize; i++) 
                gc.add(Cuda.newStream_Blocking());
        }
    }
    
    @Override
    public synchronized long getStream() { 
        if(!inited) { init(); inited = true;}
        long stream = gc.get(gc_index);
        gc_index = (gc_index + 1) % max_streamsize;
        return stream;
    }
    
    @Override
    public synchronized long[] getStreamArray(int length) {
        if(!inited) { init(); inited = true; }
        //max_getsize_oneturn <= max_poolsize
        if(length > max_getsize_oneturn) length = max_getsize_oneturn;
        long[] streams = new long[length];
        
        for(int index= 0; index < length; index ++){
            streams[index] = gc.get(gc_index);
            gc_index = (gc_index + 1) % max_streamsize;
        }
        
        return streams;
    }
    
    @Override public void returnStream(long stream) {}
    @Override public void returnStreamArray(long[] streams) {}
    
    @Override
    protected void __clear__() {
        long[] streams = gc.toArray_int64();
        Cuda.deleteStream(streams, streams.length);
        gc.clear(); 
    }
    //</editor-fold>
}

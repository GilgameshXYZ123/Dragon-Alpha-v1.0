/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import java.util.TreeSet;
import z.dragon.common.int64.ArrayList_int64;

/**
 * Hashed Simple StreamPool with available set.
 * @author Gilgamesh
 */
public class CudaStreamp2 extends CudaStreamPool
{
    protected TreeSet<Long> available;
    protected ArrayList_int64 gc;//size = gc.size() //how much stream has been created
    protected int gc_index = 0;
    
    public CudaStreamp2(CudaDevice dev, int maxStreamSize) { this(dev, maxStreamSize, 16); }
    public CudaStreamp2(CudaDevice dev, int maxStreamSize, int maxGetSizeOneTurn) {
        super(dev, maxStreamSize, maxGetSizeOneTurn);
        this.available = new TreeSet<>();
        this.gc = new ArrayList_int64();
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    @Override public int total_stream_size() { return gc.size();}
    public int available_stream_size() { return available.size(); }
    public int used_stream_size() { return total_stream_size() - available_stream_size(); }
    
    @Override public long[] stream_addrs() { return gc.toArray_int64(); }
    
    public TreeSet<Long> available() { return available; }
    public ArrayList_int64 gc() { return gc; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    @Override
    public synchronized long getStream() { 
        if(!available.isEmpty()) return available.pollFirst();
        if(gc.size() < max_streamsize) {
            long stream;
            synchronized(Cuda.class) {
                Cuda.setDevice(dev.id); 
                stream = Cuda.newStream_Blocking(); 
            }
            gc.add(stream);
            return stream;
        }
        
        long stream = gc.get(gc_index);
        gc_index = (gc_index + 1) % max_streamsize;
        return stream;
    }
    
    @Override
    public synchronized long[] getStreamArray(int length) {
        //max_getsize_oneturn <= max_poolsize
        if(length > max_getsize_oneturn) length = max_getsize_oneturn;
        long[] streams = new long[length];
        
        int pool_size = available.size();
        if(pool_size >= length) {
            for(int i=0; i<length; i++) streams[i] = available.pollFirst();
            return streams;
        }
        
        //stream can be borrowed from pool: pool.size()
        //stream can be created: maxsize - gc.size
        //the maximum num of stream can be obtained from StreamPool: pool.size + (maxsize - gc.size)
        int gc_size = gc.size();
        if(pool_size + (max_streamsize - gc_size) >= length) {
            for(int i=0; i<pool_size; i++) streams[i] = available.pollFirst();
            synchronized(Cuda.class) {
                Cuda.setDevice(dev.id);
                for(int i = pool_size; i<length; i++)
                    gc.add(streams[i] = Cuda.newStream_Blocking());
            }
            return streams;
        }
        
        int index = 0;
        while(index < pool_size) streams[index++] = available.pollFirst();
        if(gc_size < max_streamsize) {
            synchronized(Cuda.class) {
                Cuda.setDevice(dev.id);
                while(gc_size++ < max_streamsize) 
                    gc.add(streams[index++] = Cuda.newStream_Blocking());
            }
        }
        
        while(index < length) {//gc.sizee() == max_poolsize
            streams[index++] = gc.get(gc_index);
            gc_index = (gc_index + 1) % max_streamsize;
        }
        return streams;
    }
    
    @Override
    public synchronized void returnStream(long stream) {
        available.add(stream);
    }
    
    @Override
    public synchronized void returnStreamArray(long[] streams) {
        for(long stream : streams) available.add(stream);
    }
    
    @Override
    protected void __clear__() {
        gc.clear(); 
        available.clear();
    }
    //</editor-fold>
}

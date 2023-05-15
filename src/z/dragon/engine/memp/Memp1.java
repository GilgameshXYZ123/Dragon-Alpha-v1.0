/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.memp;

import z.dragon.common.addrp.Addrpool;
import z.dragon.common.int64.ArrayList_int64;

/**
 * Mark1: Naive Memory Pool
 * @author Gilgamesh
 */
public class Memp1 extends Mempool
{
    public Memp1(long maxMemorySize) {
        super(maxMemorySize);
    }
    
    //<editor-fold defaultstate="collapsed" desc="member params && Basic-Functions">
    protected final Addrpool pool = Addrpool.instance();
    public Addrpool addrpool() { return pool; }
    
    @Override public int physical_block_num() { return gc.size(); }
    @Override public int buffered_block_num() { return pool.size(); }
    
    protected final ArrayList_int64 gc = new ArrayList_int64(64);//garbage collector
    
    public ArrayList_int64 gc() { return gc; }
    @Override public long[] physical_addrs() { return gc.toArray_int64(); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: alloc"> 
    @Override
    protected long[] search(long mem_size) {
        return pool.remove_section(mem_size, (long) (mem_size * 1.5));
    }

    @Override
    protected boolean alloc_free(long mem_size, long max_alloc_size) {
        long[] max_block = pool.remove_max();//max_block_size < mem_size
        long max_block_size = max_block[0];
        long max_block_address = max_block[1];
        
        this.__free(max_block_size, max_block_address);
        gc.remove(max_block_address);
        max_alloc_size += max_block_size;
            
        while((mem_size > max_alloc_size) && (!pool.isEmpty())) 
        {
            long dst_mem_size = mem_size - max_alloc_size;
            long[] dst_block = pool.remove_ceiling(dst_mem_size);
            if(dst_block == null) dst_block = pool.remove_max();
            
            long dst_block_size = dst_block[0];
            long dst_block_address = dst_block[1];
            
            this.__free(dst_block_size, dst_block_address);
            gc.remove(dst_block_address);
            max_alloc_size += dst_block_size;
        }    
        
        return (mem_size <= max_alloc_size);//true if mem_size <= max_alloc_size
    }
    
    @Override
    protected long[] alloc(final long mem_size)  {
        long mem_address = this.__malloc(mem_size);
        gc.add(mem_address);//register new mem_block to gc
        return new long[]{ mem_size, mem_address };
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: free & clear"> 
    @Override
    protected boolean recycle(long mem_size, long mem_address) {
        pool.put(mem_size, mem_address);
        return true;
    }
    
    @Override
    protected void __clear__() {
        pool.clear(); 
        gc.clear();
    }
    //</editor-fold>
}

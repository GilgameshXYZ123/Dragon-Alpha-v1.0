/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import java.util.HashSet;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.common.MemStatus;
import z.dragon.common.int64.ArrayList_int64;
import static z.dragon.engine.EngineCore.NULL;
import z.dragon.common.addrp.Addrp2;
import z.dragon.common.addrp.Addrpool;
import z.dragon.data.EntryList;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
public class PinnedMempool implements MemStatus
{
    public static final long MEM_1GB = alpha.MEM_1GB;
    public static final long MEM_1MB = alpha.MEM_1MB;
    
    public PinnedMempool(long maxMemorySize) {
        this.max_mem_size = maxMemorySize;
    }
    
    //<editor-fold defaultstate="collapsed" desc="member params">
    protected long max_mem_size;//max_mem_size = max_mem_length * sizeof_datatype
    protected long total_mem_size = 0;//the length of memory has been allocated
    protected long used_mem_size = 0;//the length of memory has been used
    
    //return {block_length, block_address}
    //Key: length of memory; memsize >= (Tensor.length_4x<<2L) [sizeof(float)]
    //Value: address of memory;
    //Integer: javaArray.length <= Integer.MAX_VALUE
    protected final Addrpool pool = new Addrp2();
    protected HashSet<Long> grave = new HashSet<>();
    protected ArrayList_int64 gc = new ArrayList_int64(64);//garbage collector
    //</editor-fold>
 
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Addrpool pool(){ return pool; }
    public ArrayList_int64 gc() { return gc; }
    
    public long[] physical_addrs() { return gc.toArray_int64(); }
    
    @Override public final long max_mem_size() { return max_mem_size; }
    @Override public final long total_mem_size() { return total_mem_size; }
    @Override public final long used_mem_size() { return used_mem_size; }
    
    public int physical_block_num() { return gc.size(); }
    public int buffered_block_num() { return pool.size(); }
    
    public EntryList<String, Object> meta_data() {
        EntryList<String, Object> mp = new EntryList<>();
        mp.put("max_mem_size(MB)", max_mem_size_MB());
        mp.put("total_mem_size(MB)", total_mem_size_MB());
        mp.put("used_mem_size(MB, percent)", "[" + used_mem_size_MB() + ", " + used_mem_percent() + "]");
        mp.put("buffered_mem_size(MB, percent)", "[" + buffered_mem_size_MB() + ", " + buffered_mem_percent() + "]");
        mp.put("buffered_block_num", buffered_block_num());
        mp.put("physical_block_num", physical_block_num());
        return mp;
    }
        
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append("{ ");
        sb.append("\nmax_mem_size(MB) = ").append(max_mem_size_MB());
        sb.append("\ntotal_mem_size(MB) = ").append(total_mem_size_MB());
        sb.append("\nused_mem_size(MB) = ").append(used_mem_size_MB());
        sb.append(" }");
    }
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }
    
    @Override
    public void finalize() throws Throwable {
        super.finalize();
        this.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others"> 
    protected void memory_insufficient() {
        throw new RuntimeException(String.format(
                "[Error] Memory<%d> is insufficient. Try to alloc more",
                total_mem_size_MB()));
    }
    
    protected final void __free(long mem_size, long mem_address) {
        if(mem_address == NULL) return;
        Cuda.freeHost(mem_address);
        total_mem_size -= mem_size;
    }
    
    protected final long __malloc(long mem_size) {
        long address = Cuda.mallocHost(mem_size);
        if(address == NULL) throw new RuntimeException(
                "Cuda_malloc_host: failed to alloc pinned-memory for [mem_size = " + mem_size + " ]");
        total_mem_size += mem_size; 
        return address;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: malloc">
    protected void alloc_wait(long mem_size) {//try to gc some Tensor
        int time = 0;
        try {//max_avaiable_size = max_alloc_size + buffered_size
            while((time < 10) && (mem_size > (max_mem_size - used_mem_size))) {
                this.notifyAll(); this.wait(1000); time++;
            }
        }
        catch(InterruptedException e) {}
        if(time >= 10) memory_insufficient();
    }
    
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
    
    protected long[] alloc(final long mem_size) {//alloc new memory
        long mem_address = this.__malloc(mem_size);
        gc.add(mem_address);//register new mem_block to gc
        return new long[]{ mem_size, mem_address };
    }
    
    @Passed
    public synchronized long[] malloc(long mem_size) //mem_size = mem_length * sizeof_datatype
    {
        mem_size = (mem_size + 3) >> 2 << 2;
        
        long[] block = pool.remove_section(mem_size, (long) (mem_size * 1.5));
        if(block == null) {//max_available_size >= max_alloc_size
            long max_available_size = max_mem_size - used_mem_size;
            if(mem_size > max_available_size) alloc_wait(mem_size);
            
            long max_alloc_size = max_mem_size - total_mem_size;
            if(mem_size > max_alloc_size) {//insufficient mem_size to alloc new mem_block
                if(mem_size > buffered_mem_size() || pool.isEmpty()) memory_insufficient();
                if(!alloc_free(mem_size, max_alloc_size)) memory_insufficient();
            }
            
            //to malloc new blocks, we have: mem_length <= max_alloc_length
            if((block = alloc(mem_size)) == null) memory_insufficient();
        }
        
        used_mem_size += block[0];//block_size
        grave.remove(block[1]);//block_address, rise again
        
        //----------------------------------------------------------------------
        block[0] >>= 2;//block is produced by alloc: block[0] = mem_size, so mem_size % 4 == 0;
        //----------------------------------------------------------------------
        
        return block;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: free">
    protected boolean recycle(long mem_size, long mem_address) {//free: return tree if succeed to free
        pool.put(mem_size, mem_address);
        return true;
    }
    
    @Passed
    public synchronized boolean free(long mem_size, long mem_address) {
        //----------------------------------------------------------------------
        mem_size <<= 2;//block[0] = mem_size, recover
        //----------------------------------------------------------------------
        
        //you can't delete an NULL mem_address, or delete an mem_address twice
        if(mem_address == NULL || grave.contains(mem_address)) return false;
        
        boolean flag = recycle(mem_size, mem_address);
        
        grave.add(mem_address);
        used_mem_size -= mem_size;
        
        return flag;
    }
    
    protected void __clear__() {
        pool.clear();
        gc.clear();
    }
    
    public synchronized void clear() {
        for(long mem_address : physical_addrs()) {
            if(mem_address != NULL) Cuda.freeHost(mem_address);
        }
        
        grave.clear();
        total_mem_size = 0;
        used_mem_size = 0;
        
        __clear__();
    }
    //</editor-fold>
}

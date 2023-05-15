/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.memp;

import java.util.HashSet;
import z.dragon.common.MemStatus;
import z.dragon.data.EntryList;
import z.dragon.engine.EngineBase;
import static z.dragon.engine.EngineCore.NULL;

/**
 * <pre>
 * Memory Pool.
 * (1) malloc: return  block { block_size, block_address }
 * (2) block_length * sizeof_of_datatype = mem_length
 * (3) block_address = mem_address
 * (4) Integer.MAX_VALUE >= block_length
 * for speed: we must padding mem_length to mutiple of 4
 * </pre>
 * @author Gilgameshf
 */
public abstract class Mempool implements MemStatus
{
    public Mempool(long max_mem_size) {
        this.max_mem_size = max_mem_size;
    }
    
    //<editor-fold defaultstate="collapsed" desc="member params">
    protected EngineBase base = null;
    private boolean init_clear = false;
    
    private long max_mem_size = 0;//the upper bound of the length of memory could be allocated
    private long total_mem_size = 0;//the length of memory has been allocated
    private long used_mem_size = 0;//the length of memory has been used
    //buffered_memsize = total_mem_size - used_mem_size
    
    private final HashSet<Long> grave = new HashSet<>(64);//an address can be freed only once
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public synchronized Mempool max_mem_size(long max_mem_size) {
        if(max_mem_size <= 0) throw new IllegalArgumentException("Mempool.max_mem_size must be positive");
        if(max_mem_size < total_mem_size) throw new IllegalArgumentException(String.format(
                "[Error] Mempool: max_mem_size[%d MB] < total_mem_size[%d MB]", 
                max_mem_size_MB(), total_mem_size_MB()));
        
        this.max_mem_size = max_mem_size;
        return this;
    }
    
    @Override public final long max_mem_size() { return max_mem_size; }
    @Override public final long total_mem_size() { return total_mem_size; }
    @Override public final long used_mem_size() { return used_mem_size; }
    
    public abstract int physical_block_num();//g.,size
    public abstract int buffered_block_num();//pool.size
    
    public abstract long[] physical_addrs();
    
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
        sb.append(getClass().getName()).append(" { ");
        meta_data().forEach((String key, Object value)->{
            sb.append("\n\t").append(key).append(" = ").append(value);
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
    public void finalize() throws Throwable  {
        super.finalize();
        this.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others"> 
    protected long addr_shift(long address, long offset) { return address + offset; }
    
    protected void memory_insufficient() {
        throw new RuntimeException(String.format(
                "[Error] Memory<%d> is insufficient. Try to alloc more",
                total_mem_size_MB()));
    }
    
    protected void check_mem_size(long mem_size) {
        if(mem_size <= 0) throw new IllegalArgumentException("[Error] mem_length <= 0");
        if(mem_size > max_mem_size) throw new IllegalArgumentException(
                "[Error] mem_length > mempool.max_mem_size");
    }

    public EngineBase engineBase() { return base; }
    public synchronized Mempool engineBase(EngineBase base)  {
        if(init_clear) { clear(); init_clear = false; }
        this.base = base;
        return this;
    }
     
    protected final long __malloc(long mem_size) {
        long address = base.malloc(mem_size);
        if(address == NULL) throw new RuntimeException(String.format(
                "[Error] EngineBase<%s>: Failed to alloc memory for [mem_size = %d]",
                base.getClass().getName(), mem_size));
        total_mem_size += mem_size;
        return address;
    }
    
    protected final void __free(long mem_size, long mem_address) {
        if(mem_address == NULL) return;
        base.free(mem_address);
        total_mem_size -= mem_size;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: malloc"> 
    protected abstract long[] search(long mem_size);//use buffered mem_block
    
    protected void alloc_gc(long mem_size) {//try to gc some Tensor
        int time = 0;
        try {//max_avaiable_size = max_alloc_size + buffered_size
            while((time < 10) && (mem_size > (max_mem_size - used_mem_size))) {
                System.gc(); this.notifyAll(); this.wait(1000); time++;
            }
        }
        catch(InterruptedException e) {}
        if(time >= 10) memory_insufficient();
    }
    
    protected abstract boolean alloc_free(long mem_size, long max_alloc_size);
    
    protected abstract long[] alloc(final long mem_size);//alloc new memory,
   
    public synchronized long[] malloc(boolean check, int mem_length, long L_sizeof_datatype) 
    {
        mem_length = (mem_length + 3) >> 2 << 2;//padding: mem_length % 4 == 0
        final long mem_size = mem_length << L_sizeof_datatype;
        
        if(check) {
            if(L_sizeof_datatype < 0) throw new IllegalArgumentException("[Error] sizeof_datatype < 0");
            if(mem_length > Integer.MAX_VALUE) throw new IllegalArgumentException(
                    "[Error] mem_length > Integer.MAX_VALUE(Maximum Array Length in Java)");
            check_mem_size(mem_size);
        }
        
        long[] block = search(mem_size);
        
        if(block == null) {//max_available_size >= max_alloc_size
            long max_available_size = max_mem_size - used_mem_size;
            if(mem_size > max_available_size) alloc_gc(mem_size);
            
            long max_alloc_size = max_mem_size - total_mem_size;
            if(mem_size > max_alloc_size) {//insufficient mem_size to alloc new mem_block
                if(mem_size > buffered_mem_size() || buffered_block_num() == 0) memory_insufficient();
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
    
    //<editor-fold defaultstate="collapsed" desc="running-area: free & clear"> 
    protected abstract boolean recycle(long mem_size, long mem_address);//free: return tree if succeed to free
    public synchronized boolean free(boolean check, long mem_size, long mem_address) {
        //----------------------------------------------------------------------
        mem_size <<= 2;//block[0] = mem_size, recover
        //----------------------------------------------------------------------
        
        if(check) check_mem_size(mem_size);
        
        //you can't delete an NULL mem_address, or delete an mem_address twice
        if(mem_address == NULL || grave.contains(mem_address)) return false;
        
        boolean flag = recycle(mem_size, mem_address);
        
        grave.add(mem_address);
        used_mem_size -= mem_size;
        
        return flag;
    }
    
    protected abstract void __clear__();
    public synchronized void clear() {
        for(long mem_address : physical_addrs()) {
            if(mem_address != NULL) base.free(mem_address);
        }
        
        grave.clear();
        total_mem_size = 0;
        used_mem_size = 0;
        init_clear = false;
        
        __clear__();
    }
    //</editor-fold>
}

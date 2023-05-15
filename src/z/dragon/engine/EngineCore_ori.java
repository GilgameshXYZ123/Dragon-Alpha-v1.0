/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.TreeMap;

/**
 * 4D[N, IH, IW, IC]
 * dim: 0, 1, 2, 3.
 * @author Gilgamesh
 */
public class EngineCore_ori extends EngineCore
{
    public EngineCore_ori(long maxMemorySize, boolean check) {
        this.max_mem_size = maxMemorySize;
        this.check = check;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override
    public synchronized EngineCore engineBase(EngineBase base) {
        if(base == null) throw new NullPointerException("EngineBase is null");
        this.base = base; base.core = this;
        this.L_sizeof_datatype = base.L_sizeof_datatype;
        this.max_mem_length = this.max_mem_size >> L_sizeof_datatype;
        this.max_mem_size = this.max_mem_length << L_sizeof_datatype;
        return this;
    }
    
    @Override
    public synchronized EngineCore max_mem_size(long maxMemSize) {
        this.max_mem_length = maxMemSize >> L_sizeof_datatype;
        this.max_mem_size = this.max_mem_length << L_sizeof_datatype;
        return this;
    }
    
    @Override public long max_mem_size()    { return max_mem_length << L_sizeof_datatype; }
    @Override public long max_mem_size_MB() { return max_mem_length << L_sizeof_datatype >> 20; }
    
    @Override public long total_mem_size()    { return total_mem_length << L_sizeof_datatype; }
    @Override public long total_mem_size_MB() { return total_mem_length << L_sizeof_datatype >> 20; }

    @Override public long used_mem_size()    { return used_mem_length << L_sizeof_datatype; }
    @Override public long used_mem_size_MB() { return used_mem_length << L_sizeof_datatype >> 20; }
        
    @Override public long buffered_mem_size()    { return (total_mem_length - used_mem_length) << L_sizeof_datatype; }
    @Override public long buffered_mem_size_MB() { return (total_mem_length - used_mem_length) << L_sizeof_datatype >> 20;}

    public TreeMap<Integer, LinkedList<Long>> pool() { return memPool; }
    public HashSet<Long> memDic() { return memDic; }
    public ArrayList<Long> gc() { return gc; }
    
    @Override
    public void finalize() throws Throwable  {
        super.finalize();
        this.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Memory Pooling: create & delete"> 
    protected long max_mem_size;//max_mem_size = max_mem_length * sizeof_datatype
     
    protected long max_mem_length = 0;//the upper bound of the length of memory could be allocated
    protected long total_mem_length = 0;//the length of memory has been allocated
    protected long used_mem_length = 0;//the length of memory has been used
    
    //malloc: return { block_length, block_address }
    //Key: length of memory; memsize >= (Tensor.length_4x<<2L) [sizeof(float)]
    //Value: address of memory;
    //Integer: javaArray.length <= Integer.MAX_VALUE
    protected TreeMap<Integer, LinkedList<Long>> memPool = new TreeMap<>();
    protected HashSet<Long> memDic = new HashSet<>();
    protected ArrayList<Long> gc = new ArrayList<>(64);//garbage collector
    
    //padding to (1 << L_sizeof_datatype), As sizeof_datatype may not a power of 2
    //return: {mem_block_length : mem_block_address}
    //tensor.mem_length = mem_block_length
    
    @Override
    public synchronized long[] malloc_int8(int mem_length) {
        int mem_size = mem_length + (1 << L_sizeof_datatype) - 1;
        mem_length = mem_size >> L_sizeof_datatype;
        return malloc(mem_length);
    }
    
    @Override
    public synchronized long[] malloc_int32(int mem_length) {
        int mem_size = (mem_length << 2) + (1 << L_sizeof_datatype) - 1;
        mem_length = mem_size >> L_sizeof_datatype;
        return malloc(mem_length);
    }
    
    @Override
    public synchronized long[] malloc(int mem_length) 
    {
        if(check) {
            if(mem_length <= 0) throw new IllegalArgumentException("the memory.length must be unpositive");
            if(mem_length > Integer.MAX_VALUE) throw new IllegalArgumentException(
                    "the memory.length is greater than Integer.MAX_VALUE(Java greatest Array Length)");
            if(mem_length > max_mem_length) throw new IllegalArgumentException(
                    "the memory.length is greater than the maximum memory length can be alloced");
        }
        
        final Entry<Integer, LinkedList<Long>> block = memPool.ceilingEntry(mem_length);//key = ceiling(length)
        if(block != null)// exists memory block with enough length
        {
            int block_length = block.getKey();//block_length >= mem_length
            if(block_length < 1.5f * mem_length) {
                LinkedList<Long> block_addresses = block.getValue();
                used_mem_length += block_length;
                
                long block_address = block_addresses.remove();
                if(block_addresses.isEmpty()) memPool.remove(block_length); 
                
                memDic.remove(block_address);
                return new long[]{ block_length, block_address };//address = kv.value
            }
        }
        
        long max_alloc_length = max_mem_length - total_mem_length;//the size of maximize memory can be alloced
        long buffered_length = total_mem_length - used_mem_length;//the size of buffered memory in memPool
        try //the maximum memory can be used: buffered_memory + max_alloc_memory
        {
            int time = 0;
            while(time++ < 10 && mem_length > max_alloc_length + buffered_length) {
                System.gc();//try to gc some Tensor
                this.wait(1000);
            } 
            if(time >= 10) throw new RuntimeException("EngineCore: need more GPU memory, Try to alloc more");
        }
        catch(InterruptedException e) {}
            
        //(1) mem_length <= max_alloc_length + buffered_length
        //(2) mem_length > max_alloc_length
        //we need free some memory block in memPool
        //(1) + (2) -> buffered_length > 0, so memPool is not empty
        while(mem_length > max_alloc_length)
        {
            Entry<Integer, LinkedList<Long>> max_block = memPool.lastEntry();
            int max_block_length = max_block.getKey();
            
            //to alleviates memory fragmentation: remove smaller memory block first
            if(max_block_length > max_alloc_length) {
                max_block = memPool.ceilingEntry((int)max_alloc_length);
                max_block_length = max_block.getKey();
            }
            
            LinkedList<Long> block_addresses = max_block.getValue();
            long address = block_addresses.remove();
            if(block_addresses.isEmpty()) memPool.remove(max_block_length);
            
            if(address != NULL) {//free the mem_address only is not null
                base.free(address);
                total_mem_length -= max_block_length;
                max_alloc_length = max_mem_length - total_mem_length;
            }
        }
    
        //alloc new memory, we have: mem_length <= max_alloc_length
        long mem_size = mem_length << L_sizeof_datatype;//mem_size = mem_length * sizeof_datatype
        long address = base.malloc(mem_size);
        if(address == NULL) throw new RuntimeException(
                "EngineBase: failed to alloc memory for mem_size = " + mem_size);
        
        gc.add(address); 
        total_mem_length += mem_length; 
        used_mem_length += mem_length;
        
        return new long[]{ mem_length, address };
    }
    
    @Override
    public synchronized boolean free(long mem_size, long mem_address) 
    {
        //you can't delete an NULL mem_address, or delete an mem_address twice
        if(mem_address == NULL || memDic.contains(mem_address)) return false;
        
        int mem_length = (int) (mem_size);
        
        LinkedList<Long> block_addresses = memPool.get(mem_length);
        if(block_addresses == null) {
            memPool.put(mem_length, block_addresses = new LinkedList<>());
        }
        
        memDic.add(mem_address);
        block_addresses.add(mem_address);
        used_mem_length -= mem_length;
        
        return true;
    }
    
    @Override
    public synchronized void clear()
    {
        Iterator<Long> iter = gc.iterator();
        while(iter.hasNext()) {//clear all memory allocation
            long address = iter.next();
            if(address != NULL) base.free(address);
        }
        
        memPool.clear();
        gc.clear();
        memDic.clear();
        total_mem_length = 0;
        used_mem_length = 0;
    }
    //</editor-fold>
}

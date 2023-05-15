/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.addrp;

import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.TreeMap;

/**
 *
 * @author Gilgamesh
 */
public class Addrp2 implements Addrpool
{
    public static final long max_mem_size_int32 = (1L << 30) - 1;
    
    private int size = 0;
    protected final TreeMap<Long   , LinkedList<Long>> int64 = new TreeMap<>();
    protected final TreeMap<Integer, LinkedList<Long>> int32 = new TreeMap<>();
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public int size() { return size; }
    @Override public boolean isEmpty() { return size == 0; }
    
    public void append(StringBuilder sb) {
        sb.append(getClass().getName()).append(" {");
        sb.append("\n[mem_size < max_mem_size_int32] = ").append(int32.size());
        int32.forEach((Integer block_size, LinkedList<Long> block_list)->{
            sb.append("\n\t").append(block_size).append(" : ").append(block_list);
        });
        sb.append("\n[mem_size >= max_mem_size_int32] = ").append(int64.size());
        int64.forEach((Long block_size, LinkedList<Long> block_list)->{
            sb.append("\n\t").append(block_size).append(" : ").append(block_list);
        });
    }
    
    @Override 
    public String toString() {
        StringBuilder sb = new StringBuilder();
        this.append(sb);;
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: put && remove & clear">
    @Override
    public boolean put(long mem_size, long mem_address) 
    {
        LinkedList<Long> block_list;
        if(mem_size <= max_mem_size_int32) {
            int mem_size_int32 = (int) mem_size;
            if((block_list = int32.get(mem_size_int32)) == null) 
                int32.put(mem_size_int32, block_list = new LinkedList<>());
        }
        else {
            long mem_size_int64 = mem_size;
            if((block_list = int64.get(mem_size_int64)) == null) 
                int64.put(mem_size_int64, block_list = new LinkedList<>());
        }
        
        boolean flag = block_list.add(mem_address); size++;
        return flag;
    }
    
    @Override
    public boolean remove(long mem_size, long mem_address) 
    {
        LinkedList<Long> block_list; boolean flag;
        if(mem_size <= max_mem_size_int32) {
            int mem_size_int32 = (int) mem_size;
            if((block_list = int32.get(mem_size_int32)) == null) return false;
            
            flag = block_list.remove(mem_address);
            if(block_list.isEmpty()) int32.remove(mem_size_int32);
        }
        else {
            long mem_size_int64 = mem_size;
            if((block_list = int64.get(mem_size_int64)) == null) return false;
            
            flag = block_list.remove(mem_address);
            if(block_list.isEmpty()) int64.remove(mem_size_int64);
        }
        
        size--;
        return flag;
    }
    
    public void clear() {
        int32.clear(); int64.clear(); size = 0;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: remove<max, ceiling, floor, section>">
    @Override
    public long[] remove_max() 
    {
        LinkedList<Long> block_list; long block_size, block_address;
        if(int64.isEmpty()) //try to remove in int32
        {
            final Entry<Integer, LinkedList<Long>> block_int32 = int32.lastEntry();
            if(block_int32 == null) return null;
            
            block_size = block_int32.getKey();
            block_address = (block_list = block_int32.getValue()).remove();
            if(block_list.isEmpty()) int32.remove((int)block_size);
        }
        else 
        {
            final Entry<Long, LinkedList<Long>> block_int64 = int64.lastEntry();
            if(block_int64 == null) return null;
            
            block_size = block_int64.getKey();
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
        }
        return new long[] { block_size, block_address };
    }
    
    @Override
    public long[] remove_ceiling(long min_mem_size)//mem_size >= min_mem_size
    {
        LinkedList<Long> block_list; long block_size, block_address;
        if(min_mem_size > max_mem_size_int32) 
        {
            long mem_size_int64 = min_mem_size;
            final Entry<Long, LinkedList<Long>> block_int64 = int64.ceilingEntry(mem_size_int64);
            if(block_int64 == null) return null;
            
            block_size = block_int64.getKey();
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
        }
        else for(;;)//min_mem_size <= max_memsize_int32
        {
            final Entry<Integer, LinkedList<Long>> block_int32 = int32.ceilingEntry((int)min_mem_size);
            if(block_int32 != null) {
                block_size = block_int32.getKey(); 
                block_address = (block_list = block_int32.getValue()).remove();
                if(block_list.isEmpty()) int32.remove((int)block_size);
                break;
            }
            
            final Entry<Long, LinkedList<Long>> block_int64 = int64.ceilingEntry(min_mem_size);
            if(block_int64 == null) return null;
            
            block_size = block_int64.getKey();
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
            break;
        }
        
        size--;
        return new long[] { block_size, block_address };
    }
    
    @Override
    public long[] remove_floor(long max_mem_size)//mem_size <= min_mem_size
    {
        LinkedList<Long> block_list; long block_size, block_address;
        if(max_mem_size <= max_mem_size_int32) 
        {
            final Entry<Integer, LinkedList<Long>> block_int32 = int32.floorEntry((int)max_mem_size);
            if(block_int32 == null) return null;
            
            block_size = block_int32.getKey();
            block_address = (block_list = block_int32.getValue()).remove();
            if(block_list.isEmpty()) int32.remove((int)block_size);
        }
        else for(;;)//max_mem_size > max_memsize_int32
        {
            final Entry<Integer, LinkedList<Long>> block_int32 = int32.floorEntry((int)max_mem_size);
            if(block_int32 != null) {
                block_size = block_int32.getKey();
                block_address = (block_list = block_int32.getValue()).remove();
                if(block_list.isEmpty()) int32.remove((int)block_size);
                break;
            }
            
            final Entry<Long, LinkedList<Long>> block_int64 = int64.floorEntry(max_mem_size);
            if(block_int64 == null) return null;
            
            block_size = block_int64.getKey();
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
            break;
        }
         
        size--;
        return new long[] { block_size, block_address };
    }    
    
    @Override
    public long[] remove_section(long min_mem_size, long max_mem_size) 
    {   
        LinkedList<Long> block_list; long block_size, block_address;
        if(min_mem_size > max_mem_size) {//min_mem_size <= max_mem_size
            long t = min_mem_size;
            min_mem_size = max_mem_size;
            max_mem_size = t;
        } 
       
        if(max_mem_size <= max_mem_size_int32)//[int32, int32]
        {
            final Entry<Integer, LinkedList<Long>> block_int32 = int32.ceilingEntry((int)min_mem_size);
            if(block_int32 == null) return null;
            
            if((block_size = block_int32.getKey()) > max_mem_size) return null;
            
            block_address = (block_list = block_int32.getValue()).remove();
            if(block_list.isEmpty()) int32.remove((int)block_size);
        }
        else if(min_mem_size > max_mem_size_int32)//[int64, int 64]
        {
            final Entry<Long, LinkedList<Long>> block_int64 = int64.ceilingEntry(min_mem_size);
            if(block_int64 == null) return null;
            
            if((block_size = block_int64.getKey()) > max_mem_size) return null;
            
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
        }
        else for(;;)//[int32, int64]
        {
            Entry<Integer, LinkedList<Long>> block_int32 = int32.ceilingEntry((int)min_mem_size);
            if(block_int32 != null) {
                if((block_size = block_int32.getKey()) > max_mem_size) return null;
                
                block_address = (block_list = block_int32.getValue()).remove();
                if(block_list.isEmpty()) int32.remove((int)block_size);
                break;
            }
            
            Entry<Long, LinkedList<Long>> block_int64 = int64.ceilingEntry(min_mem_size);
            if(block_int64 == null) return null;
            
            if((block_size = block_int64.getKey()) > max_mem_size) return null;
            
            block_address = (block_list = block_int64.getValue()).remove();
            if(block_list.isEmpty()) int64.remove(block_size);
            break;
        }
        
        size--;
        return new long[] { block_size, block_address };
    }
    //</editor-fold>
}

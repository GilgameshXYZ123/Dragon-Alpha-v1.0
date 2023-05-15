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
 * [mem_size, blockList[addresses]]
 * @author Gilgamesh
 */
public class Addrp1 extends TreeMap<Long, LinkedList<Long>>
        implements Addrpool
{
    //<editor-fold defaultstate="collapsed" desc="member params & Basic-Functions">
    private static final long serialVersionUID = 20233201250L;
    
    private int size = 0;
    
    @Override public int size() { return size; }
    
    @Override public void clear() { super.clear(); size = 0; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: put && remove & clear">
    @Override
    public boolean put(long mem_size, long mem_address) {
        LinkedList<Long> block_list = get(mem_size);
        if(block_list == null) put(mem_size, block_list = new LinkedList<>());
        
        boolean flag = block_list.add(mem_address); size++;
        return flag;
    }
    
    @Override
    public boolean remove(long mem_size, long mem_address) {
        LinkedList<Long> block_list = get(mem_size);
        if(block_list == null) return false;
        
        boolean flag = block_list.remove(mem_address); size--;
        if(block_list.isEmpty()) remove(mem_size);
        return flag;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: put && remove & clear">
    @Override
    public long[] remove_max()//return { mem_size, mem_address }
    {
        final Entry<Long, LinkedList<Long>> block = lastEntry();
        if(block == null) return null;
        
        long block_size = block.getKey();
        
        LinkedList<Long> block_list = block.getValue();
        long block_address = block_list.remove();
        if(block_list.isEmpty()) remove(block_size);
        
        size--;
        return new long[] { block_size, block_address };
    }
    
    @Override
    public long[] remove_ceiling(long min_mem_size) 
    {
        final Entry<Long, LinkedList<Long>> block = ceilingEntry(min_mem_size);//block_size >= min_mem_size
        if(block == null) return null;
        
        long block_size = block.getKey();
        
        LinkedList<Long> block_list = block.getValue();
        long block_address = block_list.remove();
        if(block_list.isEmpty()) remove(block_size);
        
        size--;
        return new long[] { block_size, block_address };
    }
    
    @Override
    public long[] remove_floor(long max_mem_size) 
    {
        final Entry<Long, LinkedList<Long>> block = floorEntry(max_mem_size);
        if(block == null) return null;
        
        long block_size = block.getKey();
        
        LinkedList<Long> block_list = block.getValue();
        long block_address = block_list.remove();
        if(block_list.isEmpty()) remove(block_size);
        
        size--;
        return new long[] { block_size, block_address };
    }
    
    //max_mem_size >= block_size >= min_mem_size
    @Override
    public long[] remove_section(long min_mem_size, long max_mem_size)
    {
        final Entry<Long, LinkedList<Long>> block = ceilingEntry(min_mem_size);//block_size >= min_mem_size
        if(block == null) return null;
        
        long block_size = block.getKey();
        if(block_size > max_mem_size) return null;//block_size >= max_mem_size
        
        LinkedList<Long> block_list = block.getValue();
        long block_address = block_list.remove();
        if(block_list.isEmpty()) remove(block_size);
        
        size--;
        return new long[] { block_size, block_address };
    }
    //</editor-fold>
}

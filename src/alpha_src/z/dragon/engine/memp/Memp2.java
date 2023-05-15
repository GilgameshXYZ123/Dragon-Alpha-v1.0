/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.memp;

import z.dragon.common.addrp.Addrpool;
import java.util.HashMap;
import z.dragon.data.EntryList;

/**
 * Mark2
 * @author Gilgamesh
 */
public class Memp2 extends Mempool
{
    //<editor-fold defaultstate="collapsed" desc="class: Mnode">
    protected class Mnode 
    {
        public Mnode(boolean free, long mem_size, long mem_address) {
            this.free = free;
            this.mem_size = mem_size;
            this.mem_address = mem_address;
        }
        
        //<editor-fold defaultstate="collapsed" desc="members && Basic-Functions">
        protected long mem_size;
        protected long mem_address;
        protected boolean free;//free = true, Mnode.adress is in pool
         
        protected Mnode next = null;
        protected Mnode last = null;
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(256);
            sb.append("{ free = ").append(free);
            sb.append(", mem_size = ").append(mem_size);
            sb.append(", mem_address = ").append(mem_address);
            sb.append(", [has_next, has_last] = [")
                    .append(last != null).append(", ")
                    .append(next != null).append("] }");
            return sb.toString();
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: split">
        protected Mnode[] split(long mem_size1)
        {
            Mnode n1 = new Mnode(false, mem_size1, mem_address);//n1 is used            
            Mnode n2 = new Mnode(true, //add n2 to pool
                    mem_size - mem_size1,//mem_size2
                    mem_address + mem_size1);//mem_address + mem_size1
            
            //Link: last -> [n1 -> n2] -> next
            n1.last = last; if(last != null) last.next = n1;
            n1.next = n2; n2.last = n1;
            n2.next = next; if(next != null) next.last = n2;
            
            //this.add = n1.address, So: when put n1, this is removed
            this.next = null; this.last = null;
            node_map.put(n1.mem_address, n1);
            node_map.put(n2.mem_address, n2);
            
            return new Mnode[]{ n1, n2 };
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: union">
        protected void remove() {
            node_map.remove(mem_address); next = null; last = null;
            pool.remove(mem_size, mem_address);
        }
        
        protected Mnode union() 
        {
            Mnode n; free = true;
            Mnode n0 = last, n1 = this, n2 = next;
            
            boolean union_last = (last != null) && (last.free);
            boolean union_next = (next != null) && (next.free);
            
            if(union_last && union_next) {
                n = new Mnode(true,//add n to pool
                        n0.mem_size + n1.mem_size + n2.mem_size,
                        n0.mem_address);
                
                //Link: n0.last -> [n = n0 + n1 + n2] -> n2.next
                n.last = n0.last; if(n0.last != null) n0.last.next = n; 
                n.next = n2.next; if(n2.next != null) n2.next.last = n;
                
                last.remove();
                next.remove();
                node_map.remove(mem_address); next = null; last = null;
            }
            else if(union_last) {
                n = new Mnode(true,//add n to pool
                        n0.mem_size + n1.mem_size,
                        n0.mem_address);
                
                //Link: n0.last -> [n = n0 + n1] -> n1.next
                n.last = n0.last; if(n0.last != null) n0.last.next = n;
                n.next = n1.next; if(n1.next != null) n1.next.last = n;

                last.remove();
                node_map.remove(mem_address); next = null; last = null;
            }
            else if(union_next) {
                n = new Mnode(true,//add n to pool
                        n1.mem_size + n2.mem_size,
                        n1.mem_address);
                
                //Link: n1.last -> [n = n1 + n2] -> n2.next
                n.last = n1.last; if(n1.last != null) n1.last.next = n;
                n.next = n2.next; if(n2.next != null) n2.next.last = n;
                
                next.remove();
                node_map.remove(mem_address); next = null; last = null;
            }
            else {
                if(last == null && next == null) {//remove if an isolated node
                    node_map.remove(mem_address); next = null; last = null;
                }
                return this;
            }
            
            if(n.last != null || n.next != null) {//n is not an isolated node
                node_map.put(n.mem_address, n);
            }
            return n;
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    public Memp2(long maxMemorySize) {
        super(maxMemorySize);
    }
    
    //<editor-fold defaultstate="collapsed" desc="member params && Basic-Functions">
    protected final Addrpool pool = Addrpool.instance();
    protected final HashMap<Long, Long> gc = new HashMap<>(128);//garbage collector[mem_address, mem_size]
    
    public Addrpool pool() { return pool; }
    public HashMap<Long, Long> gc() { return gc; }
    
    @Override public int physical_block_num() { return gc.size(); }
    @Override public int buffered_block_num() { return pool.size(); }
   
    @Override public long[] physical_addrs() { 
        long[] addrs = new long[gc.size()]; int index = 0;
        for(long addr : gc.keySet()) addrs[index++] = addr;
        return addrs;
    }
    
    protected HashMap<Long, Mnode> node_map = new HashMap<>(64);
    public HashMap<Long, Mnode> node_map() { return node_map; }
    
    protected long min_split_threshold = 4096;//4 K
    public long min_split_threshold() { return min_split_threshold; }
    public Memp2  min_split_threshold(long v) { min_split_threshold = v; return this; }
    
    protected long split_abs = 2 * 1024 * 1024;//2 M
    public long split_threshold_absolute() { return split_abs; }
    public Memp2 split_threshold_absolute(long v) { split_abs = v; return this; }
    
    protected double split_rel = 0.2;
    protected double split_threshold_relative() { return split_rel; }
    protected Memp2 split_threshold_relative(double v) { split_rel = v; return this; }
    
    @Override
    public EntryList<String, Object> meta_data() {
        EntryList<String, Object> mt = super.meta_data();
        mt.put("node_map.size", node_map.size());
        return mt;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: alloc">
    @Override
    protected long[] search(long mem_size) 
    {
        if(mem_size < min_split_threshold) {//avoid arror of small memsize
            return pool.remove_section(mem_size, (long) (mem_size * 1.5));
        }
        
        long[] block = pool.remove_ceiling(mem_size);
        if(block == null) return null;
        
        long block_size = block[0];
        long block_address = block[1];
        Mnode node = node_map.get(block_address);
        
        long sub_size = block_size - mem_size;
        long split_Rel = (long) (split_rel * block_size);
        
        if(((sub_size > split_abs) || (sub_size > split_abs)) && 
           ((sub_size > split_Rel) || (sub_size > split_Rel)))
        {
            if(node == null) node = new Mnode(true, block_size, block_address);
            
            Mnode[] sub_node = node.split(mem_size);
            Mnode node1 = sub_node[0], node2 = sub_node[1];
            
            block[0] = node1.mem_size;//block_size = node1.mem_size
            block[1] = node1.mem_address; //block_address = node1.mem_address
            
            pool.put(node2.mem_size, node2.mem_address);
        }
        else {
            if(node != null) node.free = false;
        }
        return block;
    }
    
    protected boolean is_physical_block(long mem_size, long mem_address) {
        Long ori_size = gc.get(mem_address);
        return (ori_size != null) && (ori_size == mem_size);
    }
    
    @Override
    protected boolean alloc_free(long mem_size, long max_alloc_size)
    {
        if(pool.isEmpty()) return false;//no buffered mem_block to release
        
        long[] max_block = pool.remove_max();//max_block_size < mem_size
        long max_block_size = max_block[0];
        long max_block_address = max_block[1];
        
        if(this.is_physical_block(max_block_size, max_block_address)) {
            this.__free(max_block_size, max_block_address);
            gc.remove(max_block_address);
            max_alloc_size += max_block_size;
        }
        else pool.put(max_block_size, max_block_address);
        
        long ceil_block_size = max_block_size - 1;
        
        while((mem_size > max_alloc_size) && (!pool.isEmpty())) 
        {
            long dst_mem_size = mem_size - max_alloc_size;
            if(dst_mem_size > ceil_block_size) dst_mem_size = ceil_block_size;
            
            long[] dst_block = pool.remove_floor(dst_mem_size);
            if(dst_block == null) return false;
            
            long dst_block_size = dst_block[0];
            long dst_block_address = dst_block[1];
            
            if(this.is_physical_block(dst_block_size, dst_block_address)) {
                this.__free(dst_block_size, dst_block_address);
                gc.remove(dst_block_address);
                max_alloc_size += dst_block_size;
            }
            else pool.put(dst_block_size, dst_block_address);
            
            ceil_block_size = dst_block_size - 1;
        }
        
        return true;
    }
    
    @Override
    protected long[] alloc(long mem_size) {
        long mem_address = this.__malloc(mem_size);
        gc.put(mem_address, mem_size);
        return new long[]{ mem_size, mem_address };
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: free & clear"> 
    @Override
    protected boolean recycle(long mem_size, long mem_address) 
    {
        Mnode node = node_map.get(mem_address);//pool.contains(mem_address) == false
        if(node != null) {//union node with its neighbors
            Mnode unode = node.union();
            mem_size = unode.mem_size;
            mem_address = unode.mem_address;
        }
        
        pool.put(mem_size, mem_address);
        return true;
    }

    @Override
    protected void __clear__()  {
        pool.clear();
        gc.clear(); 
        node_map.clear();
    }
    //</editor-fold>
}

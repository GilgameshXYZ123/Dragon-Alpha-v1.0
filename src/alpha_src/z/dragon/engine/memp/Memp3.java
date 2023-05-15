/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.memp;

import java.util.HashMap;
import java.util.Map.Entry;
import z.dragon.engine.memp.Memp2.Mnode;

/**
 * Mark3
 * @author Gilgamesh
 */
public class Memp3 extends Memp2
{
    //<editor-fold defaultstate="collapsed" desc="class: Combiner">
    public class Combiner 
    {
        protected HashMap<Long, Long> blocks = new HashMap<>(8);//[mem_address, mem_size]
        protected long total_size = 0;//sum of mem_size
        protected long total_time = 0;
       
        protected long max_recv_size_abs = 1024 * 1024 * 32;//32 M
        protected double max_recv_size_rel = 0.1;//25.6M
        
        protected long max_combined_size = 1024 * 1024 * 512;//512 M
        
        protected int num_threshold = 5;
        protected int time_threshold = 30;
        protected long size_threshold = 1024 * 1024 * 64;//64 M
        
        public long max_recv_size_abs() { return max_recv_size_abs; }
        public double max_recv_size_rel() { return max_recv_size_rel; }
        
        public long max_combined_size() { return max_combined_size; }
        
        public int num_threshold() { return num_threshold; }
        public long size_threshold() { return size_threshold; }
        
        public Combiner max_recv_size_abs(long v) { max_recv_size_abs = v; return this; }
        public Combiner max_recv_size_rel(double v) { max_recv_size_rel = v; return this; }
        
        public Combiner max_combined_size(long v) { max_combined_size = v; return this; }
        
        public Combiner num_threshold(int v) { num_threshold = v; return this; }
        public Combiner size_threshold(long v) { size_threshold = v; return this; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area">
        public void remove(long block_address) {
            Long block_size = blocks.remove(block_address);
            if(block_size != null) { total_size -= block_size; }
        }
        
        public void clear() { 
            blocks.clear();
            total_size = 0;
            total_time = 0;
        }
        
        protected long[] combine() 
        {
            long max_size = 0;
            for(Entry<Long, Long> kv : blocks.entrySet()) 
            {
                long block_address = kv.getKey();
                long block_size = kv.getValue();
               
                if(max_size < block_size) max_size = block_size;
                
                pool.remove(block_size, block_address);
                if(!gc.remove(block_address, block_size)) System.exit(-2);
                Memp3.this.__free(block_size, block_address);
            }
            
            long new_size = (long) (total_size * 0.95);
            if(new_size < max_size) {
                new_size = (long) (max_size * 1.05);
                if(new_size > total_size) new_size = total_size;
            }
            long new_address = Memp3.this.__malloc(new_size);
            
            this.clear();
            
            return new long[]{ new_size, new_address };
        }
        
        public long[] put(long block_size, long block_address)
        {
            //only accept small blocks as memory segment
            long recv_size = max_combined_size - total_size;
            if(recv_size > max_recv_size_abs) recv_size = max_recv_size_abs;
            
            long recv_rel = (long) (Memp3.this.max_mem_size() * max_recv_size_rel);
            if(recv_size > recv_rel) recv_size = recv_rel;
            
            if(block_size > recv_size) return null;
           
            //update stats of combiner 
            blocks.put(block_address, block_size);
            total_size += block_size;
            total_time++;
            
            //free small mem blocks -> alloc a big one
            int size = blocks.size();
            if((size > 1)) {//at leaset 2 blocks to combine
                if((size > num_threshold) ||
                   (total_time > time_threshold) ||
                   (total_size > size_threshold)) return this.combine();
            }
            
            return null;
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    public Memp3(long maxMemorySize) {
        super(maxMemorySize);
    }
    
    protected final Combiner combiner = new Combiner();
    public Combiner combiner() { return combiner; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: alloc"> 
    @Override
    protected long[] search(long mem_size) {
        long[] block = super.search(mem_size);//Memp2.search
        if(block != null) combiner.remove(block[1]);//block_address
        return block;
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
            combiner.remove(max_block_address);
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
                combiner.remove(dst_block_address);
                max_alloc_size += dst_block_size;
            }
            else pool.put(dst_block_size, dst_block_address);
            
            ceil_block_size = dst_block_size - 1;
        }
        
        return true;
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
        
        boolean ori = is_physical_block(mem_size, mem_address);
        if(ori) { 
            long[] block = combiner.put(mem_size, mem_address);
            if(block != null) {
                mem_size = block[0];
                mem_address = block[1];
            }
        }
        
        pool.put(mem_size, mem_address);
        return true;
    }
    
    @Override
    protected void __clear__() {
        pool.clear();
        gc.clear(); 
        node_map.clear();
        combiner.clear();
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.addrp;

/**
 *
 * @author Gilgamesh
 */
public interface Addrpool 
{
    public int size();
    default boolean isEmpty() { return size() == 0; }
    
    public boolean put(long mem_size, long mem_address);
    public boolean remove(long mem_size, long mem_address);
    public void clear();
    
    public long[] remove_max();
    public long[] remove_ceiling(long min_mem_size);
    public long[] remove_floor(long max_mem_size);
    public long[] remove_section(long min_mem_size, long max_mem_size);
    
    //<editor-fold defaultstate="collapsed" desc="interface: AddrpoolFactory">
    public static interface AddrpoolFactory {
        public abstract Addrpool create();
    }
    
    public static class maker {
        private static AddrpoolFactory default_factory = ()-> { return new Addrp4(); };
    }
    
    public static void default_factory(AddrpoolFactory factory) { 
        if(factory == null) throw new RuntimeException();
        synchronized(maker.class) { 
            maker.default_factory = factory; 
        }
    }
    
    public static Addrpool instance() { 
        synchronized(maker.class) {
            return maker.default_factory.create(); 
        }
    }
    //</editor-fold>
}

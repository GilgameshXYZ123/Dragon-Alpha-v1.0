/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int64;


/**
 * <long, V>
 * @author Gilgamesh
 * @param <V>
 */
public interface Entry_int64<V> 
{
    public long getKey();
    
    public V getValue();
    
    public V setValue(V value);
    
    //<editor-fold defaultstate="collapsed" desc="static class: FinalEntry_int64<V>">
    public static class FinalEnrty_int64<V> implements Entry_int64<V> 
    {
        private final long key;
        private final V value;

        public FinalEnrty_int64(long key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override public long getKey() { return key; }
        @Override public V getValue() { return value; }
        @Override public V setValue(V value) { throw new UnsupportedOperationException(); }
     
        @Override public String toString() {  return key + "=" + value; }
    }
    //</editor-fold>
}

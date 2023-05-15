/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int32;

/**
 * <int, V>
 * @author Gilgamesh
 * @param <V>
 */
public interface Entry_int32<V> 
{
    public int getKey();
    
    public V getValue();
    
    public V setValue(V value);
    
    //<editor-fold defaultstate="collapsed" desc="static class: FinalEntry_int32<V>">
    public static class FinalEnrty_int32<V> implements Entry_int32<V> 
    {
        private final int key;
        private final V value;

        public FinalEnrty_int32(int key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override public int getKey() { return key; }
        @Override public V getValue() { return value; }
        @Override public V setValue(V value) { throw new UnsupportedOperationException(); }
     
        @Override public String toString() {  return key + "=" + value; }
    }
    //</editor-fold>
}

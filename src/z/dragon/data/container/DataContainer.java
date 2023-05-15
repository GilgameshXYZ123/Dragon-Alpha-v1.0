/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import z.dragon.data.BatchIter;
import z.dragon.data.Pair;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public interface DataContainer<K, V>
{
    public int size();
    default boolean isEmpty() { return size() == 0; }
    
    public Class<K> input_class();
    public Class<V> label_class();
    
    public void shuffle(double percent);
    default void shuffle() { shuffle(0.25); }
    
    public void add(K key, V value);
    public void add(K[] keys, V[] values);
    
    public Pair<K, V> get();
    public Pair<K[], V[]> get(int batch);
     
    public void clear();
    
    public BatchIter<K[], V[]> batch_iter();
}

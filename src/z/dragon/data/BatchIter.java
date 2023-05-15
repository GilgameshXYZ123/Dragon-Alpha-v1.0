/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public interface BatchIter<K, V> 
{
    default void reset() { reset(false); }
    public void reset(boolean suffle);
    
    public boolean hasNext();
    
    public Pair<K, V> next(int batch);
}

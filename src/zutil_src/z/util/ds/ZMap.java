/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

import z.util.ds.imp.Indexable;
import java.util.Map;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import z.util.ds.imp.MapRemoveExtensive;
import z.util.ds.imp.Numberable;

/**
 *
 * @author dell
 * @param <K>
 * @param <V>
 */
public abstract class ZMap<K extends Comparable,V> implements Map<K,V>, Indexable, MapRemoveExtensive, Numberable
{
    protected int size;
    
    @Override
    public int size()
    {
        return size;
    }
    @Override
    public boolean isEmpty() 
    {
        return size==0;
    }
    //<editor-fold defaultstate="collapsed" desc="Remove Extensive">
    protected abstract boolean innerRemoveAll(Predicate pre);
    protected abstract boolean innerRemoveAll(BiPredicate pre);
    @Override
    public boolean removeAll(BiPredicate pre) 
    {
        return (this.isEmpty()? false:this.innerRemoveAll(pre));
    }
    @Override
    public boolean retainAll(BiPredicate pre) 
    {
        return (this.isEmpty()? false:this.innerRemoveAll(pre.negate()));
    }
    @Override
    public boolean removeAll(Predicate pre) 
    {
        return (this.isEmpty()? false:this.innerRemoveAll(pre));
    }
    @Override
    public boolean retainAll(Predicate pre) 
    {
        return (this.isEmpty()? false:this.innerRemoveAll(pre.negate()));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Abstract Method">
    public abstract boolean contains(Object key, Object value);
    /**
     * iterate through all keys in this Map;
     * @param con 
     */
    public abstract void forEachKey(Consumer<? super K> con);
    /**
     * iterate through all values in this Map.
     * @param con 
     */
    public abstract void forEachValue(Consumer<? super V> con);
    /**
     * iterate through all entries in this Map.
     * @param con 
     */
    public abstract void forEachEntry(Consumer<? super Entry<K, V>> con);
    /**
     * This Method is different from {@link Map#keySet() }.
     * As keySet() may give you an logical view of the KeySet of this Map, while 
     * this Method will create a new ZSet, and add all keys.
     * @return 
     */
    public abstract ZSet<K> replicaKeySet();
    /**
     * This Method is different from {@link Map#values() ) }.
     * As keySet() may give you an logical view of the ValueCollection of this Map, while 
     * this Method will create a new ZCollection, and add all values.
     * @return 
     */
    public abstract ZCollection<V> replicaValues();
     /**
     * This Method is different from {@link Map#entrySet() ) ) }.
     * As keySet() may give you an logical view of the Entries of this Map, while 
     * this Method will create a new ZSet, and add all Entries in.
     * @return 
     */
    public abstract ZSet<? extends ZEntry<K,V>> replicaEntrySet();
    //</editor-fold>
}

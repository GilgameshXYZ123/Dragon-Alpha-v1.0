/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

import java.io.Serializable;
import java.util.Map.Entry;

/**
 * if you implement ZEntry, you'd better overwrite ZEntry.compareTo().
 * @author dell
 * @param <K>
 * @param <V>
 */
public class ZEntry<K extends Comparable, V> implements Entry<K,V>, Serializable, Comparable
{
    //columns-------------------------------------------------------------------
    public K key;
    public V value;
    
    //constructors--------------------------------------------------------------
    public ZEntry() {}
    public ZEntry(K key, V value) 
    {
        this.key = key;
        this.value = value;
    }
    //functions-----------------------------------------------------------------
    @Override
    public K getKey() 
    {
        return key;
    }
    @Override
    public V getValue() 
    {
        return value;
    }
    @Override
    public V setValue(V value) 
    {
        V ov=this.value;
        this.value=value;
        return ov;
    }
    public void append(StringBuilder sb)
    {
        sb.append(key).append(" = ").append(value);
    }
    @Override
    public String toString()
    {
        return key+" = "+value;
    }
    @Override
    public int compareTo(Object o) 
    {
       return (o instanceof ZEntry? 
                    key.compareTo(((ZEntry)o).key):
                    key.compareTo(o));
    }
}

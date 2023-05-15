/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.Map.Entry;

/**
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public class Pair<K, V> implements Entry<K, V>
{
    public K input;
    public V label;

    public Pair(K input, V label) {
        this.input = input;
        this.label = label;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public void append(StringBuilder sb) {
        sb.append("{ input = ").append(input);
        sb.append("label = ").append(label).append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }

    @Override public K getKey() { return input; }
    @Override public V getValue() { return label; }
    @Override public V setValue(V value) {  V old = label; label = value; return old; }
    //</editor-fold>
}

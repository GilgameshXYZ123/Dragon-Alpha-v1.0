/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import z.dragon.data.BatchIter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import z.dragon.data.Pair;
import z.util.math.ExRandom;

/**
 * @author Gilgamesh
 * @param <K>
 * @param <V> 
 */
 @SuppressWarnings(value = "unchecked")
public class ListContainer<K, V> implements DataContainer<K, V>
{
    private final ExRandom exr = new ExRandom();
    
    protected final Class<K> kclazz;
    protected final Class<V> vclazz;
    
    private final ArrayList<K> karr;
    private final ArrayList<V> varr;
   
    public ListContainer(Class<K> input_clazz, Class<V> label_clazz, int initCapacity)
    {
        if(input_clazz == null) throw new NullPointerException("input_class is null");
        if(label_clazz == null) throw new NullPointerException("label_class is null");
        
        this.kclazz = input_clazz;
        this.vclazz = label_clazz;
        karr = new ArrayList<>(initCapacity);
        varr = new ArrayList<>(initCapacity);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public int size() { return karr.size(); }

    @Override public Class<K> input_class() { return kclazz; }
    @Override public Class<V> label_class() { return vclazz; }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="operators">
    public synchronized void ensureCapacity(int capacity) {
        karr.ensureCapacity(capacity);
        varr.ensureCapacity(capacity);
    }
    
    @Override
    public synchronized void shuffle(double percent)
    {
        int size = karr.size();
        int num = (int) Math.ceil(size * percent);
        if(num > size) num = size; 
        num = (num + 1) >> 1;

        for(int i=0; i<num; i++) {
            int index1 = exr.nextInt(0, size - 1);
            int index2 = exr.nextInt(0, size - 1);
            K k = karr.get(index1); karr.set(index1, karr.get(index2)); karr.set(index2, k);
            V v = varr.get(index1); varr.set(index1, varr.get(index2)); varr.set(index2, v);
        }
    }
    
    @Override
    public void add(K input, V label) {
        if(input == null || label == null) return;
        synchronized(this) {
            karr.add(input); 
            varr.add(label);
        }
    }
    
    @Override
    public void add(K[] inputs, V[] labels) 
    {
        if(inputs == null || labels == null) return;
        if(inputs.length != labels.length) throw new IllegalArgumentException(String.format(
                "inputs.length[%d] != labels.length[%d]", inputs.length, labels.length));
        
        synchronized(this) {
            karr.ensureCapacity(karr.size() + inputs.length);
            varr.ensureCapacity(varr.size() + labels.length);
            for(int i=0; i<inputs.length; i++) {
                if(inputs[i] == null || labels[i] == null) continue;
                karr.add(inputs[i]);
                varr.add(labels[i]);
            }
        }
    }

    @Override
    public Pair<K, V> get() {
        K k; V v;
        synchronized(this) {
            if (karr.isEmpty()) throw new NullPointerException("The Container is empty");
            int index = exr.nextInt(0,  karr.size() - 1);
            k = karr.get(index);
            v = varr.get(index);
        }
        return new Pair<>(k, v);
    }
     
    @Override
    public Pair<K[], V[]> get(int batch)  
    {
        if(batch <= 0) throw new IllegalArgumentException("batch must a positive number");
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        synchronized(this) {
            if(karr.isEmpty()) throw new NullPointerException("The Container is empty");
            for(int i=0, size = karr.size(); i<batch; i++) {
                int index = exr.nextInt(0, size - 1);
                ks[i] = karr.get(index);
                vs[i] = varr.get(index);
            }
        }
        return new Pair<>(ks, vs);
    }

    @Override
    public synchronized void clear() {
        karr.clear(); varr.clear();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="class BatchIterater">
    class BIter implements BatchIter<K[], V[]>
    {
        private int index = 0;
        
        @Override 
        public void reset(boolean shuffle) { 
            if(shuffle) shuffle();
            index = 0;  
        }
        
        @Override public boolean hasNext() { return index < karr.size(); }
        
        @Override
        public Pair<K[], V[]> next(int batch) 
        {
            if(batch <= 0) throw new IllegalArgumentException("batch must a positive number");
            K[] ks = (K[]) Array.newInstance(kclazz, batch);
            V[] vs = (V[]) Array.newInstance(vclazz, batch);
            synchronized(ListContainer.this) {
                if(karr.isEmpty()) throw new NullPointerException("The Container is empty");
                for(int i=0, size = karr.size(); i<batch; i++) {
                    int mod_index = index % size;
                    ks[i] = karr.get(mod_index);
                    vs[i] = varr.get(mod_index);
                    index++;
                }
            }
            return new Pair<>(ks, vs);
        }
    }
    //</editor-fold>
    @Override
    public BatchIter<K[], V[]> batch_iter()  { return new BIter(); }
}

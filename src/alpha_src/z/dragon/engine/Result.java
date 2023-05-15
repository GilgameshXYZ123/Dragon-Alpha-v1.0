/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.Arrays;

/**
 * @author Gilgamesh
 * @param <V>
 */
public abstract class Result<V> 
{
    private V value;
    private volatile boolean done = false;
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public boolean isDone() { return done; }
    
    protected abstract V waitResult();
    
    public V get() {
        synchronized(this) { if(!done) { value = waitResult(); done = true; } }
        return value;
    }
    
    public void append(StringBuilder sb) 
    {
        sb.append("Result { value = "); this.get();
        
        Class<?> cls = value.getClass();
        if(cls == float[].class) sb.append(Arrays.toString((float[])value));
        else if(cls == double[].class) sb.append(Arrays.toString((double[])value));
        else if(cls == int[].class) sb.append(Arrays.toString((int[])value));
        else if(cls == long[].class) sb.append(Arrays.toString((long[])value));
        else if(cls == short[].class) sb.append(Arrays.toString((short[])value));
        else if(cls == byte[].class ) sb.append(Arrays.toString((byte[])value));
        else if(cls == boolean[].class) sb.append(Arrays.toString((boolean[])value));
        else sb.append(value); 
        
        sb.append(" }");
    }
    
    @Override
    public String toString() { 
        StringBuilder sb = new StringBuilder(64);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: DualResult<V>">
    public static final class DualResult<V> extends Result<V>
    {
        private final Result<V> rt;
        private final Syncer sc;
        
        public DualResult(Result<V> rt, Syncer sc) {
            this.rt = rt;
            this.sc = sc;
        }

        @Override
        protected V waitResult() {
            V value = rt.get();
            sc.sync();
            return value;
        }
    }
    //</editor-fold>
    public static <V> Result<V> dual(Result<V> rt, Syncer sc) { 
        return new DualResult<>(rt, sc); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="class: IndexedResult<V>">
    public static class IndexedValue<V> 
    {
        private final int index;
        private final V value;
        
        public IndexedValue(int index, V value) {
            this.index = index;
            this.value = value;
        }
        
        public int index() { return index; }
        public V value() { return value; }
    }
    
    //(maxValue, index of the maxValue)
    public static abstract class IndexedResult<V> extends Result<IndexedValue<V>> {
        public int index() { return get().index; }
        public V value() { return get().value; }
    }
    //</editor-fold>
}

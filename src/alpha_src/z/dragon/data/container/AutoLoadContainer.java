/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data.container;

import z.dragon.data.BatchIter;
import java.lang.reflect.Array;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.dragon.data.Pair;
import z.util.math.ExRandom;

/**
 * 
 * @author Gilgamesh
 * @param <K>
 * @param <V> 
 */
@SuppressWarnings(value = "unchecked")
public class AutoLoadContainer<K, V> implements DataContainer<K, V> 
{
    public static interface Loader<K, V> {
        public Pair<K, V> load();
    }
    
    public static interface Triger {
        public boolean needLoad(AutoLoadContainer con, int batch);
    }
    
    //<editor-fold defaultstate="collapsed" desc="class: Full">
    public static final class Full implements Triger 
    {
        static final Full full = new Full();
                
        Full() {}
        
        @Override
        public boolean needLoad(AutoLoadContainer con, int batch) {
            return con.size() < con.capacity();
        }
    }
    //</editor-fold>
    public static Full full() { return Full.full; }
    
    //<editor-fold defaultstate="collapsed" desc="class: Update">
    public static class Update implements Triger 
    {
        private int count = 0;
        private double threshold = 0.5;

        Update(double threshold) {
            this.threshold = threshold;
        }

        @Override
        public boolean needLoad(AutoLoadContainer con, int batch) {
            if (con.size() < con.capacity())  return true;

            //get: batchSize = +batchSize
            count += batch;//load: batchSize = -batchSize
            double percent = ((float) count) / con.capacity();
            return percent >= threshold;
        }
    };
    //</editor-fold>
    public static Update update(double threshold) { return new Update(threshold); }
    
    //<editor-fold defaultstate="collapsed" desc="parameters">
    protected final Class<K> kclazz;
    protected final Class<V> vclazz;
    protected final int capacity;
    protected volatile int size = 0;
    protected final K[] karr; //contain keys
    protected final V[] varr; //contain values
    protected final ExRandom exr = new ExRandom();
    
    private int thread_num;
    private Loader<K, V> loader;
    private Triger triger;
    //</editor-fold>
    public AutoLoadContainer(Class<K> input_cls, Class<V> label_cls,int capacity) 
    {
        if(input_cls == null) throw new NullPointerException("input_class is null");
        if(label_cls == null) throw new NullPointerException("label_class is null");
        
        this.kclazz = input_cls;
        this.vclazz = label_cls;
        this.capacity = capacity;
        karr = (K[]) Array.newInstance(input_cls, capacity);
        varr = (V[]) Array.newInstance(label_cls, capacity);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public int size() {  return size; }
    public int capacity() { return capacity; }

    @Override public Class<K> input_class() { return kclazz; }
    @Override public Class<V> label_class() { return vclazz; }
    
    public int thread_num() { return thread_num; }
    public AutoLoadContainer<K, V> thread_num(int thread_num) {
        if(thread_num <= 0) throw new IllegalArgumentException("thread_num must a positive num");
        this.thread_num = thread_num;
        return this; 
    }
    
    public Loader<K, V> loader() {return loader;}
    public AutoLoadContainer<K, V> loader(Loader<K, V> loader) {
        if(loader == null) throw new NullPointerException("DataLoader is null");
        this.loader = loader;
        return this;
    }

    public Triger triger() { return triger; }
    public AutoLoadContainer<K, V> triger(Triger triger) {
        if(triger == null) throw new NullPointerException("LoaderTriger is null");
        this.triger = triger;
        return this;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="autoload">
    private static final ThreadFactory daemonThreadFactory = new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    };
    private ExecutorService exec;
    protected ExecutorService createExecutorService() {
        return Executors.newFixedThreadPool(thread_num, daemonThreadFactory); 
    }
    
    private int load_batch;
    private volatile boolean running = false;
    private Thread loader_thread;
    
    private void notify_loader(int batchSize) {
        synchronized(loader) {
            if(triger.needLoad(this, batchSize) && running) loader.notify();
        }
    }
    
    private void next_load(int batchSize) {
        try{ 
            synchronized(loader) {
                if(triger.needLoad(this, -batchSize)) loader.notifyAll();
                else loader.wait();
            }
        } catch(InterruptedException e) {}
    }
    
    protected void load(int batch)
    {
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        
        Future[] futures = new Future[batch];
        for(int i=0; i<batch; i++) {
            int index = i;
            futures[i] = exec.submit(()->{
                  Pair<K, V> kv = loader.load();
                    ks[index] = kv.input;
                    vs[index] = kv.label;
            });
        }
        
        for (Future future : futures) {
            try { future.get(); } 
            catch(InterruptedException | ExecutionException e) {}
        }
        this.add(ks, vs);
    }
    
    public void start_load(int batch) 
    {
        if(batch <= 0) throw new IllegalArgumentException("batch must be a postive number");
        if(batch > capacity) batch = capacity;
        
        synchronized(this) {
            load_batch = batch;
            if(running) return; running = true;
            if(exec == null || exec.isShutdown()) exec = createExecutorService();
            
            int first_load = load_batch << 1;
            if(first_load > capacity) first_load = capacity;
            this.load(first_load);
            
            loader_thread = new Thread(()-> {
                while(running) { next_load(load_batch); load(load_batch); }
            });
            loader_thread.start();
        }
    }
    
    public synchronized void stop_load()
    {
        if(!running) return; running = false;
        if(!exec.isShutdown()) exec.shutdown(); exec = null;
        
        this.notifyAll();
        if(!loader_thread.isInterrupted()) loader_thread.interrupt();
        loader_thread = null;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    @Override
    public synchronized void shuffle(double percent) 
    {
        int num = (int) Math.ceil(size * percent);
        if(num > size) num = size; 
        num = (num + 1) >> 1;
    
        for(int i=0; i<num; i++) {
            int index1 = exr.nextInt(0, size - 1);
            int index2 = exr.nextInt(0, size - 1);
            K k = karr[index1]; karr[index1] = karr[index2]; karr[index2] = k;
            V v = varr[index1]; varr[index1] = varr[index2]; varr[index2] = v;
        }
    }
    
    @Override
    public void add(K input, V label) {
        if(input == null || label == null) return;
        synchronized(this) {
            int index = size < capacity ? size++ : exr.nextInt(0, capacity - 1);
            karr[index] = input;
            varr[index] = label;
        }
    }
    
    @Override
    public void add(K[] inputs, V[] labels)  
    {
        if(inputs == null || labels == null) return;
        if(inputs.length != labels.length) throw new IllegalArgumentException(String.format(
                "inputs.length[%d] != labels.length[%d]", inputs.length, labels.length));
        
        synchronized(this) {
            for (int i = 0; i < inputs.length; i++) {
                if(inputs[i] == null || labels[i] == null) continue;
                int index = size < capacity ? size++ : exr.nextInt(0, capacity - 1);
                karr[index] = inputs[i];
                varr[index] = labels[i];
            }
        }
    }

    @Override
    public Pair<K, V> get()  {
        K k; V v;
        synchronized(this) {
            if(size == 0) throw new NullPointerException("The Container is empty");
            int index = exr.nextInt(0, size - 1);
            k = karr[index];
            v = varr[index];
        }
        notify_loader(1);     
        return new Pair<>(k, v);
    }
    
    @Override
   
    public Pair<K[], V[]> get(int batch) 
    {
        if(batch <= 0) throw new IllegalArgumentException("batch must a positive number");
        K[] ks = (K[]) Array.newInstance(kclazz, batch);
        V[] vs = (V[]) Array.newInstance(vclazz, batch);
        synchronized(this) {
            if(size == 0) throw new NullPointerException("The Container is empty");
            for(int i=0; i<batch; i++) {
                int index = exr.nextInt(0, size - 1);
                ks[i] = karr[index];
                vs[i] = varr[index];
            }
        }
        notify_loader(batch);
        return new Pair<>(ks, vs);
    }

    @Override
    public synchronized void clear() {
        for (int i=0; i<size; i++) { karr[i] = null; varr[i] = null; }
        size = 0;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batch_iterator">
    class BIter implements BatchIter<K[], V[]>
    {
        private int index = 0;
        
        @Override
        public void reset(boolean shuffle) {
            if(shuffle) shuffle();
            index = 0; 
        }
        
        @Override public boolean hasNext() {  return index < size; }

        @Override
        public Pair<K[], V[]> next(int batch) 
        {
            if(batch <= 0) throw new IllegalArgumentException("batch must a positive number");
            K[] ks = (K[]) Array.newInstance(kclazz, batch);
            V[] vs = (V[]) Array.newInstance(vclazz, batch);
            synchronized(AutoLoadContainer.this) {
                if(size == 0) throw new NullPointerException("The Container is empty");
                for(int i=0; i<batch; i++) {
                    int mod_index = index % size;
                    ks[i] = karr[mod_index];
                    vs[i] = varr[mod_index];
                    index++;
                }
            }
            notify_loader(batch);
            return new Pair<>(ks, vs);
        }
    }
    //</editor-fold>
    @Override 
    public BatchIter<K[], V[]> batch_iter() { 
        return new BIter();
    } 
}

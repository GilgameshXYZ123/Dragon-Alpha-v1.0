/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.container.DataContainer;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public class DataSet<K, V>
{
    protected DataContainer<K, V> con;
    protected Transform<K[]> ktf;
    protected Transform<V[]> vtf;
    
    public DataSet(DataContainer<K, V> conta) {
        this.con = conta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Transform<K[]>  input_transform() { return ktf; }
    public DataSet<K, V> input_transform(Transform<K[]> input_transform) {
        if(input_transform == null) throw new NullPointerException("input_transform is null");
        this.ktf = input_transform;
        return this;
    }
    
    public Transform<V[]>  label_transform() { return vtf; }
    public DataSet<K, V> label_transform(Transform<V[]> label_transform) {
        if(label_transform == null) throw new NullPointerException("label_transform is null");
        this.vtf = label_transform;
        return this;
    }
    
    public DataContainer<K, V> container() { return con; }
    public DataSet<K, V> container(DataContainer<K, V> conta) {
        if(conta == null) throw new NullPointerException("DataContainer is null");
        this.con = con;
        return this;
    }
    
    public Class<K> input_class() { return con.input_class(); }
    public Class<V> label_class() { return con.label_class(); }
    public int size() { return con.size(); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    private static final ThreadFactory daemonThreadFactory = new ThreadFactory() {
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    };
    private final ExecutorService exec = Executors.newFixedThreadPool(2, daemonThreadFactory); 
    
    public DataSet<K, V> shuffle() { con.shuffle(); return this; }
    public DataSet<K, V> shuffle(double percent) {
        con.shuffle(percent);
        return this;
    }
    
    public TensorPair get(Engine eg) { return get(eg, 1); }
    public TensorPair get(Engine eg, int batch) 
    {
        Pair<K[], V[]> kv = con.get(batch);
        Future<Tensor> finput = exec.submit(() -> { return ktf.transform(eg, kv.input); });
        Future<Tensor> flabel = exec.submit(() -> { return vtf.transform(eg, kv.label); });

        Tensor input, label;
        try {
            input = finput.get();
            label = flabel.get();
        }
        catch(InterruptedException | ExecutionException e) { 
            throw new RuntimeException(e); 
        }
        return new TensorPair(input, label);
    }
    
    public void clear() { con.clear(); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="batch_iterator">
    class BIter implements TensorIter
    {
        private final BatchIter<K[], V[]> iter;
        
        BIter(BatchIter<K[], V[]> iter) { this.iter = iter; }
        
        @Override public void reset(boolean shuffle) { iter.reset(shuffle); }
        @Override public boolean hasNext() { return iter.hasNext(); }

        @Override
        public TensorPair next(Engine eg, int batch) 
        {
            Pair<K[], V[]> kv = iter.next(batch);
            Future<Tensor> finput = exec.submit(() -> { return ktf.transform(eg, kv.input); });
            Future<Tensor> flabel = exec.submit(() -> { return vtf.transform(eg, kv.label); });

            Tensor input, label;
            try {
                input = finput.get();
                label = flabel.get();
            }
            catch(InterruptedException | ExecutionException e) { 
                throw new RuntimeException(e); 
            }
            return new TensorPair(input, label);
        }

      
    }
    //</editor-fold>
    public TensorIter batch_iter() {  return new BIter(con.batch_iter());  }
    
    public BufferedTensorIter buffered_iter(Engine eg, int batch_size) { 
        return new BufferedTensorIter(batch_iter(), eg, batch_size);
    }
}

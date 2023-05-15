/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * 
 * @author Gilgamesh
 */
public class Parameter 
{
    //<editor-fold defaultstate="collapsed" desc="static class: ParamSet">
    public static class ParamSet extends HashSet<Parameter> 
    {
        private static final long serialVersionUID = 1L;
        
        public ParamSet() { super(); }
        public ParamSet(int init_capacity) { super(init_capacity); }

        @Override
        public boolean add(Parameter param) {
            return (isNull(param)? false : super.add(param));
        }
        
        public boolean add(Parameter...params) {
            if(params == null || params.length == 0) return false;
            boolean result = true;
            for(Parameter param : params) {
                result &= (isNull(param)? false : super.add(param));
            }
            return result;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: TensorMap">
    public static class ParamMap<K> extends HashMap<K, Parameter>
    {
        private static final long serialVersionUID = 1L;
        
        public ParamMap() { super(); }
        public ParamMap(int init_capacity) { super(init_capacity); }

        @Override
        public Parameter put(K key, Parameter value) {
            return (isNull(value) ? null: super.put(key, value)); 
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: GradList">
    public static class GradList extends ArrayList<Tensor> 
    {
        private static final long serialVersionUID = 511515125121L;
        private final Parameter param;
        
        public GradList(Parameter p) { super(); this.param = p; }
        public GradList(Parameter p, int init_capacity) { super(init_capacity); this.param = p; }
        
        @Override
        public synchronized boolean add(Tensor ts) {
            if(!param.need_grads() || Tensor.isNull(ts)) return false;
            if(ts.is_grad) throw new IllegalArgumentException("One tensor cannot be the gradient of two tensors");
            
            ts.is_grad = true;
            param.tensor.grad = ts;//map the lastest grad to ts
            return super.add(ts);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: VirtualList">
    public static final class VirtualList extends ArrayList<Tensor> 
    {
        private static final long serialVersionUID = 4441241241L;
        
        VirtualList() { super(); }
        
        @Override public boolean isEmpty() { return true; }
        @Override public int size() { return 0; }
        
        @Override public boolean add(Tensor ts) { return false; }
        @Override public void clear() {}
    }
    
    public static final VirtualList vir_grads = new VirtualList();
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: VirtualParameter">
    public static final class VirtualParameter extends Parameter 
    {
        VirtualParameter(Tensor tensor) { super(tensor, vir_grads); }
        
        @Override public boolean need_grads() { return false; }
        @Override public Parameter need_grads(boolean flag) { return this; }
        
        @Override public List<Tensor> grads() { return grads; }
        @Override public Tensor grad(int index) { return null; }
        @Override public synchronized void clear_grads() {}
    }
    //</editor-fold>
    public static Parameter virual(Tensor tensor) { return new VirtualParameter(tensor); }
    
    protected Tensor tensor;
    protected final List<Tensor> grads;//For parameters: W -> deltaW[0], deltaW[1],....deltaW[n].
   
    protected Parameter(Tensor tensor, List<Tensor> grads) {
        if(tensor == null) throw new NullPointerException("Parameter.tensor must not be  null");
        this.tensor = tensor;
        this.grads = grads;
    }

    public Parameter(Tensor tensor) { 
         if(tensor == null) throw new NullPointerException("Parameter.tensor must not be  null");
        this.tensor = tensor;
        this.grads = new GradList(this, 2);
    }
     
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor ts() { return tensor; }
    public Parameter tensor(Tensor data) { this.tensor = data; return this; } 
    
    public void append(StringBuilder sb) {
        sb.append('<').append(getClass().getSimpleName()).append('>');
        tensor.append(sb);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        this.append(sb);
        return sb.toString();
    }
    
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        delete();
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="running-area: Tensor extended">
    public static boolean isNull(Parameter param) { return param == null || param.tensor.isNull(); }
    public static void delete(Parameter... params) { for(Parameter param : params) if(param != null) param.delete(); }
    public static void sync(Parameter... params) { for(Parameter param : params) if(param != null) param.c(); }
    
    public boolean isNull() { return tensor.isNull(); }
    
    public int[] dim() { return tensor.dim; }
    public int dim(int index) { return tensor.dim(index); }
    
    public Tensor c() { return tensor.c(); }

    public void delete() { tensor.delete(); clear_grads(); }
    
    public boolean need_grads() { return tensor.need_grad; }
    public Parameter need_grads(boolean flag) { tensor.need_grad(flag); return this; }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="running-area: grad_list">
    public List<Tensor> grads() { return grads; }
    
    public Tensor grad(int index) {//0 the earliest grad, -1 the lastest grad
        if(grads.isEmpty()) return null;
        if(index < 0) index = grads.size() + index;
        return grads.get(index);
    }
    
    public synchronized void clear_grads() {
        if(grads.isEmpty()) return;
        grads.forEach((Tensor g) -> { g.delete(); });
        grads.clear();
        tensor.grad = null;//tensor.grad is in grad_list 
    }
    //</editor-fold>
}

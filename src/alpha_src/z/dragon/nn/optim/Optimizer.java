/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateReader;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.optim.lr_schedular.LrSchedular;
import z.dragon.common.state.State.Stateful;
import z.dragon.engine.Parameter;

/**
 *
 * @author Gilgamesh
 */
public abstract class Optimizer implements Stateful, StateReader
{
    protected float lr;
    private LrSchedular lr_schedular;
    
    protected final Parameter[] params;
    protected String[] param_names = null;
    protected HashMap<String, Parameter> param_map;
   
    public Optimizer(Parameter[] params, float lr) {
        if(params == null || params.length == 0) throw new IllegalArgumentException(
                "params == null or params.length == 0");
        
        this.params = new Parameter[params.length];
        System.arraycopy(params, 0, params, 0, params.length);
        this.lr = lr;
    }
    
    public Optimizer(Collection<Parameter> params, float lr) {
        if(params == null || params.isEmpty()) throw new IllegalArgumentException(
                "params == null or params.isEmpty");
        
        this.params = params.toArray(new Parameter[params.size()]);
        this.lr = lr;
    }
    
    public Optimizer(Map<String, Parameter> param_map, float lr)  {
        if(param_map == null || param_map.isEmpty()) throw new IllegalArgumentException(
                "param_map == null or param_map.isEmpty");
        
        this.param_map = new HashMap<>(param_map);
        int size = param_map.size();
        this.params = new Parameter[size];
        this.param_names = new String[size]; int index = 0;
        for(Entry<String, Parameter> kv : param_map.entrySet()) {
            param_names[index] = kv.getKey();
            params[index] = kv.getValue();
            index++;
        }
        this.lr = lr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float learning_rate() { return lr; }
    public Optimizer learning_rate(float lr) {
        this.lr = lr; 
        if(lr_schedular != null) lr_schedular.init(lr);
        return this;
    }
    
    public LrSchedular lrSchedular() { return lr_schedular; }
    public Optimizer lrSchedular(LrSchedular lrSchedular) {
        this.lr_schedular = lrSchedular;
        lr_schedular.init(lr);
        return this;
    }
    
    public Parameter[] params() { return params; }
    public String[] param_names() { return param_names; }
    public Map<String, Parameter> param_map() { return param_map; }
    public Parameter param(String param_name) { return param_map.get(param_name); }
    
    public abstract void append(StringBuilder sb);
    @Override public String toString() { StringBuilder sb = new StringBuilder(256); append(sb); return sb.toString(); }
    public Optimizer println() { System.out.println(this.toString()); return this; }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        this.clear();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    @Override public State state() { State dic = new State(params.length); state(dic); return dic;} 
    @Override public State read() { return state(); }
      
    protected abstract void hypher_state(State dic);
    protected abstract void param_state(State dic, int index, String param_name);
    @Override
    public void state(State dic) {
        if(param_names == null) return;
        hypher_state(dic);
        for(int i=0; i<params.length; i++) 
            param_state(dic, i, param_names[i]);
    }
  
    protected abstract void update_hypher_state(State dic, boolean partial);
    protected abstract void update_param_state(State dic, boolean partial, int index, String param_name);
    @Override
    public void update_state(State dic, boolean partial) {
        if(param_names == null) return;
        update_hypher_state(dic, partial);
        for(int i=0; i<params.length; i++)
            update_param_state(dic, partial, i, param_names[i]);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: max, min, clip">
    public synchronized Optimizer max(float vmax) {
        for(Parameter param : params) param.c().max(true, vmax);
        for(Parameter param : params) param.c();
        return this;
    }
    
    public synchronized Optimizer min(float vmin) {
        for(Parameter param : params) param.c().min(true, vmin);
        for(Parameter param : params) param.c();
        return this;
    }
    
    public synchronized Optimizer clip(float vmin, float vmax) {
        for(Parameter param : params) param.c().clip(true, vmin, vmax);
        for(Parameter param : params) param.c();
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: decay, zero_nan">
    public synchronized Optimizer decay(float L2coef, float L1coef) {
        float alpha = (1 - L2coef) * lr;
        float beta = -lr * L1coef;
        for(Parameter param : params) param.c().linear(true, alpha, beta);
        for(Parameter param : params) param.c();
        return this;
    }
    
    public synchronized Optimizer zero_nan_params() { 
        for(Parameter param : params) param.c().zero_nan();
        for(Parameter param : params) param.c();
        return this;
    }
    
    public synchronized Optimizer zero_nan_gradients() {  
        for(Parameter param : params) {
            List<Tensor> grads = param.grads() ; 
            if(grads.isEmpty()) continue;
            for(Tensor grad : grads) grad.c().zero_nan();
        }
        
        for(Parameter param : params) {
            List<Tensor> grads = param.grads() ; 
            if(grads.isEmpty()) continue;
            for(Tensor grad : grads) grad.c();
        }
        return this;
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="running-area: update">   
    public synchronized Optimizer avg_grads() {
        for(Parameter param : params) {
            List<Tensor> grads = param.grads(); if(grads.size() <= 1) continue;
            Engine eg = param.ts().engine();
            eg.mean(true, grads);//inplace: grad(0) = avg(grads)
        }
        
        for(Parameter param : params) {
            List<Tensor> grads = param.grads(); if(grads.size() <= 1) continue;
            Tensor avg = grads.get(0).c();//wait until the average grads has been calculated
            for(Tensor grad : grads) { if(grad != avg) grad.delete(); }
            grads.clear(); grads.add(avg);//reset grad_list
        }
        return this;
    }
    
    protected abstract void __before_update__();
    protected abstract void __update__(int index, Tensor gradient, Engine eg);
    protected abstract void __update__(int index, Collection<Tensor> gradients, Engine eg);//to have greater batchsize logically
    public synchronized Optimizer update() {
        if(lr == 0) System.err.println("Warning: Learning rate == 0");
        __before_update__();
        
        for(int i=0; i<params.length; i++){
            Parameter param = params[i];
            List<Tensor> grads = param.grads();
            if(grads.isEmpty()) continue;
            
            int gsize = grads.size();//grads, params are synchronized before update
            if (gsize == 1) __update__(i, grads.get(0).c(), param.c().engine());
            else if (gsize > 1) {//to have greater batchsize logically
                for(Tensor grad : grads) grad.c();
                __update__(i, grads, param.c().engine());
            }
        }
        
        if(lr_schedular != null) lr = lr_schedular.next_learning_rate();
        for(Parameter param : params) param.c();
        return this;
    }
    
    public synchronized Optimizer clear_grads() {
        for(Parameter param : params) { param.c(); param.clear_grads(); }
        return this;
    }
    
    protected abstract void __clear__();
    public synchronized void clear() { 
        __clear__(); 
        for(int i=0; i<params.length; i++) {
            params[i].clear_grads(); params[i] = null; 
        }
        lr_schedular = null;
        param_names = null;
    }
    //</editor-fold>
}

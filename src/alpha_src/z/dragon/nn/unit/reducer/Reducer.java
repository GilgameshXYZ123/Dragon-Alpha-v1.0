/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Trace;
import z.dragon.nn.unit.BaseUnit;
import z.dragon.nn.unit.GradientControllable;
import z.dragon.nn.unit.Unit;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class Reducer extends BaseUnit
        implements GradientControllable 
{ 
    private final UnitMap<Object> graph = new UnitMap<>();
    
    private int input_tensor_num = -1;
    
    private Tensor[] X, deltaX;
    private Tensor   Y, deltaY;
    private boolean last_need_grads;
    
    private int[] X_mod_count;
    private int   Y_mod_count;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int input_tensor_num() {return input_tensor_num;}
    public Tensor[] X() { return X; }
    public Tensor   Y() { return Y; }
    
    public Tensor holdX(int index) { return X[index].hold(X_mod_count[index], name); }
    public Tensor holdY() { return Y.hold(Y_mod_count, name); }
    
    public boolean isHoldX(int index) { return X[index].isHold(X_mod_count[index]); }
    public boolean isHoldY(int index) { return Y.isHold(Y_mod_count); }
    
    public Tensor[] deltaX() { return deltaX; }
    public Tensor   deltaY()  {return deltaY; }
    
    @Override public Reducer name(String name) { this.name = name; return this;}
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public boolean isComplex() { return false; }
    
    //after backward: the references for next units will be cleared
    @Override public Collection<Unit> next() { return graph.keySet(); }
    
    private boolean backward_grads_switch = true;
    @Override public boolean backward_grads() { return backward_grads_switch;}
    @Override public Reducer backward_grads(boolean flag) { backward_grads_switch = flag; return this; }
    
    @Override
    public void vars(TensorSet vars) {
        vars.add(X); vars.add(deltaX);
        vars.add(Y, deltaY);
    }
    
    @Override
    public void gc() {
        eg.delete(X); X = null;
        eg.delete(Y); Y = null;
        eg.delete(deltaX); deltaX = null;
        eg.delete(deltaY); deltaY = null;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor __forward__(Tensor[] X);
    
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        graph.clear();//all one-off Unit reference has been cleared
       
        X = input; input_tensor_num = X.length;
        Trace[] Xtrace = new Trace[X.length];
        X_mod_count = new int[X.length];
        for(int i=0; i< X.length; i++) {
           //receive from last: I recevice your output, and it's my ith input
            if((Xtrace[i] = X[i].trace()) != null) Xtrace[i].callback(this, i);
            X_mod_count[i] = X[i].mod_count();//save: X.mod_count
        }
        
        for(Tensor x : X) x.c();//wait until X is cauculated
        Y = __forward__(X);
       
        Y_mod_count = Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        last_need_grads = false;
        for(int i=0; i<X.length; i++) {
            last_need_grads = (last_need_grads || X[i].need_grad());
            if(Xtrace[i] != null) last_need_grads = (last_need_grads || Xtrace[i].need_grads());
        }
        boolean need_grads = (last_need_grads || this.need_grads());
        Y.setTrace(this, 0, need_grads);
        
        return new Tensor[]{ Y };
    }
    
    @Override
    protected synchronized void traceBack(Unit next, int out_index, int next_in_index) {
        Object value = graph.get(next);
        if(value == null) graph.put(next, next_in_index);
        else if(value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(4);
            indexes.add((Integer) value);
            indexes.add(next_in_index);
            graph.put(next, indexes);
        }
        else {
            HashSet<Integer> indexes = (HashSet<Integer>) value;
            indexes.add(next_in_index);
        }
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override //if deltaYarr[0] == null: means the next Scale doesn't backward any gradient
    public synchronized Tensor[] collectGradientFromNext() {
        deltaY = this.aggregateGradient(graph);
        graph.clear();
        return (deltaY == null? null : new Tensor[]{ deltaY });
    }
    
    protected abstract Tensor[] __backward__(Tensor deltaY, int input_tensor_num, 
            boolean grad_inplace, boolean backward_grads);
    
    @Override //1 in multiple out
    public synchronized Tensor[] backward(Tensor... gradient)  {
        if(gradient == null) {//no gradient, no backward
            deltaY = null; deltaX = null;
            return null;
        }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = (backward_grads_switch && last_need_grads);
        deltaX = __backward__(deltaY.c(), input_tensor_num, grad_inplace, backward_grads);
        
        if(deltaX != null) {//collect gradient for X
            for(int i=0; i<X.length; i++) 
                if(X[i].need_grad()) X[i].grad(deltaX[i]);
        }
    
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? deltaX : null);
    }
    
    @Override
    public synchronized Tensor gradient(int index) {
        if(index > deltaX.length) throw new IllegalArgumentException("tensor index out of range");
        return (deltaX == null? null : deltaX[index]);
    }
    //</editor-fold>
}

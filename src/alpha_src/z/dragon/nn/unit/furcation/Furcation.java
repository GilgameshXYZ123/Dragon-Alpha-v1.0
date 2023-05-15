/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Trace;
import z.dragon.nn.unit.BaseUnit;
import z.dragon.nn.unit.GradientControllable;
import z.dragon.nn.unit.Unit;

/**
 * One input, Multiple output.
 * @author Gilgamesh
 */
 @SuppressWarnings("unchecked")
public abstract class Furcation extends BaseUnit
        implements GradientControllable 
{
    private UnitMap<Object>[] arcs;
    private final UnitSet nexts = new UnitSet();
    
    private int output_tensor_num = -1;
    
    private Tensor X, deltaX;
    private Tensor[] Y, deltaY;
    private boolean last_need_grads;
    
    private int   X_mod_count = -1;
    private int[] Y_mod_count;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int output_tensor_num() { return output_tensor_num; }
    public Tensor   X() { return X; }
    public Tensor[] Y() { return Y; }
    
    public Tensor   deltaX() { return deltaX; }
    public Tensor[] deltaY() { return deltaY; }
    
    public Tensor holdX() { return X.hold(X_mod_count, name); }
    public Tensor holdY(int index) { return Y[index].hold(Y_mod_count[index], name); }
    
    public boolean isHoldX() { return X.isHold(X_mod_count); }
    public boolean isHoldY(int index) { return Y[index].isHold(Y_mod_count[index]); } 
    
    @Override public Furcation name(String name) { this.name = name; return this;}
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ruuning-area: others">
    @Override public boolean isComplex() { return false; }
    
    @Override public Collection<Unit> next() { return nexts; }
    
    private boolean backward_grads_switch = true;
    @Override public boolean backward_grads() { return backward_grads_switch;}
    @Override public Furcation backward_grads(boolean flag) { backward_grads_switch = flag; return this; }
    
    //variables && gc-----------------------------------------------------------
    @Override
    public void vars(TensorSet vars) {
        vars.add(X, deltaX);
        vars.add(Y); vars.add(deltaY);
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
    protected abstract Tensor[] __forward__(Tensor X);
    
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        nexts.clear();//all one-off Unit reference has been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X = input[0]; Trace Xtrace = X.trace(); if(Xtrace != null) X.trace().callback(this, 0);
      
        X_mod_count = X.mod_count();//save: X.mod_count
        
        Y = __forward__(X.c());//wait until X is cauculated
      
        Y_mod_count = new int[output_tensor_num = Y.length];//save Y.mod_count
        for(int i=0; i<output_tensor_num; i++) Y_mod_count[i] = Y[i].mod_count();
        
        arcs = new UnitMap[output_tensor_num];//one graph mapped to one output, used to collect gradient
        
        //send trace to the next Unit: send my ith output to you
        last_need_grads = X.need_grad();
        if(Xtrace != null) last_need_grads = (last_need_grads || Xtrace.need_grads());
        boolean need_grads = (last_need_grads || this.need_grads());
        for(int i=0; i<Y.length; i++) Y[i].setTrace(this, i, need_grads);
        
        return Y;
    }

    @Override
    protected synchronized void traceBack(Unit next, int out_index, int next_in_index) {
        nexts.add(next);
        
        if(arcs[out_index] == null) arcs[out_index] = new UnitMap<>(2);
        UnitMap<Object> graph = arcs[out_index];
        
        Object value = graph.get(next);
        if(value == null) graph.put(next, next_in_index);
        else if(value instanceof Integer) {
            HashSet<Integer> indexes = new HashSet<>(4);
            indexes.add((Integer)value);
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
        boolean allNull = true;
        deltaY = new Tensor[arcs.length];
        
        for(int i=0; i<arcs.length; i++) {
            if(arcs[i] == null) continue;
            
            deltaY[i] = this.aggregateGradient(arcs[i]);
            arcs[i] = null;
            
            //if deltaY != null: allNull = (allNull && false) = false
            allNull = allNull && (deltaY[i] == null);//at least one grad != null
        }
        
        arcs = null;
        return (allNull? null : deltaY);
    }

    protected abstract Tensor __backward__(Tensor[] deltaY, //compute deltaX
            boolean[] grad_inplace, boolean backward_grads);
    
    @Override //backward(gradient = deltaY = this.collectGradientFromNext)
    public synchronized Tensor[] backward(Tensor... gradient) {
        if(gradient == null) {//no gradient, no backward
            deltaY = null; deltaX = null;
            return null;
        }
        
        deltaY = gradient;//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean[] grad_inplace = new boolean[output_tensor_num];
        for(int i=0; i<output_tensor_num; i++) grad_inplace[i] = (!Y[i].need_grad());
        boolean backward_grads = (backward_grads_switch && last_need_grads);
        deltaX = __backward__(deltaY, grad_inplace, backward_grads);
        
        if(X.need_grad()) { X.grad(deltaX); }//collect gradient for X
        
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? new Tensor[]{ deltaX } : null);
    }

    @Override
    public synchronized Tensor gradient(int index) {
        if(index != 0) throw new IllegalArgumentException("tensor index out of range");
        return deltaX;
    }
    //</editor-fold>
}

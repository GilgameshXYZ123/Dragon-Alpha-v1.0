/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.nn.unit.GradientControllable;
import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Trace;
import z.dragon.nn.unit.BaseUnit;
import z.dragon.nn.unit.Unit;

/**
 * one input, and one output.
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class SimpleUnit extends BaseUnit//One in and One out
        implements GradientControllable 
{
    private final UnitMap<Object> arc = new UnitMap<>();//only one output
    
    private Tensor X, deltaX;//input, input.grads
    private Tensor Y, deltaY;//output, output.grads
    private boolean last_need_grads;
    
    private int X_mod_count = -1;
    private int Y_mod_count = -1;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor X() { return X; }
    public Tensor Y() { return Y; }
    
    public Tensor deltaX() { return deltaX; }
    public Tensor deltaY() { return deltaY; }
    
    protected Tensor holdX() { return X.hold(X_mod_count, name + ".X"); } 
    protected Tensor holdY() { return Y.hold(Y_mod_count, name + ".Y"); }
    
    protected boolean isHoldX() { return X.isHold(X_mod_count); }
    protected boolean isHoldY() { return Y.isHold(Y_mod_count); }
    
    @Override public SimpleUnit name(String name) { this.name = name; return this;}
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public boolean isComplex() { return false; }
    
    //after backward: the references for next units will be cleared
    @Override public Collection<Unit> next() { return arc.keySet(); }
    
    private boolean backward_grads_switch = true;
    @Override public boolean backward_grads() { return backward_grads_switch;}
    @Override public SimpleUnit backward_grads(boolean flag) { backward_grads_switch = flag; return this; }
    
    @Override public void vars(TensorSet vars) {  vars.add(X, Y, deltaX, deltaY);  }
    
    @Override
    public void gc() {
        eg.delete(X); X = null;
        eg.delete(Y); Y = null;
        eg.delete(deltaX); deltaX = null; 
        eg.delete(deltaY); deltaY = null;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor __forward__(Tensor X);
    
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        arc.clear();//all one-off Unit reference has been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X = input[0]; Trace Xtrace = X.trace(); if(Xtrace != null) Xtrace.callback(this, 0);
      
        X_mod_count = X.mod_count();//save: X.mod_count
        
        Y = __forward__(X.c());//wait until the computation on X is end
        
        Y_mod_count = Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        last_need_grads = X.need_grad();
        if(Xtrace != null) last_need_grads = (last_need_grads || Xtrace.need_grads());
        boolean need_grads = (last_need_grads || this.need_grads());
        Y.setTrace(this, 0, need_grads);
     
        return new Tensor[]{ Y };
    }
    
    @Override
    protected synchronized void traceBack(Unit next, int out_index, int next_in_index) {
        Object value = arc.get(next);
        if(value == null) arc.put(next, next_in_index);//create new arc for next node
        else if(value instanceof Integer) {//output[0] used twice for one specific next node
            HashSet<Integer> indexes = new HashSet<>(2);
            indexes.add((Integer) value);
            indexes.add(next_in_index);
            arc.put(next, indexes);
        }
        else {//value instance of HashSet<Integer>
            HashSet<Integer> indexes = (HashSet<Integer>) value;
            indexes.add(next_in_index);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override//if deltaY == null: means the next Unit doesn't backward any gradient
    public synchronized Tensor[] collectGradientFromNext() {
        deltaY = aggregateGradient(arc);
        arc.clear();
        return (deltaY == null? null : new Tensor[]{ deltaY });
    }
    
    protected abstract Tensor __backward__(Tensor deltaY,//compute deltaX
            boolean grad_inplace, boolean backward_grads);
    
    @Override //backward(gradient = deltaY = this.collectGradientFromNext)
    public synchronized Tensor[] backward(Tensor... gradient) {
        if(gradient == null) {//no gradient, no backward
            deltaY = null; deltaX = null;
            return null;
        }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = (backward_grads_switch && last_need_grads);
        deltaX = __backward__(deltaY.c(), grad_inplace, backward_grads);
        
        if(X.need_grad()) X.grad(deltaX);//collect gradient for X
        
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
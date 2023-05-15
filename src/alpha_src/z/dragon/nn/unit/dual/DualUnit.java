/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual;

import java.util.Collection;
import java.util.HashSet;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Trace;
import z.dragon.nn.unit.BaseUnit;
import z.dragon.nn.unit.GradientControllable;
import z.dragon.nn.unit.Unit;

/**
 * two input, and two output.
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class DualUnit extends BaseUnit
        implements GradientControllable 
{
    private final UnitMap<Object> arc = new UnitMap<>();//only one output
    
    private Tensor X1, deltaX1;//input, input.gradients
    private Tensor X2, deltaX2;
    private Tensor Y, deltaY;//output, output.gradient
    private boolean last_need_grads;
    
    private int X1_mod_count = -1;
    private int X2_mod_count = -1;
    private int Y_mod_count  = -1;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Finctions">
    public Tensor X1() { return X1; }
    public Tensor X2() { return X2; }
    public Tensor Y() { return Y; }
    
    public Tensor deltaX1() { return deltaX1; }
    public Tensor deltaX2() { return deltaX2; }
    public Tensor deltaY() { return deltaY; }
    
    protected Tensor holdX1() { return X1.hold(X1_mod_count, name + ".X1"); } 
    protected Tensor holdX2() { return X2.hold(X2_mod_count, name + ".X2"); } 
    protected Tensor holdY() { return Y.hold(Y_mod_count, name + ".Y"); }
    
    protected boolean isHoldX1() { return X1.isHold(X1_mod_count); }
    protected boolean isHoldX2() { return X2.isHold(X2_mod_count); }
    protected boolean isHoldY() { return Y.isHold(Y_mod_count); }
    
    @Override public DualUnit name(String name) { this.name = name; return this;}
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area：others">
    @Override public boolean isComplex() { return false; }
    
    //after backward: the references for next units will be cleared
    @Override public Collection<Unit> next() { return arc.keySet(); }
    
    @Override
    public void vars(TensorSet vars) {
        vars.add(X1, X2, Y, deltaX1, deltaX2, deltaY);
    }
    
    @Override
    public void gc() {
        eg.delete(X1); X1 = null;
        eg.delete(X2); X2 = null;
        eg.delete(Y); Y = null;
        eg.delete(deltaX1); deltaX1 = null;
        eg.delete(deltaX2); deltaX2 = null;
        eg.delete(deltaY); deltaY = null;
    }
   
    private boolean backward_grads_switch = true;
    @Override public boolean backward_grads() { return backward_grads_switch;}
    @Override public DualUnit backward_grads(boolean flag) { backward_grads_switch = flag; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected abstract Tensor __forward__(Tensor X1, Tensor X2);
    
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        arc.clear();//all one-off Unit reference has been cleared
        
        //receive from last: I recevice your output, and it's my 0th input
        X1 = input[0]; Trace X1trace = X1.trace(); if(X1trace != null) X1trace.callback(this, 0);
        X2 = input[1]; Trace X2trace = X2.trace(); if(X2trace != null) X2trace.callback(this, 1);
        
        X1_mod_count = X1.mod_count();//save: X1.mod_count
        X2_mod_count = X2.mod_count();//save: X2.mod_count
        
        Y = __forward__(X1.c(), X2.c());//wait until the computation on [X1, X2] is end
        
        Y_mod_count  =  Y.mod_count();//save: Y.mod_count
        
        //send trace to the next: I will send my 0th output to you
        boolean last_need_grads1 = X1.need_grad();
        boolean last_need_grads2 = X2.need_grad();
        if(X1trace != null) last_need_grads1 = (last_need_grads1 || X1trace.need_grads());
        if(X2trace != null) last_need_grads2 = (last_need_grads2 || X2trace.need_grads());
        last_need_grads = (last_need_grads1 || last_need_grads2);
        boolean need_grads = (last_need_grads || this.need_grads());
        Y.setTrace(this, 0, need_grads);
        
        return new Tensor[]{ Y };
    }

    @Override
    protected synchronized void traceBack(Unit next, int out_index, int next_in_index) {
        Object value = arc.get(next);
        if(value == null) arc.put(next, next_in_index);
        else if(value instanceof Integer) {
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
    @Override //if deltaYarr[0] == null: means the next Unit doesn't backward any gradient
    public synchronized Tensor[] collectGradientFromNext() {
        deltaY = aggregateGradient(arc); 
        arc.clear();
        return (deltaY == null? null : new Tensor[]{ deltaY });
    }

    protected abstract Tensor[] __backward__(Tensor deltaY,//compute deltaX1 and deltaX2
            boolean grad_inplace, boolean backward_grads);
    
    @Override //backward(gradient = deltaY = this.collectGradientFromNext)
    public synchronized Tensor[] backward(Tensor... gradient) {
        if(gradient == null) {//no gradient, no backward
            deltaY = null; deltaX1 = null; deltaX2 = null;
            return null;
        }
        
        deltaY = gradient[0];//deltaYarr[0] = gradient[0] = deltaYarr[0]
        
        if(before_backward != null) before_backward.callback(this);
        
        //backward_grads: when last unit need grads, and backward_grads_switch is on
        boolean grad_inplace = (!Y.need_grad());
        boolean backward_grads = (backward_grads_switch && last_need_grads);
        Tensor[] deltaX = __backward__(deltaY.c(), grad_inplace, backward_grads);
        
        if(deltaX != null) {//collect gradient for [X1, X2]
            deltaX1 = deltaX[0]; deltaX2 = deltaX[1];//[deltaX1, deltaX2]
            if(X1.need_grad()) X1.grad(deltaX1);
            if(X2.need_grad()) X2.grad(deltaX2);
        }
        
        if(after_backward != null) after_backward.callback(this);
        
        return (backward_grads? deltaX : null);//deltaXarr.length == 2
    }

    @Override
    public synchronized Tensor gradient(int index) {//index == 0: deltaX1; index == 1: deltaX2
        if(index > 1) throw new IllegalArgumentException("tensor index out of range");
        return (index == 0? deltaX1 : deltaX2);
    }
    //</editor-fold>
}

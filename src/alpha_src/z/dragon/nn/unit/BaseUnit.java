/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.function.BiConsumer;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorList;
import z.dragon.nn.unit.BackwardHookable.Hook;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class BaseUnit extends Unit implements BackwardHookable
{
    protected Engine eg;
    public Engine engine() {return eg;}
    
    //<editor-fold defaultstate="collapsed" desc="running-area: init">
    @Override public BaseUnit name(String name) { this.name = name; return this;}
    
    protected abstract void __init__(Engine eg);
    @Override public final <T extends Unit> T init(Engine eg, String name) { 
        this.name = name;
        __init__(this.eg = eg); 
        return (T) this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: gradient-aggregation">
    //<editor-fold defaultstate="collapsed" desc="class: GradientAggregator">
    public static class GradientAggregator implements BiConsumer<Unit, Object>
    {
        private final TensorList grads = new TensorList(4);
        public TensorList grads() { return grads; }
        
        @Override
        public void accept(Unit next, Object value) {
            if(value == null) return;
            if(value instanceof Integer) {//used as 1 input: 1 -> 1
                grads.add(next.gradient((int) value));
            }
            else {//used as multiple input: 1 -> m
                HashSet<Integer> indexes = (HashSet<Integer>) value;
                for(int index : indexes) grads.add(next.gradient(index));
            }
        }
    }
    //</editor-fold>
    private final GradientAggregator aggr = new GradientAggregator();
    
    protected final Tensor aggregateGradient(Map<Unit, Object> arc) {
        arc.forEach(aggr); 
        TensorList grads = aggr.grads;
        
        if(grads.isEmpty()) return null;

        Tensor deltaY;
        if(grads.size() == 1) { deltaY = grads.get(0); grads.clear(); }
        else//find the summary of gradients
        {
            for(Tensor grad : grads) grad.c();//wait all grads are cauculated
            
            deltaY = eg.summary(true, grads);

            for(Tensor grad : grads) deltaY.carry(grad);//gc for oneOff Units, if grad.need_carry = true
            
            deltaY.dual(()-> {//when deltaY is cauculated, grad[1:n-1] are not in need
                Iterator<Tensor> iter = grads.iterator();
                for(iter.next(); iter.hasNext(); ) iter.next().delete();//exclude grad[0]
                grads.clear();
            });
        }
        return deltaY;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: hook-mechanism">
    protected Hook before_backward;
    protected Hook after_backward;
    
    @Override
    public <T extends Unit> T hook_before_backward(Hook hook) {
        before_backward = hook;
        return (T) this;
    }

    @Override
    public <T extends Unit> T hook_after_backward(Hook hook) {
        after_backward = hook;
        return (T) this;
    }
    //</editor-fold>
}

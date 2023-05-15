/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Train2Eval;

/**
 *
 * @author Gilgamesh
 */
public class Dropout extends SimpleInplaceFunction<Dropout> 
        implements Train2Eval
{
    protected float p;
    protected Tensor R;
    
    public Dropout(boolean inplace, float nonzero_p) {
        super(inplace);
        nonzero_percent(nonzero_p);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float nonzero_percent() { return p; }
    public final Dropout nonzero_percent(float nonzero_p) {
        if(nonzero_p == 0) throw new IllegalArgumentException("nonzero_p can't be zero");
        this.p = nonzero_p;
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", nonzero_percent = ").append(p).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void gc() {
        super.gc(); 
        eg.delete(R); R = null;
    }
    
    @Override
    public void vars(TensorSet set) {
        super.vars(set);
        set.add(R);
    }
    
    protected boolean training = true;
    @Override public boolean training() { return training; }
    @Override public Dropout train() { this.training = true; return this; }
    @Override public Dropout eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        if(!training) return (inplace? X : eg.copy(X));//exp = (1/p)*p + (1 - p)*0

        Tensor[] outs = eg.dropout(X, p);
        R = outs[1];//R = eg.Bernouli(p, pr, 0, X.dim());
        return outs[0];//Y = outs[0] = g.mul(inplace, X, R.c());
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? //when deltaX is cauculated, rber is not needed
                eg.mul(grad_inplace, deltaY, R).dual(()->{ R.delete(); }):
                null);
    }
    //</editor-fold>
}

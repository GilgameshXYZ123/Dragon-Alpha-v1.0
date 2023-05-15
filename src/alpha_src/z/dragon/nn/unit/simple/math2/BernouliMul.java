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
public class BernouliMul extends SimpleInplaceFunction<BernouliMul> 
        implements Train2Eval
{
    protected float p, v1, v2;
    protected Tensor R;
    
    public BernouliMul(boolean inplace, float p, float v1, float v2) {
        super(inplace);
        this.p = p;
        this.v1 = v1;
        this.v2 = v2;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float expect() { return v1*p + v2*(1.0f - p); } 
    
    public float p() { return p; }
    public BernouliMul p(float p) { this.p = p; return this; }
    
    public float v1() { return v1; }
    public BernouliMul v1(float v1) { this.v1 = v1; return this; }
    
    public float v2() { return v2; }
    public BernouliMul v2(float v2) { this.v2 = v2; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", p = ").append(p);
        sb.append(", [v1, v2] = [").append(v1).append(", ").append(v2).append(" ]");
        sb.append(" }");
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
    @Override public BernouliMul train() { this.training = true; return this; }
    @Override public BernouliMul eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        if(!training) return eg.linear(inplace, expect(), X, 0);//exp = (v1*p + v2*(1.0f - p))
        
        Tensor[] outs = eg.bernouli_mul(X, p, v1, v2);
        R = outs[1];
        return outs[0];//Y = outs[0]
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? //when deltaX is cauculated, rber is not needed
                eg.mul(grad_inplace, deltaY, R).dual(()->{ R.delete(); }):
                null);
    }
    //</editor-fold>
}

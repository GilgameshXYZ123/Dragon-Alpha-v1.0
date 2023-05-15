/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Elu extends SimpleInplaceFunction<Elu>
{   
    protected float alpha;
    protected float k;
    
    public Elu(boolean inplace, float alpha, float negative_slope) {
        super(inplace);
        this.alpha = alpha;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Elu alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float negative_slope() { return k; }
    public Elu negative_slope(float negative_slope) { k = negative_slope; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", negative_slope = ").append(k).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.elu(inplace, X, alpha, k);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (isHoldY()?
                eg.elu_deltaX_v1(grad_inplace, deltaY, holdY(), alpha, k): //V1: Y is not changed
                eg.elu_deltaX_v2(grad_inplace, deltaY, holdX(), alpha, k));//V2: X is not changed
    }
    //</editor-fold>
}

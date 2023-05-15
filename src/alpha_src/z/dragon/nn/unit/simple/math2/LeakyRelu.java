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
public class LeakyRelu extends SimpleInplaceFunction<LeakyRelu>
{
    protected float k;
    
    public LeakyRelu(boolean inplace, float negative_slope) {
        super(inplace);
        this.k = negative_slope;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float negative_slop() { return k; }
    public LeakyRelu negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", negative_slope = ").append(k).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.leakyRelu(inplace, X, k);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (isHoldY()? 
                eg.leakyRelu_deltaX_v1(grad_inplace, deltaY, holdY(), k): //V1: Y is not changed
                eg.leakyRelu_deltaX_v2(grad_inplace, deltaY, holdX(), k));//V2: X is not changed
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math1;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Cos extends SimpleFunction
{
    protected float alpha;
    protected float beta;

    public Cos(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Cos alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Cos beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Tensor X) {
        return eg.cos(false, alpha, X, beta);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.cos_deltaX(grad_inplace, deltaY, holdX(), alpha, beta) :
                null);
    }
    //</editor-fold>
}

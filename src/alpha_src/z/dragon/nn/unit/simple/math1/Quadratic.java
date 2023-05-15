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
public class Quadratic extends SimpleFunction
{
    protected float alpha;
    protected float beta;
    protected float gamma;
    
    public Quadratic(float alpha, float beta, float gamma) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Quadratic alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Quadratic setBeta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Quadratic setGamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Tensor X) {
        return eg.quadratic(false, X, alpha, beta, gamma);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.quadratic_deltaX(grad_inplace, deltaY, holdX(), alpha, beta):
                null);
    }
    //</editor-fold>
}

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
public class Rpl extends SimpleInplaceFunction<Rpl>
{
    private float alpha;
    private float beta;
    private float gamma;
    
    public Rpl(boolean inplace, float alpha, float beta, float gamma) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Rpl alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Rpl beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Rpl gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", beta = ").append(beta);
        sb.append(", alpha = ").append(alpha);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.rpl(inplace, alpha, X, beta, gamma);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.rpl_deltaX(grad_inplace, deltaY, holdY(), alpha, gamma):
                null);
    }
    //</editor-fold>
}

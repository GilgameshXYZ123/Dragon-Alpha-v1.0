/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 * Y = alpha * X + beta.
 * alpha and beta are constants
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Linear extends SimpleInplaceFunction<Linear>
{
    protected float alpha;
    protected float beta;
    
    public Linear(boolean inplace, float alpha, float beta) {
        super(inplace);
        this.alpha = alpha;
        this.beta = beta;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Linear alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Linear beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.linear(inplace, alpha, X, beta);
    }

    //Y = alpha*X + beta
    //deltaX = alpha * deltaY
    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.linear(grad_inplace, alpha, deltaY, 0):
                null);
    }
    //</editor-fold>
}

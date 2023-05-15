/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math2;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Linear2 extends DualLikeFunction<Linear2>
{
    protected float alpha, beta, gamma;
      
    public Linear2(boolean likeX1, 
            float alpha, float beta, float gamma)
    {
        super(likeX1);
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public Linear2 alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public Linear2 beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() { return gamma; }
    public Linear2 gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1());
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override //(1) Y = alpha*X1 + beta*X2 + gamma
    protected Tensor __forward__(Tensor X1, Tensor X2, boolean likeX1) {
        return eg.linear2(false, likeX1, X1, X2, alpha, beta, gamma);
    }

    @Override//(2) [deltaX1, deltaX2] = [deltaY*alpha + deltaY*beta]
    protected Tensor[] __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads ? 
                eg.linear_2out(grad_inplace, deltaY, alpha, 0, beta, 0): //{deltaX1, deltaX2}
                null);
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class HalfSin extends SimpleInplaceFunction<HalfSin> 
{
    protected float amp;
    protected float alpha;
    protected float beta;

    public HalfSin(boolean inplace, float amp, float alpha, float beta) {
        super(inplace);
        this.amp = amp;
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float amp() {return alpha;}
    public HalfSin amp(float ampltitude) {this.amp = ampltitude; return this;}
    
    public float alpha() {return alpha;}
    public HalfSin setAlpha(float alpha) {this.alpha = alpha; return this;}
    
    public float beta() {return beta;}
    public HalfSin beta(float beta) {this.beta = beta; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", amp = ").append(amp);
        sb.append(", alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.halfSin(inplace, amp, alpha, X, beta);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.halfSin_deltaX(grad_inplace, deltaY, holdY(), amp, alpha) :
                null);
    }
    //</editor-fold>
}

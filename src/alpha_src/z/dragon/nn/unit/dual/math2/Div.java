/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math2;

import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Div extends DualLikeFunction<Div>
{
    protected float alpha1, beta1;
    protected float alpha2, beta2;
    protected float gamma;
    
    public Div(boolean likeX1, 
            float alpha1, float beta1,
            float alpha2, float beta2,
            float gamma)
    {
        super(likeX1);
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.gamma = gamma;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha1() { return alpha1; }
    public Div alpha1(float alpha1) { this.alpha1 = alpha1; return this; }
    
    public float beta1() { return beta1; }
    public Div beta1(float beta1) { this.beta1 = beta1; return this; }
    
    public float alpha2() { return alpha2; }
    public Div alpha2(float alpha2) { this.alpha2 = alpha2; return this; }
    
    public float beta2() { return beta2; }
    public Div beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float gamma() { return gamma; }
    public Div gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1());
        sb.append(", [alpha1, beta1] = [").append(alpha1).append(", ").append(beta1).append(']');
        sb.append(", [alpha2, beta2] = [").append(alpha2).append(", ").append(beta2).append(']');
        sb.append(", gamma = ").append(gamma).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X1, Tensor X2, boolean likeX1) {
        return eg.div(false, likeX1, //inplace = false
                alpha1, X1, beta1,
                alpha2, X2, beta2,
                gamma);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor[] deltaX = eg.div_deltaX(false, deltaY, 
                        holdX1(), alpha1, beta1, 
                        holdX2(), alpha2, beta2);
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
            Tensor deltaX1 = deltaX[0], deltaX2 = deltaX[1];
            CountGc gc = new CountGc(2, deltaY);
            deltaX1.dual(()-> { gc.countDown(); });
            deltaX2.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}

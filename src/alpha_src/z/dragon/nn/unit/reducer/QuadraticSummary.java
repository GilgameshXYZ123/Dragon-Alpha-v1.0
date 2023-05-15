/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class QuadraticSummary extends ReduceFunction
{
    protected float alpha;
    protected float beta;
    protected float gamma;
    
    public QuadraticSummary(float alpha, float beta, float gamma) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() {return alpha;}
    public QuadraticSummary alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() {return beta;}
    public QuadraticSummary beta(float beta) { this.beta = beta; return this; }
    
    public float gamma() {return beta;}
    public QuadraticSummary gamma(float gamma) { this.gamma = gamma; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append("{ alpha = ").append(alpha);
        sb.append(", beta = ").append(beta);
        sb.append(", gamma = ").append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor[] X) {
        return eg.quadratic_summary(false, alpha, beta, gamma, X);
    }
    
    //Y = sum(alpha*X^2 + beta*X + gamma
    //Y' = 2*alpha*X + beta
    //deltaX = deltaY * Y' = deltaY * (2*alpha*X + beta)
    //deltaX = 2*alpha * (deltaY*X) + beta * deltaY
    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num) {
        Tensor[] deltaX = new Tensor[input_tensor_num];
        
        float alpha2 = 2.0f * alpha;
        for(int i=0; i<deltaX.length; i++) //deltaX = 2*alpha * (deltaY*X) + beta * deltaY
            deltaX[i] = eg.quadratic2(false, deltaY, holdX(i), 
                    0.0f, alpha2, 0, 
                    beta, 0, 0);
        
        return deltaX;
    }
    //</editor-fold>
}
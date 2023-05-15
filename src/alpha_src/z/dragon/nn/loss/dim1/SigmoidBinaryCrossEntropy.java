/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss.dim1;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class SigmoidBinaryCrossEntropy extends Loss1D
{   
    protected float alpha = 1.0f;
    protected float beta = 1.0f;
    
    public SigmoidBinaryCrossEntropy(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public SigmoidBinaryCrossEntropy alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public SigmoidBinaryCrossEntropy beta(float beta) { this.beta = beta; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(this.getClass().getSimpleName());
        sb.append("{ average = ").append(average());
        sb.append(", zero_nan = ").append(zero_nan());
        sb.append("[alpha, beta] = [").append(alpha).append(", ").append(beta);
        sb.append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __loss_tensor__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.sigmoid_binaryCrossEntropy(Yh, Y, alpha, beta);
    }

    @Override
    protected Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.sigmoid_binaryCrossEntropy_deltaX(Yh, Y, alpha, beta);
    }
    //</editor-fold>
}

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
public class BinaryCrossEntropy extends Loss1D
{
    protected float alpha;
    protected float beta;
    
    public BinaryCrossEntropy(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() { return alpha; }
    public BinaryCrossEntropy alpha(float alpha) { this.alpha = alpha; return this; }
    
    public float beta() { return beta; }
    public BinaryCrossEntropy beta(float beta) { this.beta = beta; return this; }
    
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
        return eg.binaryCrossEntropy(Yh, Y, alpha, beta);
    }

    @Override
    protected Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.binaryCrossEntropy_deltaYh(Yh, Y, alpha, beta);
    }
    //</editor-fold>
}

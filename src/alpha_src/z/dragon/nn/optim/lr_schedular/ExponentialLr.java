/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim.lr_schedular;


/**
 *
 * @author Gilgamesh
 */
public final class ExponentialLr extends LrSchedular
{
    private float gamma;
    private float minLr;
    
    public ExponentialLr(float gamma, float minLr) {
        if(gamma <= 0 || gamma > 1) throw new IllegalArgumentException("lamba belongs to (0, 1)");
        this.gamma = gamma;
        this.minLr = minLr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float gamma() { return gamma; }
    public ExponentialLr setGamma(float lamba) {this.gamma = lamba; return this;}
    
    public float minLearningRate() { return gamma; }
    public ExponentialLr setMinLearningRate(float minLr) {this.minLr = minLr; return this;}
    
    @Override
    public void append(StringBuilder sb) {
        sb.append("Exponential Learning Rate Schedular {")
                .append("init_learning_rate = ").append(initLr)
                .append("learning_rate = ").append(lr)
                .append(", gamma = ").append(gamma)
                .append(", min_learning_rate = ").append(minLr).append("}");
    }
    //</editor-fold>

    @Override
    public float next_learning_rate() {
        return lr = Math.max(minLr, lr * gamma);
    }
}

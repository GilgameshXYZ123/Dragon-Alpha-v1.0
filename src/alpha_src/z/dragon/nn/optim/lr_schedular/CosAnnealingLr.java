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
public final class CosAnnealingLr extends LrSchedular
{
    private float minLr;
    private float Tmax;
    private int epoch;
    private float threshold;
    private float period;
    
    public CosAnnealingLr(float Tmax, float minLr) {
        if(Tmax == 0) throw new IllegalArgumentException("Tmax can't be zero.");
        this.Tmax = Tmax;
        this.minLr = minLr;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float min_learning_rate() {return minLr;}
    public CosAnnealingLr setMinLearningRate(float minLr) {this.minLr = minLr; return this;}
    
    public float Tmax() {return Tmax;}
    public CosAnnealingLr setTmax(float Tmax) {this.Tmax = Tmax; return this;}
    
    @Override
    public void append(StringBuilder sb) {
        sb.append("Cosine Annealing Learning Rate Schedular {")
                .append("init_learning_rate = ").append(initLr)
                .append("learning_rate = ").append(lr)
                .append("min_learning_rate = ").append(minLr)
                .append("Tmax = ").append(Tmax);
    }
    //</editor-fold>

    @Override
    public void init(float initLr) {
        super.init(initLr); 
        this.epoch = 0;
        this.threshold = initLr - minLr;
        this.period = (float) (Math.PI / Tmax);
    }

    @Override
    public float next_learning_rate() {
        return lr = (float) (minLr + threshold * (1 + Math.cos(epoch++ * period)));
    }
}

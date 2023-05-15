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
public abstract class LrSchedular 
{
    protected float initLr;
    protected float lr;
    
    public abstract void append(StringBuilder sb);
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    
    public void init(float initLr) {
        this.initLr = initLr;
        this.lr = initLr;
    }
    
    public float init_learning_rate() { return initLr; }
    
    public float learining_rate() {return lr;}
    public abstract float next_learning_rate();
}

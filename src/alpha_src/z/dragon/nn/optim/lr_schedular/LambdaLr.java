/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim.lr_schedular;

import java.util.function.Function;

/**
 *
 * @author Gilgamesh
 */
public final class LambdaLr extends LrSchedular
{
    private Function<Float, Float> updater;
    
    public LambdaLr(Function<Float, Float> updater) {
        if(updater == null) throw new NullPointerException();
        this.updater = updater;
    }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append("Lambdar Learning Rate Schedular {")
                .append("learning_rate = ").append(lr).append('}');
    }

    @Override
    public float next_learning_rate() {
        return lr = updater.apply(lr);
    }
}

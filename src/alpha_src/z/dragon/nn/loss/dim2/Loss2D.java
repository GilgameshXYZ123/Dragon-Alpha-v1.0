/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss.dim2;

import z.dragon.engine.Engine;
import z.dragon.engine.Result;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;

/**
 *
 * @author Gilgamesh
 */
public abstract class Loss2D extends LossFunction
{
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override
    protected Result<Float> mean_loss(Tensor loss, Engine eg) {
        int batch = loss.length() / loss.lastDim();
        float mean_coef = 1.0f / batch;
        return eg.straight_sum(loss, mean_coef);//sum(loss) / batch
    }

    @Override
    protected Tensor mean_gradient(Tensor grad, Engine eg) {
        int batch = grad.length() / grad.lastDim();
        return eg.sdiv(true, grad, batch);
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss.dim1;

import z.dragon.engine.Engine;
import z.dragon.engine.Result;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;

/**
 *
 * @author Gilgamesh
 */
public abstract class Loss1D extends LossFunction
{
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override
    protected Result<Float> mean_loss(Tensor loss, Engine eg) {
        return eg.straight_mean(loss);
    }

    @Override
    protected Tensor mean_gradient(Tensor grad, Engine eg) {
        return eg.sdiv(true, grad, grad.length());
    }
    //</editor-fold>
}

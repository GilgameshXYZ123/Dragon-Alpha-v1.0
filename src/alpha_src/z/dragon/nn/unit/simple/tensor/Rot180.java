/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.math2.SimpleInplaceFunction;

/**
 *
 * @author Gilgamesh
 */
public class Rot180 extends SimpleInplaceFunction<Rot180>
{
    public Rot180(boolean inplace) { 
        super(inplace); 
    }

    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.rot180(inplace, X);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.rot180(grad_inplace, deltaY) : 
                null);
    }
    //</editor-fold>
}

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
public class Flatten extends SimpleInplaceFunction<Flatten>
{
    protected int[] inDim;
    protected int[] outDim;
    
    public Flatten(boolean inplace) {
        super(inplace);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int[] in_dim() { return inDim; }
    public int[] out_dim() { return outDim; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        inDim = X.dim();
        outDim = new int[] { X.dim(0), X.length() / X.dim(0) };
        return eg.reshape(inplace, X, outDim);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.reshape(grad_inplace, deltaY, inDim):
                null);
    }
    //</editor-fold>
}

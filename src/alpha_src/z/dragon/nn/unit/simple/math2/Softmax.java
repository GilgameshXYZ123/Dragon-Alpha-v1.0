package z.dragon.nn.unit.simple.math2;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Softmax extends SimpleInplaceFunction<Softmax>
{
    protected int features;
    
    public Softmax(boolean inplace, int features) {
        super(inplace);
        this.features = features;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int features() { return features; }
    public Softmax features(int features) { this.features = features; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", features = ").append(features).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.softmax(inplace, X, features);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.softmax_deltaX(grad_inplace, deltaY, holdY(), features):
                null);
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss.dim2;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class SoftmaxCrossEntropy extends Loss2D
{
    protected int features;
    
    public SoftmaxCrossEntropy(int features) {
        this.features = features;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int features() { return features; }
    public SoftmaxCrossEntropy features(int features) {
        this.features = features;
        return this;
    }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{ average = ").append(average());
        sb.append(", zero_nan = ").append(zero_nan());
        sb.append(", features = ").append(features);
        sb.append(" }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __loss_tensor__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.softmax_crossEntropy(Yh, Y, features);
    }

    @Override
    protected Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.softmax_crossEntropy_deltaX(Yh, Y, features);
    }
    //</editor-fold>
}

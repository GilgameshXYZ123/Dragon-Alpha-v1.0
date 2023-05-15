/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class AdaptiveNaiveMaxPool2D extends AdaptivePool2D
{
    public AdaptiveNaiveMaxPool2D (int out_height, int out_width) {
         super(out_height, out_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">  
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append("AdpativeMaxPool2D Naive { ");
        sb.append("out_size = [").append(OH).append(", ").append(OW).append("]}");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running area">
    @Override
    protected Tensor __forward__(Tensor X) {
         __adaptive__(X);
        return eg.pool2D_max(X, FH, FW, sh, sw, 0, 0);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.upool2D_max(deltaY, holdY(), holdX(), FH, FW, sh, sw, 0, 0);
        
        //when deltaX is cauculated, deltaY is not needed
        if(grad_inplace) deltaX.dual(()-> { deltaY.delete(); });
        
        return deltaX;
    }
    //</editor-fold>
}

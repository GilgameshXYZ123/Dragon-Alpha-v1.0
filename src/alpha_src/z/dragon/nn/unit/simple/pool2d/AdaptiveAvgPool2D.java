/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class AdaptiveAvgPool2D extends AdaptivePool2D
{
    protected boolean ignore_padding;
    
    public AdaptiveAvgPool2D(boolean ignore_padding,
            int out_height, int out_width) {
        super(out_height, out_width);
        this.ignore_padding = ignore_padding;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AdaptiveAvgPool2D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", out_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running area">
    @Override
    protected Tensor __forward__(Tensor X) {
        __adaptive__(X);
        return (ignore_padding? 
                eg.pool2D_avg_ignore_padding(X, FH, FW, sh, sw, 0, 0):
                eg.pool2D_avg(X, FH, FW, sh, sw, 0, 0));
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = (ignore_padding?
                eg.upool2D_avg_ignore_padding(deltaY, IH, IW, FH, FW, sh, sw, 0, 0):
                eg.upool2D_avg(deltaY, IH, IW, FH, FW, sh, sw, 0, 0));
        
        //when deltaX is cauculated, deltaY is not needed
        if(grad_inplace) deltaX.dual(()-> { deltaY.delete(); });
        
        return deltaX;
    }
    //</editor-fold>
}

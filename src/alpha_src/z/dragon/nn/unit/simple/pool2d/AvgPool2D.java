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
public class AvgPool2D extends Pool2D
{
    protected boolean ignore_padding;
    protected int IH, IW;
    
    public AvgPool2D(boolean ignore_padding,
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width, 
            int padding_height, int padding_width) 
    {
        super(kernel_height, kernel_width,
              stride_height, stride_width,
              padding_height, padding_width);
        this.ignore_padding = ignore_padding;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean ignore_padding() { return ignore_padding; }
    public AvgPool2D ignore_padding(boolean flag) { this.ignore_padding = flag; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { ignore_padding = ").append(ignore_padding);
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append("], ");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("], ");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running area">
    @Override
    protected Tensor __forward__(Tensor X) {
        int[] dim = X.dim(); IH = dim[1]; IW = dim[2];
        return (ignore_padding? 
                eg.pool2D_avg_ignore_padding(X, FH, FW, sh, sw, ph, pw):
                eg.pool2D_avg(X, FH, FW, sh, sw, ph, pw)); 
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = (ignore_padding?
                eg.upool2D_avg_ignore_padding(deltaY, IH, IW, FH, FW, sh, sw, ph, pw):
                eg.upool2D_avg(deltaY, IH, IW, FH, FW, sh, sw, ph, pw));
        
        //when deltaX is cauculated, deltaY is not needed
        if(grad_inplace) deltaX.dual(()-> { deltaY.delete(); });
        
        return deltaX;
    }
    //</editor-fold>
}

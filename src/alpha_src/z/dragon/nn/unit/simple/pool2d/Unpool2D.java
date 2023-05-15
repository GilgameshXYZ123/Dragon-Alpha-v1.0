/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.math1.SimpleFunction;

/**
 *
 * @author Gilgamesh
 */
public abstract class Unpool2D extends SimpleFunction
{
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int IH, IW;
    
    public Unpool2D(//A reverse of AvgPool2D
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width, 
            int padding_height, int padding_width,
            int output_height, int output_width) 
    {
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.IH = output_height;  this.IW = output_width;//A reverse of AvgPool2D
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int[] kernel() { return new int[]{ FH, FW }; }
    public int[] stride() { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw };}
    public int[] out_size() { return new int[] { IH, IW }; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        sb.append(" kernel = [").append(FH).append(", ").append(FW).append(']');
        sb.append(", stride = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(']');
        sb.append(", output_size = [").append(IH).append(", ").append(IW).append("] }");
    }
    //</editor-fold>

    protected void __compute_output_size(Tensor X) {
        if(IH == -1) IH = (X.dim(1) - 1)*sh + FH - (ph << 1);
        if(IW == -1) IW = (X.dim(2) - 1)*sw + FW - (pw << 1);
    }
}

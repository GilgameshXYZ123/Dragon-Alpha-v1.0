/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.nn.unit.simple.math1.SimpleFunction;

/**
 *
 * @author Gilgamesh
 */
public abstract class Pool2D extends SimpleFunction
{
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    
    protected Pool2D(int kernel_height, int kernel_width, 
            int step_height, int step_width,
            int padding_height, int padding_width)
    {
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = step_height;    this.sw = step_width;
        this.ph = padding_height; this.pw = padding_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">  
    public int[] kernel() { return new int[]{ FH, FW }; }
    public int[] stride() { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw }; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { kernel = [").append(FH).append(", ").append(FW).append("], ");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("], ");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("] }");
    }
    //</editor-fold>
}

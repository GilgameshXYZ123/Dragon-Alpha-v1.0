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
public abstract class AdaptivePool2D extends SimpleFunction{
    
    protected int OH, OW;
    protected int FH, FW;
    protected int sh, sw;
    protected int IH, IW;
    
    protected AdaptivePool2D(int out_height, int out_width) {
        if(out_height <= 0) throw new IllegalArgumentException("Output height must > 0");
        if(out_width  <= 0) throw new IllegalArgumentException("Output width must > 0");
        
        this.OH = out_height;
        this.OW = out_width;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public int[] out_size() {return new int[]{ OH, OW };}
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ output_size = [").append(OH).append(", ").append(OW).append("] }");
    }
    
    protected void __adaptive__(Tensor X) {
        int[] dim = X.dim(); //X[N, IH, IW, IC]
        if(dim.length != 4) throw new IllegalArgumentException("Tensor X.ndim must != 4");
        
        IH = dim[1]; sh = Math.floorDiv(IH, OH); FH = IH - (OH - 1)*sh;
        IW = dim[2]; sw = Math.floorDiv(IW, OW); FW = IW - (OW - 1)*sw;
    }
    //</editor-fold>
}

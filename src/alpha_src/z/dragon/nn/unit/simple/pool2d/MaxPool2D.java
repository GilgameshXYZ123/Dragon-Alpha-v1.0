/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.pool2d;

import z.dragon.engine.Tensor;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.Train2Eval;

/**
 *
 * @author Gilgamesh
 */
public class MaxPool2D extends Pool2D implements Train2Eval
{
    protected Tensor Index;//Tensor<int32>
    protected int IH, IW;
    
    public MaxPool2D(
            int kernel_height, int kernel_width,
            int step_height, int step_width,
            int padding_height, int padding_width) 
    {
        super(kernel_height, kernel_width, 
              step_height, step_width,
              padding_height, padding_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append("], ");
        sb.append(", stride = [").append(sh).append(", ").append(sw).append("], ");
        sb.append(", padding = [").append(ph).append(", ").append(pw).append("] }");
    }
    
    public Tensor Index() { return Index; }
    
    @Override
    public void vars(TensorSet set) {
        super.vars(set);
        set.add(Index);
    }

    @Override
    public void gc() { 
        super.gc(); 
        eg.delete(Index); Index = null;
    }
    
    protected boolean training = true;
    @Override public boolean training() { return training; } 
    @Override public MaxPool2D train() { this.training = true; return this; }
    @Override public MaxPool2D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Tensor X) {
        if(!training) return eg.pool2D_max(X, FH, FW, sh, sw, ph, pw);
        
        //when training, we need to compute the Index for mapping: X -> Y
        int[] dim = X.dim(); IH = dim[1]; IW = dim[2];

        Tensor[] result = eg.pool2D_max_indexed(X, FH, FW, sh, sw, ph, pw);
        Index = result[1];
        return result[0];//result[0] = deltaY
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.upool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, sh, sw, ph, pw);
        
        //when deltaX is cauculated, deltaY & Index is not needed 
        if(grad_inplace) deltaX.dual(()-> { deltaY.delete(); Index.delete(); });
        else deltaX.dual(()-> { Index.delete(); });
            
        return deltaX;
    }
    //</editor-fold>
}

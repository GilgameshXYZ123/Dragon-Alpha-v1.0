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
public class AdaptiveMaxPool2D extends AdaptivePool2D
        implements Train2Eval
{
    protected Tensor Index;//Tensor<int32>
    
    public AdaptiveMaxPool2D(int out_height, int out_width) {
        super(out_height, out_width);
    }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { training = ").append(training);
        sb.append(", out_size = [").append(OH).append(", ").append(OW).append("] }");
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
    @Override public AdaptiveMaxPool2D train() { this.training = true; return this; }
    @Override public AdaptiveMaxPool2D eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running area: propagation">
    @Override
    protected Tensor __forward__(Tensor X) {
       __adaptive__(X);
       
       if(!training) return eg.pool2D_max(X, FH, FW, sh, sw, 0, 0);
       
        //when training, we need to compute the Index for mapping: X -> Y
        Tensor[] result = eg.pool2D_max_indexed(X, FH, FW, sh, sw, 0, 0);
        Index = result[1];
        return result[0];//result[0] = deltaY
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.upool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, sh, sw, 0, 0);
        
        //when deltaX is cauculated, deltaY & Index are not needed
        if(grad_inplace) deltaX.dual(()-> { deltaY.delete(); Index.delete(); });
        else deltaX.dual(()-> { Index.delete(); });
             
        return deltaX;
    }
    //</editor-fold>
}

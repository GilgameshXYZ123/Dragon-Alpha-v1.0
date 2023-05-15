/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.math2.SimpleInplaceFunction;
import z.util.math.vector.Vector;


/**
 *
 * @author Gilgamesh
 */
public class Reshape extends SimpleInplaceFunction<Reshape>
{
    private int[] inDim;
    private int[] outDim;
    
    public Reshape(boolean inplace, int...outDim){ 
        super(inplace); 
        output_dim(outDim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int[] in_dim() { return inDim; }
    public int[] out_dim() { return outDim; }
    
    public final Reshape output_dim(int...outDim) {//null: the same as flaten
        this.outDim = (outDim == null || outDim.length == 0 ? null : outDim);
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", outDim = [");Vector.append(sb, outDim); sb.append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        inDim = X.dim();
        int[] odim = (outDim == null?
                new int[]{ X.dim(0), X.length() / X.dim(0) }://the same as flaten
                Vector.arrayCopy(outDim));
        
        return eg.reshape(inplace, X, odim);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads?
                eg.reshape(grad_inplace, deltaY, inDim):
                null);
    }
    //</editor-fold>
}

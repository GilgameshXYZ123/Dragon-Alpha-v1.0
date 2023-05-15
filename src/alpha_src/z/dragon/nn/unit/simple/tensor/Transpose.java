/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.math2.SimpleInplaceFunction;

/**
 *
 * @author Gilgamesh
 */
public class Transpose extends SimpleInplaceFunction<Transpose>
{
    private int idx1;//dim Idx1
    private int idx2;//dim Idx2
    
    public Transpose(boolean inplace, int dimIdx1, int dimIdx2) {
        super(inplace);
        this.idx1 = dimIdx1;
        this.idx2 = dimIdx2;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int dimIdx1() { return idx1; }
    public Transpose dimIdx1(int dimIdx1) {  this.idx1 = dimIdx1; return this; }
    
    public int dimIdx2() { return idx2; }
    public Transpose setDimIdx2(int dimIdx2) { this.idx2 = dimIdx2; return this; }
    
    public int[] dimIdx() { return new int[]{ idx1, idx2 }; }
    public Transpose dimIdx(int dimIdx1, int dimIdx2) {
        this.idx1 = dimIdx1;
        this.idx2 = dimIdx2;
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", [dimIdx1, dimIdx2] = [").append(idx1).append(", ").append(idx2) .append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.transpose(inplace, X, idx1, idx2);
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        return (backward_grads? 
                eg.transpose(grad_inplace, deltaY, idx1, idx2):
                null);
    }
    //</editor-fold>
}

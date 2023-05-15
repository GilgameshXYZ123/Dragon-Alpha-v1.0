/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class Concat extends ReduceFunction
{
    private int dimIdx;
    private int[] section;
    
    public Concat(int dimIdx) {
        this.dimIdx = dimIdx;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int dimIdx() {return dimIdx;}
    public Concat dimIdx(int dimIndex) { this.dimIdx = dimIndex; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(dimIdx).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor[] X) {
        Tensor Y = eg.concat(dimIdx, X);
        section = new int[X.length];
        
        for(int i=0; i<X.length; i++) section[i] = X[i].dim(dimIdx);
        
        return Y;
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num) {
        return eg.split(deltaY, dimIdx, section);
    }
    //</editor-fold>
}

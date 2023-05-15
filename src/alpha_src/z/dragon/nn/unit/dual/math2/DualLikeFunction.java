/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math2;

import z.dragon.nn.unit.dual.math1.DualFunction;
import z.dragon.engine.Tensor;


/**
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class DualLikeFunction<T extends DualLikeFunction<?>>
        extends DualFunction 
{
    private boolean likeX1;
    
    protected DualLikeFunction(boolean likeX1) {
        this.likeX1 = likeX1;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean likeX1() { return likeX1; }
    public T likeX1(boolean flag) { likeX1 = flag; return (T) this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    protected abstract Tensor __forward__(Tensor X1, Tensor X2, boolean likeX1); 
    
    @Override 
    protected final Tensor __forward__(Tensor X1, Tensor X2) {
        return __forward__(X1, X2, likeX1);
    }
    //</editor-fold>
}

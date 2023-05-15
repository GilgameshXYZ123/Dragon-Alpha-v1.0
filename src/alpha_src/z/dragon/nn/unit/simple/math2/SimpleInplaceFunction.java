/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.nn.unit.simple.math1.SimpleFunction;
import z.dragon.engine.Tensor;


/**
 * @author Gilgamesh
 * @param <T>
 */
@SuppressWarnings("unchecked")
public abstract class SimpleInplaceFunction<T extends SimpleInplaceFunction<?>> 
        extends SimpleFunction
{
    private boolean inplace;
    
    protected SimpleInplaceFunction(boolean inplace) {
        this.inplace = inplace;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace()).append(" }");
    }
 
    public boolean inplace() { return inplace; }
    public T inplace(boolean inplace) { this.inplace = inplace; return (T) this; }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area">
    protected abstract Tensor __forward__(Tensor X, boolean inplace);
    
    @Override
    protected final Tensor __forward__(Tensor X) {
        return __forward__(X, inplace).modify(inplace);
    }
    //</editor-fold>
}

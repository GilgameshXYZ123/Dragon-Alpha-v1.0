/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public abstract class FurcationFunction extends Furcation
{
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override protected void __init__(Engine eg) {}
    
    @Override public void params(ParamSet set) {}
    @Override public void param_map(ParamMap<String> set) {}
    
    @Override public void state(State dic) {}
    @Override public void update_state(State dic, boolean partial) {}
    
    protected abstract Tensor __backward__(Tensor[] deltaY);
    
    @Override
    protected Tensor __backward__(Tensor[] deltaY, 
            boolean[] grad_inplace, boolean backward_grads)
    {
        if(!backward_grads) return null;
        
        Tensor deltaX = __backward__(deltaY);
        
        boolean flag = false;
        for(boolean gip : grad_inplace) if(gip) { flag = true; break; }
        
        if(flag) {
            deltaX.dual(()->{ 
                for(int i=0; i<grad_inplace.length; i++)
                    if(grad_inplace[i]) deltaY[i].delete();
            });
        }
        
        return deltaX;
    }
    //</editor-fold>
}

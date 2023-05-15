/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;

/**
 *
 * @author Gilgamesh
 */
public abstract class ReduceFunction extends Reducer
{ 
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override protected void __init__(Engine eg) {}
    
    @Override public void params(ParamSet set) {}
    @Override public void param_map(ParamMap<String> map) {}
    
    @Override public void state(State dic) {}
    @Override public void update_state(State dic, boolean partial) {}
    
    protected abstract Tensor[] __backward__(Tensor deltaY, int input_tensor_num);
    
    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num,
            boolean grad_inplace, boolean backward_grads)
    {
        if(!backward_grads) return null;
        
        Tensor[] deltaX = __backward__(deltaY, input_tensor_num);

        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            CountGc gc = new CountGc(input_tensor_num, deltaY);
            for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class SGD extends Optimizer 
{
    public SGD(Collection<Parameter> params, float lr) {
        super(params, lr); 
    }
    
     public SGD(Map<String, Parameter> param_map, float lr) {
        super(param_map, lr); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override public SGD learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    @Override 
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr).append(" }");
    }
    
    @Override protected void hypher_state(State dic) {}
    @Override protected void param_state(State dic, int index, String paramName) {}
    @Override protected void update_hypher_state(State dic, boolean partial) {}
    @Override protected void update_param_state(State dic, boolean partial, int index, String paramName) {}
    
    @Override  protected void __before_update__() {}
    
    @Override
    protected void __update__(int index, Tensor gradient, Engine eg) {
        eg.sgd(params[index].ts(),//inplace: param.datas
                gradient, lr);
    }

    @Override
    protected void __update__(int index, Collection<Tensor> gradients, Engine eg) {
        eg.sgd(params[index].ts(),//inplace: param.datas
                gradients, lr);
    }
    
    @Override public void __clear__() {}
    //</editor-fold>
}

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
public class SGDMN extends Optimizer
{
    protected float momentum, dampen, nesterov;
    protected Tensor[] V;
    
    protected float L1coef = 0.0f;
    protected float L2coef = 0.0f;

    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float momentum, float dampen, float nestrov) {
        this.momentum = momentum;
        this.dampen = dampen;
        this.nesterov = nestrov;
        V = Tensor.zero_like(params);
        Tensor.sync(V);
    }
    //</editor-fold>
    public SGDMN(Parameter[] params, float lr, float momentum, float dampen, float nesterov) {
        super(params, lr);
        __init__(momentum, dampen, nesterov);
    }

    public SGDMN(Collection<Parameter> params, float lr, float momentum, float dampen, float nesterov) {
        super(params, lr);
        __init__(momentum, dampen, nesterov);
    }

    public SGDMN(Map<String, Parameter> param_map, float lr, float momentum, float dampen, float nesterov) {
        super(param_map, lr);
        __init__(momentum, dampen, nesterov);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public SGDMN learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] veclocity() { return V; }
    
    public float momentum() { return momentum; }
    public SGDMN momentum(float momentum) { this.momentum = momentum; return this;}
    
    public float dampen() { return dampen; }
    public SGDMN dampen(float dampen) { this.dampen = dampen; return this; }
    
    public float nestorv() { return nesterov; }
    public SGDMN nestorv(float nestrov) { this.nesterov = nestrov; return this; }
    public SGDMN nestrov(boolean flag) { nesterov = (flag? 1 : 0); return this; }
    
    public float L1coef() { return L1coef; }
    public SGDMN L1coef(float L1coef) { this.L1coef = L1coef; return this; }
    
    public float L2cof() { return L2coef; }
    public SGDMN L2coef(float L2coef) { this.L2coef = L2coef; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{  learning_rate = ").append(lr);
        sb.append(", momentum = ").append(momentum);
        sb.append(", dampen = ").append(dampen);
        sb.append(", nestrov = ").append(nesterov);
        sb.append(", [L1coef, L2coef] = [")
                .append(L1coef).append(", ")
                .append(L2coef).append("] }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hyper state---------------------------------------------------------------
    @Override protected void hypher_state(State dic) {}
    @Override protected void update_hypher_state(State dic, boolean partial) { }

    //param state---------------------------------------------------------------
    protected String velocity_key(String param_name) { return param_name + ".velocity"; }
    
    @Override
    protected void param_state(State dic, int index, String paramName) {
        dic.put(paramName + ".velocity", V[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String velocity_key = velocity_key(param_name);
        V[index].set(dic.get(velocity_key), partial, "fail to load: " + velocity_key);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override protected void __before_update__() {}

    @Override
    protected void __update__(int index, Tensor gradient, Engine eg) {
        if(L1coef !=0 || L2coef != 0) {
            eg.sgdmn(params[index].ts(),//inplace: param.data
                    V[index], momentum, dampen, nesterov,
                    gradient, lr, 
                    L1coef, L2coef);
        }
        
        eg.sgdmn(params[index].ts(),//inplace: param.datas
                V[index], momentum, dampen, nesterov,
                gradient, lr);
    }

    @Override
    protected void __update__(int index, Collection<Tensor> gradients, Engine eg) {
        if(L1coef !=0 || L2coef != 0) {
            eg.sgdmn(params[index].ts(),//inplace: param.data
                    V[index], momentum, dampen, nesterov,
                    gradients, lr,
                    L1coef, L2coef);
        }
        
        eg.sgdmn(params[index].ts(),//inplace: param.data
                V[index], momentum, dampen, nesterov, 
                gradients, lr);
    }

    @Override
    protected void __clear__() {
        Tensor.delete(V); V = null;
    }
    //</editor-fold>
}

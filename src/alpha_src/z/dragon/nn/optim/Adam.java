/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 * V' = V / (1 - beta1^t)
 * S' = S / (1 - beta2^t)
 * W = deltaW - lr * V' / [sqrt(S') + eps]
 * W = deltaW - lr * V / (1 - beta1^t) / [sqrt(S / (1 - beta2^t)) + eps]
 * W = deltaW  - lr * [sqrt(1 - beta2^t) / (1 - beta1^t)] * V / [sqrt(S) + sqrt(1 - beta2^t)*eps]
 * let: lr_t  = lr * [sqrt(1 - beta2^t) / (1 - beta1^t)]
 *       eps_t = eps * sqrt(1 - beta2^t)
 * W = deltaW - lr_t * V / (sqrt(S) + eps_t).
 * @author Gilgamesh
 */
public class Adam extends Optimizer
{
    protected float lr_t, eps_t;
    
    protected float beta1, a1, a2, expBeta1;
    protected Tensor[] V;
     
    protected float beta2, eps, b1, b2, expBeta2;
    protected Tensor[] S;
    
    protected float L1coef = 0.0f;
    protected float L2coef = 0.0f;
    
    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta1, float beta2, float eps) {
        this.beta1 = beta1; 
        this.beta2 = beta2; this.eps = eps;
        
        V = Tensor.zero_like(params); expBeta1 = 1.0f; 
        S = Tensor.zero_like(params); expBeta2 = 1.0f;
        Tensor.sync(V); Tensor.sync(S);
    } 
    //</editor-fold>
    public Adam(Parameter[] params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adam(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adam(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
        super(param_map, lr);
        __init__(beta1, beta2, eps);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Adam learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    public Tensor[] exp_avg_sq() { return S; }
    
    public float beta1() {return beta1;}
    public Adam beta1(float beta1) { this.beta1 = beta1;  return this; }
    
    public float beta2() { return beta2; }
    public Adam beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() { return eps; }
    public Adam eps(float eps) { this.eps = eps; return this;}
    
    public float L1coef() { return L1coef; }
    public Adam L1coef(float L1coef) { this.L1coef = L1coef; return this; }
    
    public float L2coef() { return L2coef; }
    public Adam L2coef(float L2coef) { this.L2coef = L2coef; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr);
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("]");
        sb.append(", [L1coef, L2coef] = (")
                .append(L1coef).append(", ")
                .append(L2coef).append(") }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hypher-state--------------------------------------------------------------
    @Override
    protected void hypher_state(State dic) {
        dic.put("expBetas", State.floats(expBeta1, expBeta2));
    }
    
    @Override
    protected void update_hypher_state(State dic, boolean partial) {
        StateValue expBetas = dic.get("expBetas");
        State.set(expBetas, "fail to load expBetas", partial, ()->{
            float[] arr = Vector.toFloatVector(expBetas.toStringLines(), 2);
            expBeta1 = arr[0]; expBeta2 = arr[1];
        });
    }

    //param-state---------------------------------------------------------------
    protected String exp_avg_key(String param_name) { return param_name + ".exp_avg"; }
    protected String exp_avg_sq_key(String param_name) { return param_name + ".exp_avg_sq"; }
    
    @Override
    protected void param_state(State dic, int index, String param_name) {
        dic.put(exp_avg_key(param_name), V[index]);
        dic.put(exp_avg_sq_key(param_name), S[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String exp_avg_key = exp_avg_key(param_name);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load " + exp_avg_key);
        
        String exp_avg_sq_key = exp_avg_sq_key(param_name);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {
        expBeta1 *= beta1; expBeta2 *= beta2;//init: expBeta1 = expBeta2 = 1
        
        a1 = beta1; a2 = 1.0f - beta1;//exp_avg
        b1 = beta2; b2 = 1.0f - beta2;//exp_avg_sq
        
        double correct_beta2 =  Math.sqrt(1 - expBeta2);
        lr_t =  (float) (lr  * correct_beta2 / (1 - expBeta1));
        eps_t = (float) (eps * correct_beta2);
    }
    
    @Override
    protected void __update__(int index, Tensor gradient, Engine eg) {
        if(L1coef !=0 || L2coef != 0) {
            eg.adam(params[index].ts(),//inplace: param.datas
                    V[index], a1, a2,
                    S[index], b1, b2, eps_t,
                    gradient, lr_t,
                    L1coef, L2coef);
            return;
        }
        
        eg.adam(params[index].ts(),//inplace: param.data
                V[index], a1, a2,
                S[index], b1, b2, eps_t,
                gradient, lr_t);
    }
    
    @Override
    protected void __update__(int index, Collection<Tensor> gradients, Engine eg) {
        if(L1coef != 0 && L2coef != 0) {
            eg.adam(params[index].ts(),//inplace: param.data
                    V[index], a1, a2,
                    S[index], b1, b2, eps_t, 
                    gradients, lr_t, 
                    L1coef, L2coef);
            return;
        }
        
        eg.adam(params[index].ts(),//inplace: param.data
                V[index], a1, a2,
                S[index], b1, b2, eps_t,
                gradients, lr_t);
    }
    
    @Override
    protected void __clear__() {
        Tensor.delete(V); V = null;
        Tensor.delete(S); S = null;
    }
    //</editor-fold>
}
